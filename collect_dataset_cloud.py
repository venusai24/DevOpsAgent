import os
import sys
import json
import time
import subprocess
import argparse
import logging
import asyncio
from pathlib import Path

import urllib.request
import urllib.error

from kubernetes import client, config

from prometheus_anomaly import (
    BENCHMARK_PROFILE,
    PRODUCTION_PROFILE,
    PrometheusAnomalyEngine,
)

from clickhouse_log_client import query_forensic_logs_for_pod_async
from airs_v2.context.graph import ContextGraph
from live_harness import get_pod_errors, _binary_k8s_unhealthy_pods, inject_fault_sregym

try:
    from config import settings as _settings
    _CH_LOOKBACK = _settings.CLICKHOUSE_LOOKBACK_MINUTES
    _CH_MAX_ROWS = _settings.CLICKHOUSE_MAX_ROWS
except Exception:
    _CH_LOOKBACK = 60
    _CH_MAX_ROWS = 200

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pre-flight Checks
def pre_flight_checks(prometheus_url: str):
    print("🚀 Running Pre-Flight Checks...")
    
    # 1. Check K8s connection
    try:
        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()
        v1 = client.CoreV1Api()
        v1.list_namespace()
        print("✅ Kubernetes API connected.")
    except Exception as e:
        print(f"❌ K8s API connection failed: {e}")
        sys.exit(1)

    # 2. Check Prometheus
    print(f"🔍 Checking Prometheus at {prometheus_url}...")
    try:
        req = urllib.request.Request(f"{prometheus_url}/api/v1/query?query=up")
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status != 200:
                raise Exception(f"HTTP {response.status}")
        print("✅ Prometheus is reachable.")
    except Exception as e:
        print(f"❌ Prometheus connection failed: {e}")
        print("Please ensure Prometheus is running on the Telemetry VM and the URL is correct.")
        sys.exit(1)

    # 3. Check Observability Pods
    print("🔍 Checking observability pipeline pods...")
    try:
        pods = v1.list_namespaced_pod(namespace="observability")
        if not pods.items:
            print("⚠️ No pods found in 'observability' namespace. Is the pipeline deployed?")
            sys.exit(1)
        
        for pod in pods.items:
            # We skipairs-ml-sidecar if it's pending since it's not strictly required for log collection,
            # but Vector and ClickHouse are critical.
            if "vector" in pod.metadata.name or "clickhouse" in pod.metadata.name:
                if pod.status.phase not in ["Running", "Succeeded"]:
                    print(f"❌ Pod {pod.metadata.name} is not Running (Status: {pod.status.phase}).")
                    print("Please ensure the observability pipeline is fully healthy before collecting data.")
                    sys.exit(1)
        print("✅ Core observability pods are Running.")
    except Exception as e:
        print(f"❌ Failed to check observability pods: {e}")
        sys.exit(1)

    print("✅ All pre-flight checks passed!\n")
    return v1

async def collect_scenario_data(scenario, args, v1):
    namespace = args.namespace
    fault_id = scenario['fault_id']
    
    # 1. Inject Fault
    if not inject_fault_sregym(
        fault_id,
        duration=args.duration,
        tput=args.tput,
        multiplier=args.multiplier,
        telemetry_endpoint=args.telemetry_endpoint
    ):
        print(f"❌ Skipping {fault_id} due to injection failure.")
        return False

    wait_time = 300
    print(f"⏳ Waiting {wait_time}s for symptoms and logs to accumulate...")
    time.sleep(wait_time)

    print(f"🔍 Fetching pods in namespace: '{namespace}'...")
    pods = v1.list_namespaced_pod(namespace=namespace)

    # 2. Get Prometheus Anomalies
    profile = BENCHMARK_PROFILE if args.zscore_mode == "benchmark" else PRODUCTION_PROFILE
    engine = PrometheusAnomalyEngine(prom_url=args.prometheus_url, profile=profile, v1=v1, namespace=namespace)
    
    anomalous_pods = []
    if engine.is_available:
        anomalous_pods = engine.detect_anomalous_pods()
        print(f"⚠️  Prometheus detected {len(anomalous_pods)} anomalous pod(s).")
    else:
        print("⚠️  Prometheus connection failed internally.")

    # 3. K8s Binary Checks
    fallback_pods = _binary_k8s_unhealthy_pods(pods)
    if fallback_pods:
        print(f"⚠️  Found {len(fallback_pods)} unhealthy pod(s) via binary checks.")

    if not anomalous_pods and not fallback_pods:
        print("❌ All pods healthy. SREGym fault didn't trigger any symptoms. Skipping.")
        return False

    # 4. Extract Logs
    combined_errors = []
    focal_services = []
    processed_pods = set()

    if anomalous_pods:
        anomalous_pods.sort(key=lambda x: x.z_score, reverse=True)

    def _get_svc_name(pod_obj, default_name):
        return (
            pod_obj.metadata.labels.get("app")
            or pod_obj.metadata.labels.get("io.kompose.service")
            or pod_obj.metadata.labels.get("app.kubernetes.io/name")
            or default_name.split("-")[0]
        )

    async def _extract_and_append(pod_name, pod_ns, container, svc_name, is_k8s_api_source=False):
        forensic_log = await query_forensic_logs_for_pod_async(
            namespace=pod_ns, pod_name=pod_name, container_name=container,
            lookback_minutes=_CH_LOOKBACK, limit=_CH_MAX_ROWS,
            k8s_v1_client=v1, k8s_fallback_tail_lines=50
        )
        if not forensic_log or not forensic_log.strip():
            return
        for line in forensic_log.splitlines():
            if not line.strip(): continue
            if is_k8s_api_source or "[K8s API Fallback]" in forensic_log:
                lower_line = line.lower()
                if ("info" in lower_line or "debug" in lower_line) and not any(
                    kw in lower_line for kw in ("error", "fail", "fatal", "exception", "panic", "warn", "timeout", "refused", "back-off", "oom", "unreachable")
                ):
                    continue
            combined_errors.append(f"ERROR: [Pod {pod_name}] {line}")

    for ap in anomalous_pods:
        pod_name = ap.pod_name
        if pod_name in processed_pods: continue
        processed_pods.add(pod_name)
        pod_obj = None
        try:
            pod_obj = v1.read_namespaced_pod(name=pod_name, namespace=ap.namespace)
            svc_name = _get_svc_name(pod_obj, pod_name)
        except Exception:
            svc_name = ap.service_name

        if svc_name not in focal_services: focal_services.append(svc_name)
        await _extract_and_append(pod_name, ap.namespace, ap.container, svc_name)
        if pod_obj: combined_errors.extend(get_pod_errors(pod_obj, v1))

    for pod in fallback_pods:
        pod_name = pod.metadata.name
        if pod_name in processed_pods: continue
        processed_pods.add(pod_name)
        svc_name = _get_svc_name(pod, pod_name)
        if svc_name not in focal_services: focal_services.append(svc_name)
        await _extract_and_append(pod_name, namespace, None, svc_name)
        combined_errors.extend(get_pod_errors(pod, v1))

    full_error_text = "\n".join(combined_errors)
    focal_service = focal_services[0] if focal_services else "unknown"

    # 5. Capture Live Topology Snapshot
    print("📸 Capturing live K8s ContextGraph snapshot...")
    os.environ["AIRS_USE_K8S"] = "true"
    os.environ["AIRS_K8S_NAMESPACES"] = namespace
    g = ContextGraph()
    snapshot = g.export_snapshot()

    # 6. Save Dataset
    out_dir = Path("offline_datasets")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{fault_id}.json"
    
    dataset = {
        "fault_id": fault_id,
        "focal_service": focal_service,
        "full_error_text": full_error_text,
        "topology_snapshot": snapshot,
        "scenario_metadata": scenario
    }
    
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"💾 Saved scenario data to {out_path}")
    return True

async def main():
    parser = argparse.ArgumentParser(description="Cloud Data Collection for AIRS")
    parser.add_argument("--duration", type=str, help="Duration for load generator (e.g., 600s)")
    parser.add_argument("--tput", type=int, help="Throughput for load generator")
    parser.add_argument("--multiplier", type=int, help="Multiplier for load generator")
    parser.add_argument("--telemetry-endpoint", type=str, help="Prometheus remote write endpoint")
    parser.add_argument("--prometheus-url", type=str, default=os.getenv("PROMETHEUS_URL", "http://localhost:9090"))
    parser.add_argument("--zscore-mode", type=str, default="benchmark")
    parser.add_argument("--namespace", type=str, default="auto")
    args = parser.parse_args()

    v1 = pre_flight_checks(args.prometheus_url)

    if args.namespace == "auto":
        for ns in ["hotel-reservation", "astronomy-shop", "social-network", "blueprint-hotel-reservation"]:
            try:
                if v1.list_namespaced_pod(namespace=ns).items:
                    args.namespace = ns
                    print(f"✅ Auto-detected active namespace: '{args.namespace}'")
                    break
            except Exception: pass

    with open('sregym_advanced_scenarios.json', 'r') as f:
        scenarios = json.load(f)

    for scenario in scenarios:
        print(f"\n{'='*80}\n--- Collecting Data for Scenario: {scenario['fault_id']} ---\n{'='*80}")
        await collect_scenario_data(scenario, args, v1)

        print("🧹 Cleaning up namespace to reset state...")
        subprocess.run(
            ["kubectl", "delete", "namespace", "blueprint-hotel-reservation", "hotel-reservation", "social-network", "astronomy-shop", "--ignore-not-found"],
            check=False, timeout=300
        )
        time.sleep(2)

    print("\n🎉 All data collection complete. You can now run `evaluate_rag_offline.py` on your laptop!")

if __name__ == "__main__":
    asyncio.run(main())
