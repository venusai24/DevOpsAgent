import os
import sys
import json
import time
import subprocess
import argparse
import logging
import asyncio
import threading
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
        pods = v1.list_namespaced_pod(namespace="observe")
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

    # 4. Check ClickHouse (Python Client)
    print("🔍 Checking ClickHouse connectivity from Python...")
    try:
        ch_host = os.getenv("CLICKHOUSE_HOST", "clickhouse.observe.svc.cluster.local")
        ch_port = os.getenv("CLICKHOUSE_PORT", "8123")
        req = urllib.request.Request(f"http://{ch_host}:{ch_port}/ping")
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status != 200:
                raise Exception(f"HTTP {response.status}")
        print("✅ ClickHouse is reachable from Python.")
    except Exception as e:
        print(f"❌ ClickHouse connection failed: {e}")
        print(f"   Attempted to reach: http://{ch_host}:{ch_port}")
        print("   If you are running outside the cluster, please start a port-forward:")
        print("   kubectl port-forward svc/clickhouse -n observe 8123:8123 &")
        print("   And prefix your command with CLICKHOUSE_HOST=localhost")
        sys.exit(1)

    print("✅ All pre-flight checks passed!\n")
    return v1

def health_monitor_thread(stop_event, prom_url, v1):
    while not stop_event.is_set():
        if stop_event.wait(60):
            break
        print(f"   [Monitor] Checking service health in background...")
        try:
            req = urllib.request.Request(f"{prom_url}/api/v1/query?query=up")
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status != 200: raise Exception("Prometheus HTTP check failed")
            
            pods = v1.list_namespaced_pod(namespace="observe")
            for pod in pods.items:
                if "vector" in pod.metadata.name or "clickhouse" in pod.metadata.name:
                    if pod.status.phase not in ["Running", "Succeeded"]:
                        raise Exception(f"Pod {pod.metadata.name} is not Running (Status: {pod.status.phase})")
                        
            ch_host = os.getenv("CLICKHOUSE_HOST", "clickhouse.observe.svc.cluster.local")
            ch_port = os.getenv("CLICKHOUSE_PORT", "8123")
            ch_req = urllib.request.Request(f"http://{ch_host}:{ch_port}/ping")
            with urllib.request.urlopen(ch_req, timeout=15) as ch_response:
                if ch_response.status != 200:
                    raise Exception(f"ClickHouse HTTP ping returned {ch_response.status}")
        except Exception as e:
            print(f"❌ Aborting collection! Service health check failed: {e}")
            os._exit(1)

async def collect_scenario_data(scenario, args, v1):
    namespace = args.namespace
    fault_id = scenario['fault_id']
    
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(target=health_monitor_thread, args=(stop_monitor, args.prometheus_url, v1), daemon=True)
    monitor_thread.start()

    try:
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
        await asyncio.sleep(wait_time)

        namespace = args.namespace
        if namespace == "auto":
            for ns in ["hotel-reservation", "astronomy-shop", "social-network", "blueprint-hotel-reservation"]:
                try:
                    if v1.list_namespaced_pod(namespace=ns).items:
                        namespace = ns
                        print(f"✅ Auto-detected active namespace for scenario {fault_id}: '{namespace}'")
                        break
                except Exception: pass

        print(f"🔍 Fetching pods in namespace: '{namespace}'...")
        pods = v1.list_namespaced_pod(namespace=namespace)
    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=2)

    # 2. Extract Logs from ALL pods (Bypassing Prometheus)
    print("⚠️ Bypassing Prometheus anomaly detection. Extracting logs for ALL pods in namespace...")
    combined_errors = []
    focal_services = []

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

    for pod in pods.items:
        pod_name = pod.metadata.name
        svc_name = _get_svc_name(pod, pod_name)
        if svc_name not in focal_services: focal_services.append(svc_name)
        await _extract_and_append(pod_name, namespace, None, svc_name)
        combined_errors.extend(get_pod_errors(pod, v1))

    if not combined_errors:
        print("❌ No errors found across any pods in the namespace! Skipping.")
        return False

    full_error_text = "\n".join(combined_errors)
    # Just set focal_service to the fault_id since we know what broke!
    focal_service = fault_id.split('_')[0]

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
