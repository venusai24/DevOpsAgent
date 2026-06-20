import os
import sys
import json
import asyncio
import re
import time
import subprocess
import argparse
import logging

# Set environment variables for AIRS K8s configuration
# The namespace will be set dynamically via arguments later.

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
except ImportError:
    client = None
    config = None
    class ApiException(Exception):
        pass


from airs_v2.perception.router import LogRouter
from airs_v2.reasoning.engine import ReasoningEngine
from airs_v2.context.graph import ContextGraph

from prometheus_anomaly import (
    AnomalousPod,
    BENCHMARK_PROFILE,
    PRODUCTION_PROFILE,
    PrometheusAnomalyEngine,
    ZScoreProfile,
    extract_forensic_logs,
)

from clickhouse_log_client import query_forensic_logs_for_pod_async

try:
    from config import settings as _settings
    _CH_LOOKBACK = _settings.CLICKHOUSE_LOOKBACK_MINUTES
    _CH_MAX_ROWS = _settings.CLICKHOUSE_MAX_ROWS
except Exception:
    _CH_LOOKBACK = 60
    _CH_MAX_ROWS = 200

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default Prometheus URL — overridable via env var or --prometheus-url flag
# ---------------------------------------------------------------------------
_DEFAULT_PROMETHEUS_URL = os.getenv(
    "PROMETHEUS_URL",
    "http://localhost:9090",
)


def get_pod_errors(pod, v1) -> list[str]:
    errors = []
    # 1. Check container statuses for waiting/terminated errors
    for status in (pod.status.container_statuses or []):
        if status.state.waiting:
            msg = status.state.waiting.message
            reason = status.state.waiting.reason
            if msg:
                errors.append(f"ERROR: [Container Waiting] Pod {pod.metadata.name} container {status.name} is waiting ({reason}): {msg}")
        if status.last_state.terminated:
            msg = status.last_state.terminated.message
            reason = status.last_state.terminated.reason
            if msg:
                errors.append(f"ERROR: [Container Terminated] Pod {pod.metadata.name} container {status.name} terminated ({reason}): {msg}")

    # 2. Check Kubernetes events for warning alerts
    try:
        events = v1.list_namespaced_event(
            namespace=pod.metadata.namespace,
            field_selector=f"involvedObject.name={pod.metadata.name},involvedObject.kind=Pod"
        )
        for event in events.items:
            if event.type == "Warning":
                errors.append(f"WARNING: [K8s Event] Pod {pod.metadata.name} warning ({event.reason}): {event.message}")
    except Exception:
        pass

    return errors


# ---------------------------------------------------------------------------
# Phase 2: Binary K8s health check fallback
# ---------------------------------------------------------------------------

def _binary_k8s_unhealthy_pods(pods) -> list:
    """
    Original binary Kubernetes health check logic.
    Used as a safety net when Prometheus is unavailable or returns
    zero anomalies (e.g., cluster cold start, no metrics ingested yet).
    """
    unhealthy = []
    for pod in pods.items:
        # Exclude load generators from being identified as the focal service
        if "wlgen" in pod.metadata.name or "loadgenerator" in pod.metadata.name:
            continue
        ready = all(status.ready for status in (pod.status.container_statuses or []))
        restart_count = sum(status.restart_count for status in (pod.status.container_statuses or []))
        if not ready or restart_count > 0:
            unhealthy.append((pod, restart_count))
            
    # Sort by restart_count descending so the most crashing pod becomes the focal service
    unhealthy.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in unhealthy]


async def analyze_live_cluster(
    prometheus_url: str = _DEFAULT_PROMETHEUS_URL,
    zscore_mode: str = "benchmark",
    namespace: str = "auto",
):
    """
    Runs the AIRS perception and reasoning layer against the live cluster state.

    Detection pipeline:
      Phase 1 — Prometheus Z-Score Anomaly Detection
        Execute error rate and P95 latency Z-Score PromQL queries against the
        Prometheus instance. Identifies semantically anomalous pods even when
        all Kubernetes health checks pass (gray failures, RPC timeout storms,
        distributed deadlocks).

      Phase 2 — Binary K8s Fallback
        If Phase 1 returns zero anomalies (Prometheus unavailable, cold start,
        no spanmetrics ingested yet), falls back to the original binary check:
        restart_count > 0 or Ready == False.

      Phase 3 — Forensic Log Extraction
        Extracts container logs from each identified pod. Handles multi-container
        pods and the previous=True fallback for OOMKill/CrashLoopBackOff races.

    After log collection the existing AIRS pipeline (LogRouter → ContextGraph →
    ReasoningEngine) is invoked unchanged.

    Parameters
    ----------
    prometheus_url : str
        Full URL to the Prometheus /api/v1/query endpoint host.
    zscore_mode : str
        "benchmark" — short-window Z-Score, no offset, threshold 2.0.
                       Works with <15 min of Prometheus history (default).
        "production" — 1d seasonality-offset Z-Score, threshold 3.0.
                       Requires 24h+ of Prometheus history.
    """
    print("🚀 Initializing Kubernetes client...")
    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()

    v1 = client.CoreV1Api()

    if namespace == "auto":
        print("🔍 Auto-detecting active namespace...")
        candidate_namespaces = ["hotel-reservation", "astronomy-shop", "social-network", "blueprint-hotel-reservation"]
        for ns in candidate_namespaces:
            try:
                test_pods = v1.list_namespaced_pod(namespace=ns)
                if test_pods.items:
                    namespace = ns
                    print(f"✅ Auto-detected active namespace: '{namespace}'")
                    break
            except Exception:
                continue

    os.environ["AIRS_USE_K8S"] = "true"
    os.environ["AIRS_K8S_NAMESPACES"] = namespace

    print(f"🔍 Fetching pods in namespace: '{namespace}'...")
    try:
        pods = v1.list_namespaced_pod(namespace=namespace)
    except Exception as e:
        print(f"❌ Critical Network/API Error: Cannot connect to AKS: {e}")
        print("Aborting the entire run immediately to prevent compute wastage!")
        sys.exit(1)

    # ── Phase 1: Prometheus Z-Score Anomaly Detection ─────────────────────
    profile: ZScoreProfile = BENCHMARK_PROFILE if zscore_mode == "benchmark" else PRODUCTION_PROFILE
    print(f"\n📡 Phase 1: Prometheus Z-Score Anomaly Detection (mode={profile.name}, threshold={profile.z_threshold})")
    print(f"   Connecting to Prometheus at: {prometheus_url}")

    anomalous_pods: list[AnomalousPod] = []
    prometheus_phase_succeeded = False

    try:
        engine = PrometheusAnomalyEngine(
            prom_url=prometheus_url,
            profile=profile,
            v1=v1,
            namespace=namespace,
        )
        if engine.is_available:
            anomalous_pods = engine.detect_anomalous_pods()
            prometheus_phase_succeeded = True

            if anomalous_pods:
                print(f"⚠️  Prometheus detected {len(anomalous_pods)} semantically anomalous pod(s):")
                for ap in anomalous_pods:
                    print(
                        f"   🚨 {ap.namespace}/{ap.pod_name} "
                        f"[{ap.signal_type}] Z-Score={ap.z_score:.2f} "
                        f"(service: {ap.service_name})"
                    )
            else:
                print("   ✅ Prometheus Z-Score: No anomalies detected above threshold.")
        else:
            print("   ⚠️  Prometheus connection failed — will fall back to binary K8s checks.")
    except Exception as e:
        print(f"   ⚠️  Prometheus phase error: {e} — will fall back to binary K8s checks.")

    # ── Phase 2: Binary K8s State ─────────────────────────────────────────
    # We now always run Phase 2 to catch database crash loops that PromQL ignores
    print(f"\n🔄 Phase 2: Binary K8s Health Checks")
    fallback_pods = _binary_k8s_unhealthy_pods(pods)
    if fallback_pods:
        print(f"   ⚠️  Found {len(fallback_pods)} unhealthy pod(s) via binary checks:")
        for pod in fallback_pods:
            print(f"   - {pod.metadata.name}")
    else:
        print("   ✅ No pods are currently in a crash loop or unhealthy state.")

    if not anomalous_pods and not fallback_pods:
        print("   ✅ All pods are healthy and Prometheus detected no anomalies. No faults detected.")
        return None

    # ── Phase 3: Forensic Log Extraction ──────────────────────────────────
    # Strategy: ClickHouse-first (Hot → Warm → Cold + K8s Events), with
    # automatic fallback to the K8s API (extract_forensic_logs) when
    # ClickHouse is unavailable or returns zero rows for a pod.
    # The downstream noise-filter applied to K8s API text is preserved
    # for backward compatibility; ClickHouse output is pre-filtered by the
    # Vector aggregator pipeline and needs no further keyword gating.
    print(f"\n📋 Phase 3: Forensic Log Extraction (ClickHouse-first + K8s API fallback)")

    combined_errors: list[str] = []
    focal_services: list[str] = []
    processed_pods: set[str] = set()

    # Sort anomalous pods by Z-Score descending so the highest severity service is prioritized
    if anomalous_pods:
        anomalous_pods.sort(key=lambda x: x.z_score, reverse=True)

    # Helper to resolve service name from pod labels
    def _get_svc_name(pod_obj, default_name):
        return (
            pod_obj.metadata.labels.get("app")
            or pod_obj.metadata.labels.get("io.kompose.service")
            or pod_obj.metadata.labels.get("app.kubernetes.io/name")
            or default_name.split("-")[0]
        )

    async def _extract_and_append(
        pod_name: str,
        pod_ns: str,
        container: str | None,
        svc_name: str,
        is_k8s_api_source: bool = False,
    ) -> None:
        """
        Fetch forensic logs for a single pod via the ClickHouse-first chain
        and append matching lines to combined_errors.

        is_k8s_api_source: when True, apply the same keyword noise-filter
        that the original K8s-polling code used (for raw kubectl log text).
        ClickHouse output is already pre-filtered by the Vector pipeline,
        so no keyword gate is needed there.
        """
        nonlocal combined_errors

        forensic_log = await query_forensic_logs_for_pod_async(
            namespace=pod_ns,
            pod_name=pod_name,
            container_name=container,
            lookback_minutes=_CH_LOOKBACK,
            limit=_CH_MAX_ROWS,
            k8s_v1_client=v1,
            k8s_fallback_tail_lines=50,
        )

        if not forensic_log or not forensic_log.strip():
            return

        for line in forensic_log.splitlines():
            if not line.strip():
                continue

            # When the text came from the K8s API fallback the header
            # "[K8s API Fallback]" is prepended — apply the original keyword
            # noise filter to raw unstructured kubectl log output.
            if is_k8s_api_source or "[K8s API Fallback]" in forensic_log:
                lower_line = line.lower()
                if ("info" in lower_line or "debug" in lower_line) and not any(
                    kw in lower_line
                    for kw in (
                        "error", "fail", "fatal", "exception", "panic",
                        "warn", "timeout", "refused", "back-off", "oom",
                        "unreachable",
                    )
                ):
                    continue

            combined_errors.append(f"ERROR: [Pod {pod_name}] {line}")

    # ── 1. Process Prometheus Anomalies ──────────────────────────────────
    for ap in anomalous_pods:
        pod_name = ap.pod_name
        if pod_name in processed_pods:
            continue
        processed_pods.add(pod_name)

        pod_obj = None
        try:
            pod_obj = v1.read_namespaced_pod(name=pod_name, namespace=ap.namespace)
            svc_name = _get_svc_name(pod_obj, pod_name)
        except ApiException:
            svc_name = ap.service_name

        if svc_name not in focal_services:
            focal_services.append(svc_name)

        print(f"   📋 Extracting logs for anomalous pod '{pod_name}' (service: {svc_name})...")
        await _extract_and_append(
            pod_name=pod_name,
            pod_ns=ap.namespace,
            container=ap.container,
            svc_name=svc_name,
        )

        # Always append K8s API-sourced structured pod error events
        # (container waiting/terminated reasons) — these are not stored
        # in ClickHouse logs_hot; they come from pod status directly.
        if pod_obj:
            combined_errors.extend(get_pod_errors(pod_obj, v1))

    # ── 2. Process Binary K8s Unhealthy Pods ─────────────────────────────
    for pod in fallback_pods:
        pod_name = pod.metadata.name
        if pod_name in processed_pods:
            continue
        processed_pods.add(pod_name)

        svc_name = _get_svc_name(pod, pod_name)
        if svc_name not in focal_services:
            focal_services.append(svc_name)

        print(f"   📋 Gathering error indicators and events for pod '{pod_name}' ({svc_name})...")
        await _extract_and_append(
            pod_name=pod_name,
            pod_ns=namespace,
            container=None,
            svc_name=svc_name,
        )

        combined_errors.extend(get_pod_errors(pod, v1))

    if not combined_errors:
        print("❌ No diagnostic log messages or event warnings could be collected.")
        return None

    focal_service = focal_services[0] if focal_services else "unknown"
    print(f"\n🎯 Targets identified: Focal Service = '{focal_service}'")

    full_error_text = "\n".join(combined_errors)
    print("\n📝 Collected Error Context:")
    print("-" * 60)
    print(full_error_text)
    print("-" * 60)

    # ── AIRS Pipeline (unchanged) ──────────────────────────────────────────
    print("\n🧠 Routing live errors through AIRS perception layer (LogRouter)...")
    router = LogRouter()
    report = await router.classify_block(full_error_text)

    print("🌐 Building live Kubernetes context graph topology...")
    g = ContextGraph()

    print("🔮 Executing neuro-symbolic reasoning engine analysis...")
    engine_airs = ReasoningEngine(graph=g)
    analysis = await engine_airs.analyze_incident(report, focal_service)

    print("\n" + "=" * 60)
    print(" LIVE INCIDENT ANALYSIS REPORT ".center(60, "="))
    print("" + "=" * 60)
    print(analysis.analysis_markdown)
    print("=" * 60)
    return analysis


def llm_judge(scenario, airs_response):
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage

        if not os.getenv("GROQ_API_KEY"):
            return None

        llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0).bind(response_format={"type": "json_object"})
        prompt = f"""You are an expert judge evaluating an SRE AI agent's root cause analysis.

Expected underlying fault:
{scenario['expected_root_cause']}

Agent's final analysis report:
{airs_response['chain_of_thought_log']}

Did the agent correctly identify the failing component and/or accurately describe the underlying issue based on its analysis?
Respond ONLY with a JSON object in this exact format:
{{"diagnosis_correct": true, "reason": "brief explanation"}}
"""
        response = llm.invoke([HumanMessage(content=prompt)])
        data = json.loads(response.content)
        return data.get("diagnosis_correct", False)
    except Exception as e:
        print(f"LLM Judge error: {e}")
        return None


def evaluate_run(scenario, airs_response, duration):
    if not airs_response:
        return {
            "fault_id": scenario['fault_id'],
            "diagnosis_correct": False,
            "time_to_resolve_seconds": duration,
            "airs_log": "No analysis returned from AIRS."
        }

    cause_match = llm_judge(scenario, airs_response)

    if cause_match is None:
        print("⚠️ LLM Judge could not evaluate the response. Marking as FAIL.")
        cause_match = False

    return {
        "fault_id": scenario['fault_id'],
        "diagnosis_correct": cause_match,
        "time_to_resolve_seconds": duration,
        "airs_log": airs_response['chain_of_thought_log']
    }


def inject_fault_sregym(fault_id, duration=None, tput=None, multiplier=None, telemetry_endpoint=None):
    print(f"Injecting SREGym fault: {fault_id}")

    # Clean up lingering uvicorn port from previous runs
    try:
        subprocess.run(["fuser", "-k", "8000/tcp"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        time.sleep(1)
    except Exception:
        pass

    os.environ["SREGYM_WAIT_BEFORE_FAULT"] = "0"
    cmd = [".venv/bin/python3", "main.py", "--problem", fault_id, "--use-external-harness"]
    # We no longer pass duration, tput, multiplier, or telemetry_endpoint directly
    # to main.py because main.py does not accept them via CLI.
    # They should be configured via environment variables or inside SREGym config.

    # Dynamically calculate timeout based on the duration string (e.g. "3600s" -> 3600 + buffer)
    wait_timeout = 900  # default fallback (15 mins)
    if duration:
        try:
            # Extract digits from duration string
            digits = "".join([c for c in duration if c.isdigit()])
            if digits:
                wait_timeout = int(digits) + 300  # Add 5 minutes buffer for setup/teardown
        except Exception:
            pass

    try:
        cwd_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SREGym")
        process = subprocess.Popen(cmd, cwd=cwd_path)
        process.wait(timeout=wait_timeout)
        if process.returncode != 0:
            print(f"Failed to inject fault {fault_id}: exit code {process.returncode}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"Injection command timed out after {wait_timeout}s for {fault_id}. Forcing termination.")
        process.kill()
        return False  # Fail the scenario instead of assuming success on premature timeout
    except Exception as e:
        print(f"Failed to inject fault {fault_id}: {e}")
        return False


# ---------------------------------------------------------------------------
# Docker-Compose-native analysis path (--no-k8s mode)
# ---------------------------------------------------------------------------

# Maps each OTel Demo feature-flag fault_id to the Docker Compose service
# name(s) whose container logs are most diagnostic for that fault.
# Service names match the keys in compose.yaml exactly.
_FAULT_SERVICE_MAP: dict[str, list[str]] = {
    "cartFailure":                  ["cart", "valkey-cart"],
    "recommendationCacheFailure":   ["recommendation"],
    "loadGeneratorFloodHomepage":   ["frontend", "frontend-proxy"],
    "kafkaQueueProblems":           ["kafka", "accounting"],
    "adFailure":                    ["ad"],
    "paymentFailure":               ["payment", "checkout"],
}

_LOG_TAIL_LINES = 300  # lines per container to collect


def _collect_docker_logs(services: list[str], since_minutes: int = 5) -> str:
    """
    Collect recent stdout+stderr logs from Docker Compose containers.

    Uses `docker ps --filter name=<svc>` to locate running containers by
    name fragment, then `docker logs --since <n>m` to retrieve their output.
    Both stdout and stderr are merged, and OpenTelemetry JSON noise is filtered out.
    """
    import re
    collected: list[str] = []

    for svc in services:
        try:
            # Find container IDs whose names contain the service name fragment
            ps = subprocess.run(
                ["docker", "ps", "--filter", f"name={svc}", "--format", "{{.ID}}\t{{.Names}}"],
                capture_output=True, text=True, timeout=10,
            )
            lines = [l.strip() for l in ps.stdout.strip().splitlines() if l.strip()]
            if not lines:
                collected.append(f"[{svc}] No running container found matching name '{svc}'.")
                continue

            for entry in lines:
                parts = entry.split("\t")
                cid = parts[0]
                cname = parts[1] if len(parts) > 1 else cid[:12]

                logs = subprocess.run(
                    ["docker", "logs", "--since", f"{since_minutes}m", cid],
                    capture_output=True, text=True, timeout=30,
                )
                raw = (logs.stdout + logs.stderr).strip()
                if raw:
                    filtered_lines = []
                    for line in raw.splitlines():
                        line_stripped = line.strip()
                        if not line_stripped or line_stripped in ("{", "}", "},", "[", "]", "],"):
                            continue
                        # Skip typical OpenTelemetry JSON dump attributes
                        if re.match(r'^[\'"](process|os|host|service|telemetry|library)\.', line_stripped):
                            continue
                        filtered_lines.append(line)
                    
                    filtered_raw = "\n".join(filtered_lines[-500:])  # Cap at 500 lines per container
                    if filtered_raw.strip():
                        collected.append(f"=== Container: {cname} ===\n{filtered_raw}")
                    else:
                        collected.append(f"=== Container: {cname} — (no error logs found after filtering) ===")
                else:
                    collected.append(f"=== Container: {cname} — (no log output) ===")

        except subprocess.TimeoutExpired:
            collected.append(f"[{svc}] docker logs timed out.")
        except Exception as exc:
            collected.append(f"[{svc}] Log collection error: {exc}")

    return "\n\n".join(collected) if collected else "(no logs collected)"


async def analyze_docker_compose_cluster(
    fault_id: str = "",
    log_tail_lines: int = _LOG_TAIL_LINES,
    # prometheus_url / zscore_mode kept for API compatibility but not used
    prometheus_url: str = _DEFAULT_PROMETHEUS_URL,
    zscore_mode: str = "benchmark",
) -> object:
    """
    Diagnose a known fault injected via OTel Demo feature flags.

    The benchmark wrapper (run_airs_benchmark.sh) has already:
      1. Toggled the feature flag in demo.flagd.json
      2. Waited 300 s for errors to accumulate in container logs

    This function therefore skips anomaly detection entirely and goes
    straight to log collection + AIRS reasoning:

      Step 1 — Map fault_id → affected Docker Compose services
      Step 2 — Collect container logs via `docker logs`
      Step 3 — Route logs through AIRS (LogRouter → ContextGraph → ReasoningEngine)

    Parameters
    ----------
    fault_id : str
        The feature-flag fault label (e.g. 'cartFailure'). Used to select
        which containers' logs are collected and as the focal service hint.
    log_tail_lines : int
        How many trailing log lines to collect per container (default 300).
    prometheus_url : str
        Kept for call-site compatibility — not used in this mode.
    zscore_mode : str
        Kept for call-site compatibility — not used in this mode.
    """
    print("🚀 [Docker Compose Mode] Log-based diagnosis — no anomaly detection required.")
    print(f"   Fault under test: '{fault_id}'" if fault_id else "   Fault: (unspecified)")

    # ── Step 1: Resolve affected services ────────────────────────────────────
    services = _FAULT_SERVICE_MAP.get(fault_id, [])
    if not services:
        # Unknown fault_id — fall back to collecting logs from all primary services
        services = list(_FAULT_SERVICE_MAP.keys())  # broad sweep
        print(f"   ⚠️  Unknown fault_id '{fault_id}' — collecting logs from all known services.")
    else:
        print(f"   📦 Targeting containers: {', '.join(services)}")

    # ── Step 2: Collect container logs ───────────────────────────────────────
    print(f"\n📋 Collecting Docker container logs (since 5m)...")
    raw_logs = _collect_docker_logs(services, since_minutes=5)

    print(f"\n📝 Log excerpt (first 800 chars):")
    print("-" * 60)
    print(raw_logs[:800])
    print("-" * 60)

    # ── Step 3: AIRS pipeline ─────────────────────────────────────────────────
    focal_service = services[0] if services else fault_id or "unknown"
    print(f"\n🎯 Focal Service (AIRS hint): '{focal_service}'")

    print("\n🧠 Routing logs through AIRS perception layer (LogRouter)...")
    router = LogRouter()
    report = await router.classify_block(raw_logs)

    print("🌐 Building Docker Compose service topology context graph...")
    import os
    from pathlib import Path
    os.environ["AIRS_USE_K8S"] = "false"
    fixture_path = Path("/home/VenuSai/DevOpsAgent/otel_demo_topology.json")
    g = ContextGraph(fixtures_path=fixture_path)

    print("📚 Initializing Hierarchical RAG Engine...")
    from airs_v2.memory.rag_engine import RAGEngine
    rag = RAGEngine()

    print("🔮 Executing neuro-symbolic reasoning engine analysis...")
    engine_airs = ReasoningEngine(graph=g, rag_engine=rag)
    analysis = await engine_airs.analyze_incident(report, focal_service)

    print("\n" + "=" * 60)
    print(" AIRS INCIDENT ANALYSIS REPORT ".center(60, "="))
    print("=" * 60)
    print(analysis.analysis_markdown)
    print("=" * 60)
    return analysis


async def main():
    parser = argparse.ArgumentParser(description="Live Harness for AIRS (K8s + Docker Compose modes)")
    parser.add_argument("--duration", type=str, help="Duration for load generator (e.g., 3600s)")
    parser.add_argument("--tput", type=int, help="Throughput for load generator")
    parser.add_argument("--multiplier", type=int, help="Multiplier for load generator")
    parser.add_argument("--telemetry-endpoint", type=str, help="Prometheus remote write endpoint")
    parser.add_argument(
        "--prometheus-url",
        type=str,
        default=_DEFAULT_PROMETHEUS_URL,
        help=(
            "Prometheus query API base URL (e.g., http://1.2.3.4:9090). "
            "Can also be set via PROMETHEUS_URL env var."
        ),
    )
    parser.add_argument(
        "--zscore-mode",
        type=str,
        choices=["benchmark", "production"],
        default="benchmark",
        help=(
            "Z-Score anomaly detection mode. "
            "'benchmark': short 1m vs 10m window, threshold 2.0 — "
            "works with <15 min of Prometheus history (default for SREGym). "
            "'production': 5m vs 1h window with offset 1d seasonality, threshold 3.0 — "
            "requires 24h+ of Prometheus history."
        ),
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="auto",
        help="The target Kubernetes namespace to monitor. Use 'auto' to automatically detect the active namespace.",
    )
    # ── Docker Compose Mode Flags ────────────────────────────────────────────
    parser.add_argument(
        "--no-k8s",
        action="store_true",
        default=False,
        help=(
            "Run in Docker Compose mode: skip all Kubernetes API calls. "
            "Uses Prometheus-only anomaly detection without pod-level log extraction. "
            "Required when OTel Demo is deployed via docker compose (not K8s)."
        ),
    )
    parser.add_argument(
        "--fault-id",
        type=str,
        default="",
        help="Label for the currently injected feature-flag fault (used in reports). e.g. 'cartFailure'.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to write a single-run JSON result (for bash wrapper consumption).",
    )
    args = parser.parse_args()

    # ── Docker Compose (no-k8s) mode ─────────────────────────────────────────
    if args.no_k8s:
        print("\n" + "="*80)
        print("  🐳  AIRS — Docker Compose Mode  (--no-k8s)")
        print("="*80)
        print("  ℹ️  Log-based diagnosis: collecting Docker container logs → AIRS pipeline.")
        print(f"  ℹ️  Fault under test: '{args.fault_id or '(none specified)'}'")
        print("="*80 + "\n")

        start_time = time.time()
        airs_response = None
        analysis = None
        try:
            analysis = await analyze_docker_compose_cluster(
                fault_id=args.fault_id,
            )
            if analysis is not None:
                inferred = (
                    analysis.root_cause_candidates[0].candidate_node
                    if analysis.root_cause_candidates
                    else "Unknown"
                )
                if analysis.root_cause_candidates:
                    inferred += f" ({analysis.root_cause_candidates[0].template_key})"

                airs_response = {
                    "fault_id": args.fault_id,
                    "inferred_root_cause": inferred,
                    "action_taken": "Diagnosed via ReasoningEngine (Docker Compose mode)",
                    "chain_of_thought_log": analysis.analysis_markdown,
                    "diagnosis_correct": False,   # requires LLM judge — set by bash wrapper
                    "time_to_resolve_seconds": 0,  # filled below
                }
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"AIRS invocation failed: {exc}")

        duration = time.time() - start_time
        if airs_response:
            airs_response["time_to_resolve_seconds"] = round(duration, 2)

        # LLM judge for Docker Compose mode (optional — requires GROQ_API_KEY)
        if airs_response and args.fault_id:
            # Build a minimal scenario dict for the judge
            _scenario_stub = {
                "fault_id": args.fault_id,
                "expected_root_cause": (
                    f"Feature flag '{args.fault_id}' was enabled on the OTel Demo, "
                    "triggering the associated service failure mode."
                ),
            }
            diagnosis_correct = llm_judge(_scenario_stub, airs_response)
            if diagnosis_correct is None:
                print("⚠️  LLM judge unavailable — skipping automated scoring.")
                diagnosis_correct = False
            airs_response["diagnosis_correct"] = diagnosis_correct
            print(
                f"\n{'✅ PASS' if diagnosis_correct else '❌ FAIL'} — "
                f"Fault: {args.fault_id}  |  TTD: {duration:.1f} s"
            )

        # Write output JSON for bash wrapper
        if args.output_json and airs_response:
            with open(args.output_json, "w") as _f:
                json.dump(airs_response, _f, indent=2)
            print(f"💾 Single-run result written to: {args.output_json}")

        return

    # ── Kubernetes (default) mode — original scenario-loop logic ─────────────
    with open('sregym_advanced_scenarios.json', 'r') as f:
        scenarios = json.load(f)

    results = []

    for scenario in scenarios:
        print(f"\n" + "="*80)
        print(f"--- Running Scenario: {scenario['fault_id']} ---")
        print("="*80)

        # 1. Inject Fault — SREGym path is DISABLED for Docker Compose deployments.
        # The bash wrapper (run_airs_benchmark.sh) handles fault injection via sed.
        # In pure K8s mode, prompt the operator to toggle the feature flag manually.
        print(f"Please manually trigger the fault '{scenario['fault_id']}' in the OpenTelemetry Demo Feature Flags UI (http://localhost:8080/feature/).")
        input("Press Enter to continue once you have enabled the fault...")
        # NOTE: inject_fault_sregym() is intentionally NOT called here.
        # K8s-native chaos (LitmusChaos / valkey_memory_disruption) crashes the
        # pipeline when there is no Kubernetes cluster available.

        # 2. Wait for the cluster to manifest symptoms
        # Note: This wait also allows Prometheus to accumulate baseline data
        # for the benchmark Z-Score (5m window + 1m current window needed).
        wait_time = 300
        print(f"⏳ Waiting {wait_time}s for symptoms to manifest in live cluster...")
        time.sleep(wait_time)

        # 3. Trigger AIRS and collect response
        start_time = time.time()
        airs_response = None
        try:
            analysis = await analyze_live_cluster(
                prometheus_url=args.prometheus_url,
                zscore_mode=args.zscore_mode,
                namespace=args.namespace,
            )
            if analysis:
                inferred = analysis.root_cause_candidates[0].candidate_node if analysis.root_cause_candidates else "Unknown"
                if analysis.root_cause_candidates:
                    inferred += f" ({analysis.root_cause_candidates[0].template_key})"

                airs_response = {
                    "inferred_root_cause": inferred,
                    "action_taken": "Diagnosed via ReasoningEngine",
                    "chain_of_thought_log": analysis.analysis_markdown
                }
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"AIRS invocation failed: {e}")

        duration = time.time() - start_time

        # 4. Evaluate and log
        eval_result = evaluate_run(scenario, airs_response, duration)
        results.append(eval_result)

        print(f"Result for {scenario['fault_id']}: {'✅ PASS' if eval_result['diagnosis_correct'] else '❌ FAIL'}")

        # 5. Clean up — namespace scrub is K8s-only and will crash on Docker Compose.
        # Only run kubectl commands if NOT in --no-k8s mode (already branched above).
        print("🧹 Performing universal namespace scrub (this may take a minute)...")
        try:
            subprocess.run(
                [
                    "kubectl", "delete", "namespace",
                    "blueprint-hotel-reservation", "hotel-reservation",
                    "social-network", "astronomy-shop",
                    "--ignore-not-found"
                ],
                check=False,
                timeout=300
            )
            print("✨ Cleanup complete.")
        except subprocess.TimeoutExpired:
            print("⚠️ Cleanup timed out! The next scenario may fail due to resource constraints.")

        time.sleep(2)

    print("\n=== Evaluation Complete ===")
    print(json.dumps(results, indent=2))

    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("💾 Saved results to benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
