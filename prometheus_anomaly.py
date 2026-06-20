"""
prometheus_anomaly.py
=====================

Prometheus Z-Score Anomaly Detection Engine for the AIRS Live Harness.

Architecture
------------
The SREGym hotel-reservation workload (DeathStarBench) uses Jaeger client
libraries for tracing. Traces flow through the OTel Collector's spanmetrics
connector which converts spans into Prometheus-compatible metrics:

  - traces_span_metrics_calls_total
      Labels: service_name, status_code, namespace, span_name
      Error indicator: status_code="STATUS_CODE_ERROR"

  - traces_span_metrics_duration_milliseconds_bucket
      Labels: le, service_name, namespace, span_name
      Used for P95 latency via histogram_quantile()

Because these metrics carry service_name but NOT pod-level labels, this
module resolves service_name → pod names using the Kubernetes API label
selectors (io.kompose.service or app labels) after anomaly detection.

Dual Z-Score Modes
------------------
  benchmark   1m vs 10m sliding window, threshold > 2.0
              No offset — works with <15 min of Prometheus history.
              Default for SREGym experiments.

  production  5m vs 1h window with offset 1d seasonality, threshold > 3.0
              Requires 24h+ of Prometheus history. For long-running clusters.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from kubernetes import client as k8s_client
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guard: import prometheus_api_client gracefully so syntax checks pass even
# without the library installed.
# ---------------------------------------------------------------------------
try:
    from prometheus_api_client import PrometheusConnect
    _PROM_CLIENT_AVAILABLE = True
except ImportError:
    _PROM_CLIENT_AVAILABLE = False
    PrometheusConnect = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Service-name exclusion list — mirrors the existing alert rule filters
# in SREGym/sregym/observer/prometheus/prometheus/values.yaml L831
# ---------------------------------------------------------------------------
_EXCLUDE_SERVICES = "load-generator|flagd|kafka|valkey-cart|postgresql|wlgen"


# ---------------------------------------------------------------------------
# Z-Score profiles
# ---------------------------------------------------------------------------

@dataclass
class ZScoreProfile:
    """
    Encapsulates PromQL parameters for a specific Z-Score operating mode.

    Attributes
    ----------
    name:             Human-readable mode name ("production" | "benchmark").
    current_window:   Rate window for the current measurement (e.g., "5m").
    baseline_window:  Over-time window for the historical baseline (e.g., "1h").
    baseline_step:    Resolution step for the baseline sub-query (e.g., "1m").
                      Empty string means Prometheus uses its default step.
    offset:           PromQL offset modifier for seasonality (e.g., "offset 1d").
                      Empty string means no offset (benchmark mode).
    z_threshold:      Minimum Z-score to classify a pod as anomalous.
    """
    name: str
    current_window: str
    baseline_window: str
    baseline_step: str
    offset: str
    z_threshold: float


#: Production mode — seasonality-adjusted, requires 24h+ of history.
PRODUCTION_PROFILE = ZScoreProfile(
    name="production",
    current_window="5m",
    baseline_window="1h",
    baseline_step="",
    offset="offset 1d",
    z_threshold=3.0,
)

#: Benchmark mode — short-window baseline, 5m offset to avoid overlapping with the fault.
BENCHMARK_PROFILE = ZScoreProfile(
    name="benchmark",
    current_window="3m",
    baseline_window="5m",
    baseline_step="1m",
    offset="offset 5m",
    z_threshold=2.0,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class AnomalousPod:
    """
    A Kubernetes pod identified as semantically anomalous by Prometheus
    Z-Score analysis.

    Attributes
    ----------
    namespace:          Kubernetes namespace of the pod.
    pod_name:           Full pod name (e.g., "frontend-6d7c9b4f8d-xkbtp").
    container:          Primary container name, if resolvable from pod spec.
    service_name:       The spanmetrics service_name label that triggered the
                        anomaly (e.g., "frontend_service").
    signal_type:        Which signal detected the anomaly:
                        "error_rate_zscore" | "latency_zscore".
    z_score:            The computed Z-score magnitude (always > threshold).
    raw_metric_labels:  Full Prometheus label dictionary for debugging.
    """
    namespace: str
    pod_name: str
    container: Optional[str]
    service_name: str
    signal_type: str
    z_score: float
    raw_metric_labels: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Anomaly engine
# ---------------------------------------------------------------------------

class PrometheusAnomalyEngine:
    """
    Executes Z-Score anomaly detection against the SREGym Prometheus instance
    and resolves anomalous service names to Kubernetes pod identifiers.

    Parameters
    ----------
    prom_url:   Full URL to the Prometheus API (e.g., "http://1.2.3.4:9090").
    profile:    ZScoreProfile (PRODUCTION_PROFILE or BENCHMARK_PROFILE).
    v1:         Initialized kubernetes CoreV1Api client for pod resolution.
    namespace:  Target Kubernetes namespace. Defaults to
                "blueprint-hotel-reservation".
    """

    def __init__(
        self,
        prom_url: str,
        profile: ZScoreProfile,
        v1: k8s_client.CoreV1Api,
        namespace: str = "blueprint-hotel-reservation",
    ) -> None:
        self.profile = profile
        self.namespace = namespace
        self._v1 = v1

        if not _PROM_CLIENT_AVAILABLE:
            logger.error(
                "[PrometheusAnomalyEngine] prometheus-api-client is not installed. "
                "Run: pip install prometheus-api-client>=0.5.0"
            )
            self._prom: Optional[PrometheusConnect] = None
            return

        try:
            self._prom = PrometheusConnect(url=prom_url, disable_ssl=True)
            # Perform a lightweight connectivity check
            self._prom.custom_query(query="up")
            logger.info(
                "[PrometheusAnomalyEngine] Connected to Prometheus at %s (mode=%s)",
                prom_url, profile.name,
            )
        except Exception as exc:
            logger.warning(
                "[PrometheusAnomalyEngine] Cannot reach Prometheus at %s: %s",
                prom_url, exc,
            )
            self._prom = None

    @property
    def is_available(self) -> bool:
        """True when the Prometheus connection was established successfully."""
        return self._prom is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_anomalous_pods(self) -> list[AnomalousPod]:
        """
        Execute both Z-Score queries, resolve service names to pod names,
        and return a deduplicated list of anomalous pods.

        Returns an empty list if Prometheus is unavailable or if no
        anomalies exceed the configured threshold.
        """
        if not self.is_available:
            logger.warning(
                "[PrometheusAnomalyEngine] Prometheus unavailable — returning empty anomaly list."
            )
            return []

        anomalous_pods: list[AnomalousPod] = []
        seen_pods: set[str] = set()

        # ── Error Rate Z-Score ────────────────────────────────────────
        error_query = self._build_error_rate_zscore_query()
        logger.info("[PrometheusAnomalyEngine] Executing error rate Z-Score query (mode=%s)", self.profile.name)
        logger.debug("[PrometheusAnomalyEngine] Error query:\n%s", error_query)

        error_results = self._execute_query(error_query)
        anomalous_services_error = self._parse_anomalous_services(error_results, "error_rate_zscore")

        # ── P95 Latency Z-Score ───────────────────────────────────────
        latency_query = self._build_latency_zscore_query()
        logger.info("[PrometheusAnomalyEngine] Executing P95 latency Z-Score query (mode=%s)", self.profile.name)
        logger.debug("[PrometheusAnomalyEngine] Latency query:\n%s", latency_query)

        latency_results = self._execute_query(latency_query)
        anomalous_services_latency = self._parse_anomalous_services(latency_results, "latency_zscore")

        # ── Throughput Drop Z-Score ───────────────────────────────────
        throughput_query = self._build_throughput_drop_zscore_query()
        logger.info("[PrometheusAnomalyEngine] Executing throughput drop Z-Score query (mode=%s)", self.profile.name)
        logger.debug("[PrometheusAnomalyEngine] Throughput query:\n%s", throughput_query)

        throughput_results = self._execute_query(throughput_query)
        anomalous_services_throughput = self._parse_anomalous_services(throughput_results, "throughput_drop_zscore")

        # ── Resolve service names → pods and deduplicate ──────────────
        all_anomalous_services = anomalous_services_error + anomalous_services_latency + anomalous_services_throughput

        for svc_entry in all_anomalous_services:
            svc_name = svc_entry["service_name"]
            ns = svc_entry["namespace"]
            signal = svc_entry["signal_type"]
            z_score = svc_entry["z_score"]
            raw_labels = svc_entry["raw_labels"]

            pods = self._resolve_service_to_pods(svc_name, ns)
            if not pods:
                logger.warning(
                    "[PrometheusAnomalyEngine] Could not resolve service '%s' to any pods in '%s'.",
                    svc_name, ns,
                )
                continue

            for pod in pods:
                pod_name = pod.metadata.name
                dedup_key = f"{ns}/{pod_name}/{signal}"
                if dedup_key in seen_pods:
                    continue
                seen_pods.add(dedup_key)

                # Extract primary container name from pod spec
                container = self._get_primary_container(pod)

                anomalous_pods.append(AnomalousPod(
                    namespace=ns,
                    pod_name=pod_name,
                    container=container,
                    service_name=svc_name,
                    signal_type=signal,
                    z_score=z_score,
                    raw_metric_labels=raw_labels,
                ))

        logger.info(
            "[PrometheusAnomalyEngine] Detection complete: %d anomalous pod(s) identified.",
            len(anomalous_pods),
        )
        return anomalous_pods

    # ------------------------------------------------------------------
    # PromQL query builders
    # ------------------------------------------------------------------

    def _build_error_rate_zscore_query(self) -> str:
        """
        Build the Z-Score query for HTTP/gRPC error rate.

        Metric: traces_span_metrics_calls_total{status_code="STATUS_CODE_ERROR"}
        This is the same metric used by the HighRequestErrorRate alert in
        SREGym/sregym/observer/prometheus/prometheus/values.yaml L823.
        """
        p = self.profile
        cur_win = p.current_window
        offset = f" {p.offset}" if p.offset else ""
        step = f":{p.baseline_step}" if p.baseline_step else ":"
        base_win = p.baseline_window

        job_filter = 'job=~".*otel-demo.*"' if self.namespace == "astronomy-shop" else 'job!~".*otel-demo.*"'
        metric_err = (
            f'traces_span_metrics_calls_total{{'
            f'{job_filter},'
            f'status_code="STATUS_CODE_ERROR",'
            f'service_name!~"{_EXCLUDE_SERVICES}"}}'
        )

        query = f"""(
  sum by (service_name, namespace) (
    rate({metric_err}[{cur_win}])
  )
  -
  sum by (service_name, namespace) (
    avg_over_time(
      rate({metric_err}[{cur_win}]{offset})[{base_win}{step}]
    )
  )
)
/
clamp_min(
  sum by (service_name, namespace) (
    stddev_over_time(
      rate({metric_err}[{cur_win}]{offset})[{base_win}{step}]
    )
  ),
  0.001
) > {p.z_threshold}"""
        return query

    def _build_latency_zscore_query(self) -> str:
        """
        Build the Z-Score query for P95 latency.

        Metric: traces_span_metrics_duration_milliseconds_bucket
        This is the same metric used by the HighRequestLatency alert in
        SREGym/sregym/observer/prometheus/prometheus/values.yaml L915.

        Note: histogram_quantile() cannot be wrapped inside avg_over_time()
        directly. For benchmark mode we use the approach of computing
        histogram_quantile over the short rate window and comparing the
        current P95 value against the rolling stddev of error rate as a
        proxy signal. For a fully correct latency Z-score we use the
        Z-score on the raw rate of the histogram total calls combined
        with a separate P95 threshold check.
        """
        p = self.profile
        cur_win = p.current_window
        offset = f" {p.offset}" if p.offset else ""
        step = f":{p.baseline_step}" if p.baseline_step else ":"
        base_win = p.baseline_window

        job_filter = 'job=~".*otel-demo.*"' if self.namespace == "astronomy-shop" else 'job!~".*otel-demo.*"'
        metric_bucket = (
            f'traces_span_metrics_duration_milliseconds_bucket{{'
            f'{job_filter},'
            f'service_name!~"{_EXCLUDE_SERVICES}"}}'
        )

        # P95 latency Z-score:
        # current = histogram_quantile(0.95, rate(bucket[cur_win]))
        # baseline = avg_over_time of total_calls rate over baseline window
        # We compute Z-score on the total call rate (all spans) as the
        # denominator proxy. This detects latency spikes via anomalous
        # throughput changes while a separate absolute P95 > 3000ms check
        # (matching the HighRequestLatency alert threshold) guards for
        # pure latency degradation with steady throughput.
        job_filter = 'job=~".*otel-demo.*"' if self.namespace == "astronomy-shop" else 'job!~".*otel-demo.*"'
        metric_all = (
            f'traces_span_metrics_calls_total{{'
            f'{job_filter},'
            f'service_name!~"{_EXCLUDE_SERVICES}"}}'
        )

        query = f"""((
  histogram_quantile(0.95,
    sum by (le, service_name) (
      rate({metric_bucket}[{cur_win}])
    )
  )
  -
  sum by (service_name) (
    avg_over_time(
      (
        histogram_quantile(0.95,
          sum by (le, service_name) (
            rate({metric_bucket}[{cur_win}]{offset})
          )
        ) >= 0
      )[{base_win}{step}]
    )
  )
)
/
clamp_min(
  sum by (service_name) (
    stddev_over_time(
      (
        histogram_quantile(0.95,
          sum by (le, service_name) (
            rate({metric_bucket}[{cur_win}]{offset})
          )
        ) >= 0
      )[{base_win}{step}]
    )
  ),
  0.001
) > {p.z_threshold})
or
(
  histogram_quantile(0.95,
    sum by (le, service_name) (
      rate({metric_bucket}[{cur_win}])
    )
  ) > 2000
)"""
        return query

    def _build_throughput_drop_zscore_query(self) -> str:
        """
        Build the Z-Score query for a sudden drop in throughput (Silent Death).

        Metric: traces_span_metrics_calls_total
        Detects when the current call rate drops significantly below the
        historical baseline, which happens when the telemetry pipeline
        or node network crashes and spans stop being exported.
        """
        p = self.profile
        cur_win = p.current_window
        offset = f" {p.offset}" if p.offset else ""
        step = f":{p.baseline_step}" if p.baseline_step else ":"
        base_win = p.baseline_window

        job_filter = 'job=~".*otel-demo.*"' if self.namespace == "astronomy-shop" else 'job!~".*otel-demo.*"'
        metric_all = (
            f'traces_span_metrics_calls_total{{'
            f'{job_filter},'
            f'service_name!~"{_EXCLUDE_SERVICES}"}}'
        )

        query = f"""((
  sum by (service_name) (
    avg_over_time(
      rate({metric_all}[{cur_win}]{offset})[{base_win}{step}]
    )
  )
  -
  sum by (service_name) (
    rate({metric_all}[{cur_win}])
  )
)
/
clamp_min(
  sum by (service_name) (
    stddev_over_time(
      rate({metric_all}[{cur_win}]{offset})[{base_win}{step}]
    )
  ),
  0.001
) > {p.z_threshold})
and
(
  sum by (service_name) (
    avg_over_time(
      rate({metric_all}[{cur_win}]{offset})[{base_win}{step}]
    )
  ) > 0.5
)"""
        return query

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    def _execute_query(self, query: str) -> list[dict]:
        """
        Execute a PromQL instant query via prometheus-api-client.

        Returns an empty list on any connection or API error so the
        caller can fall back gracefully.
        """
        try:
            results = self._prom.custom_query(query=query)
            return results if results else []
        except Exception as exc:
            logger.error(
                "[PrometheusAnomalyEngine] Query execution failed: %s", exc
            )
            return []

    # ------------------------------------------------------------------
    # Result parsing
    # ------------------------------------------------------------------

    def _parse_anomalous_services(
        self,
        raw_results: list[dict],
        signal_type: str,
    ) -> list[dict]:
        """
        Extract service_name, namespace, and Z-score from raw PromQL results.

        The prometheus-api-client returns a list of dicts shaped as:
            {"metric": {"service_name": "...", ...}, "value": [timestamp, "z_score"]}
        """
        services = []
        for result in raw_results:
            labels = result.get("metric", {})
            service_name = labels.get("service_name", "")
            namespace = labels.get("namespace", self.namespace)

            if not service_name:
                logger.debug(
                    "[PrometheusAnomalyEngine] Skipping result with no service_name: %s", labels
                )
                continue

            try:
                z_score = float(result.get("value", [0, "0"])[1])
            except (ValueError, IndexError):
                z_score = 0.0

            logger.info(
                "[PrometheusAnomalyEngine] 🚨 Anomaly detected: service=%s namespace=%s "
                "signal=%s z_score=%.2f",
                service_name, namespace, signal_type, z_score,
            )
            services.append({
                "service_name": service_name,
                "namespace": namespace,
                "signal_type": signal_type,
                "z_score": z_score,
                "raw_labels": labels,
            })
        return services

    # ------------------------------------------------------------------
    # Service → Pod resolution
    # ------------------------------------------------------------------

    def _resolve_service_to_pods(
        self,
        service_name: str,
        namespace: str,
    ) -> list:
        """
        Resolve a spanmetrics service_name to the running Kubernetes pods
        in the given namespace.

        Resolution is attempted in priority order:

        Priority 0 — k8s.pod.name label (requires k8sattributes OTel processor):
            Queries pods by ``k8s.pod.name=<service_name>`` if the spanmetrics
            metric was produced with the enriched k8s.pod.name attribute injected
            by the k8sattributes processor. This is the most precise resolution
            path and survives rolling updates, ReplicaSet changes, and scaling
            events without breaking.

        Priority 1 — spanmetrics label normalization + label selectors:
            The OTel Collector's metric_relabel_config strips the
            "unknown_service:" prefix and "_process" suffix from service names,
            so "unknown_service:frontend_service_process" becomes "frontend_service".
            We attempt pod lookup via two label strategies:
              a. io.kompose.service=<service_name>  (DeathStarBench/docker-compose)
              b. app=<service_name>                 (generic Kubernetes convention)
              c. app.kubernetes.io/name, app.kubernetes.io/component

        Returns a list of V1Pod objects (may be empty if nothing matches).
        """
        if self._v1 is None:
            return []

        # ── Priority 0: k8s.pod.name label (injected by k8sattributes) ────
        # When the OTel k8sattributes processor is active (10-otel-k8sattributes-patch.yaml)
        # spanmetrics metrics carry k8s.pod.name as a label that resolves directly
        # to the pod without brittle service-name normalisation.
        try:
            pod_list = self._v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=f"k8s.pod.name={service_name}",
            )
            running = [
                p for p in pod_list.items
                if p.status.phase in ("Running", "Pending")
                and "wlgen" not in p.metadata.name
                and "loadgenerator" not in p.metadata.name
            ]
            if running:
                logger.debug(
                    "[PrometheusAnomalyEngine] Resolved '%s' → %d pod(s) via "
                    "k8s.pod.name label (k8sattributes path)",
                    service_name, len(running),
                )
                return running
        except Exception:
            pass  # Expected when k8sattributes is not yet deployed — fall through

        # ── Priority 1: Service name normalisation + label selectors ──────
        # Clean up Jaeger traces from blueprint-hotel-reservation
        # "unknown_service:frontend_service_process" -> "frontend"
        if service_name.startswith("unknown_service:") and service_name.endswith("_process"):
            service_name = service_name.replace("unknown_service:", "").replace("_process", "")
            if service_name.endswith("_service"):
                service_name = service_name.replace("_service", "")

        # In hotel reservation, pods are sometimes suffixed with -service (e.g., frontend-service)
        # So we add an extra fallback without the -service, or with it
        base_name = service_name
        if not base_name.endswith("-service"):
            fallback_name = base_name + "-service"
        else:
            fallback_name = base_name.replace("-service", "")

        selector_candidates = [
            f"io.kompose.service={service_name}",
            f"app={service_name}",
            f"app.kubernetes.io/name={service_name}",
            f"app.kubernetes.io/component={service_name}",
            f"app={fallback_name}",
            f"io.kompose.service={fallback_name}",
        ]

        for selector in selector_candidates:
            try:
                pod_list = self._v1.list_namespaced_pod(
                    namespace=namespace,
                    label_selector=selector,
                )
                running = [
                    p for p in pod_list.items
                    if p.status.phase in ("Running", "Pending")
                    and "wlgen" not in p.metadata.name
                    and "loadgenerator" not in p.metadata.name
                ]
                if running:
                    logger.debug(
                        "[PrometheusAnomalyEngine] Resolved '%s' → %d pod(s) via selector '%s'",
                        service_name, len(running), selector,
                    )
                    return running
            except ApiException as exc:
                logger.warning(
                    "[PrometheusAnomalyEngine] K8s API error resolving '%s' with selector '%s': %s",
                    service_name, selector, exc.reason,
                )
            except Exception as exc:
                logger.warning(
                    "[PrometheusAnomalyEngine] Unexpected error resolving '%s': %s",
                    service_name, exc,
                )

        logger.warning(
            "[PrometheusAnomalyEngine] No pods found for service_name='%s' in namespace='%s'. "
            "Tried k8s.pod.name label + selectors: %s",
            service_name, namespace, selector_candidates,
        )
        return []

    @staticmethod
    def _get_primary_container(pod) -> Optional[str]:
        """
        Extract the primary application container name from a pod spec.

        Skips common sidecar names (envoy, istio-proxy, fluentd, jaeger-agent).
        Returns None if no containers are found.
        """
        _SIDECAR_NAMES = {"envoy", "istio-proxy", "fluentd", "jaeger-agent", "linkerd-proxy"}
        containers = pod.spec.containers or []
        for c in containers:
            if c.name.lower() not in _SIDECAR_NAMES:
                return c.name
        return containers[0].name if containers else None


# ---------------------------------------------------------------------------
# Forensic log extraction (standalone — used by live_harness.py)
# ---------------------------------------------------------------------------

def extract_forensic_logs(
    v1: k8s_client.CoreV1Api,
    namespace: str,
    pod_name: str,
    container: Optional[str] = None,
    tail_lines: int = 250,
) -> str:
    """
    Resilient forensic log extraction from a Kubernetes pod.

    Handles the following edge cases:

    1. **Multi-container pods**: If ``container`` is None, queries the pod
       spec to enumerate all containers and concatenates their logs.

    2. **Empty live logs + previous=True fallback**: If the live container
       log buffer is empty (e.g., pod just restarted after an OOMKill or
       crash), retries with ``previous=True`` to retrieve the deceased
       container's fatal stack trace.

    3. **CrashLoopBackOff detection**: If any container is in
       CrashLoopBackOff waiting state, ``previous=True`` is attempted
       immediately rather than waiting for an empty live log.

    4. **ApiException handling**: HTTP 400 (multi-container ambiguity),
       404 (pod vanished mid-extraction), and generic errors all return
       descriptive strings rather than raising, keeping the harness alive.

    Parameters
    ----------
    v1:          Initialized ``kubernetes.client.CoreV1Api`` instance.
    namespace:   Kubernetes namespace of the pod.
    pod_name:    Full pod name.
    container:   Specific container to read. If None, all containers
                 are queried and logs are concatenated.
    tail_lines:  Number of trailing log lines to retrieve per container.

    Returns
    -------
    str
        Concatenated log text, or a descriptive error string on failure.
    """
    # Determine which containers to query
    containers_to_query: list[str] = []

    if container:
        containers_to_query = [container]
    else:
        # Query pod spec to enumerate all containers
        try:
            pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
            containers_to_query = [c.name for c in (pod.spec.containers or [])]
            if not containers_to_query:
                return f"[System] Pod '{pod_name}' has no containers in spec."
        except ApiException as exc:
            if exc.status == 404:
                return f"[Error] Pod '{pod_name}' not found in namespace '{namespace}'."
            return f"[Error] Cannot read pod spec for '{pod_name}': {exc.reason}"

    all_logs: list[str] = []

    for cname in containers_to_query:
        log_text = _read_container_logs(
            v1=v1,
            namespace=namespace,
            pod_name=pod_name,
            container=cname,
            tail_lines=tail_lines,
        )
        if log_text:
            header = f"--- Container: {cname} ---"
            all_logs.append(f"{header}\n{log_text}")

    if not all_logs:
        return f"[System] No log content retrieved from pod '{pod_name}'."

    return "\n\n".join(all_logs)


def _read_container_logs(
    v1: k8s_client.CoreV1Api,
    namespace: str,
    pod_name: str,
    container: str,
    tail_lines: int,
) -> str:
    """
    Read logs for a single named container, with previous=True fallback.

    Attempts live log first. If the live log is empty or the container is
    in CrashLoopBackOff, falls back to previous=True to retrieve the
    terminated container's log buffer.
    """
    # Check for CrashLoopBackOff to decide whether to skip straight to previous
    use_previous_first = _is_crash_looping(v1, namespace, pod_name, container)

    if not use_previous_first:
        # Attempt live log
        live_log = _fetch_log(v1, namespace, pod_name, container, tail_lines, previous=False)
        if live_log:
            return live_log
        logger.debug(
            "[extract_forensic_logs] Live log empty for %s/%s — retrying with previous=True",
            pod_name, container,
        )

    # Fallback: previous container instance
    prev_log = _fetch_log(v1, namespace, pod_name, container, tail_lines, previous=True)
    if prev_log:
        return f"[PREVIOUS CONTAINER LOGS]\n{prev_log}"

    return ""


def _fetch_log(
    v1: k8s_client.CoreV1Api,
    namespace: str,
    pod_name: str,
    container: str,
    tail_lines: int,
    previous: bool,
) -> str:
    """
    Execute a single read_namespaced_pod_log call.

    Returns the log string on success, empty string on any error or
    empty response.
    """
    try:
        logs = v1.read_namespaced_pod_log(
            name=pod_name,
            namespace=namespace,
            container=container,
            tail_lines=tail_lines,
            previous=previous,
            _preload_content=True,
        )
        return logs.strip() if logs else ""

    except ApiException as exc:
        if exc.status == 404:
            logger.warning(
                "[extract_forensic_logs] Pod '%s' not found (404) — may have terminated.",
                pod_name,
            )
        elif exc.status == 400:
            logger.warning(
                "[extract_forensic_logs] Bad request reading logs from '%s/%s': %s",
                pod_name, container, exc.reason,
            )
        else:
            logger.warning(
                "[extract_forensic_logs] ApiException reading '%s/%s' previous=%s: %s",
                pod_name, container, previous, exc.reason,
            )
        return ""

    except Exception as exc:
        logger.warning(
            "[extract_forensic_logs] Unexpected error reading '%s/%s': %s",
            pod_name, container, exc,
        )
        return ""


def _is_crash_looping(
    v1: k8s_client.CoreV1Api,
    namespace: str,
    pod_name: str,
    container: str,
) -> bool:
    """
    Check if the named container is in CrashLoopBackOff.

    Returns True if we should skip straight to previous=True log retrieval.
    """
    try:
        pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
        for status in (pod.status.container_statuses or []):
            if status.name == container:
                waiting = status.state.waiting
                if waiting and waiting.reason in (
                    "CrashLoopBackOff", "OOMKilled", "Error"
                ):
                    logger.debug(
                        "[extract_forensic_logs] Container '%s/%s' is in %s — using previous=True.",
                        pod_name, container, waiting.reason,
                    )
                    return True
        return False
    except Exception:
        return False
