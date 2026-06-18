"""
airs_ml_sidecar/features.py
============================

Feature engineering for the AIRS Isolation Forest anomaly detection sidecar.

Queries Prometheus for rolling RED metrics and produces a feature vector
suitable for IsolationForest.score_samples(). Feature engineering is the
critical differentiator between a toy IF demo and a production-grade detector.

Feature set (per service, per scrape interval)
----------------------------------------------
  error_rate_pct       — Current HTTP/query error rate %
  error_rate_delta     — Rate-of-change vs. previous window (acceleration)
  error_rate_zscore    — Normalised deviation from rolling mean
  latency_p95_ms       — 95th-percentile response time (ms)
  latency_p95_delta    — Latency acceleration
  latency_p95_zscore   — Normalised latency deviation
  throughput_rps       — Requests per second
  throughput_ratio     — Current / max-in-window (detects silent death: 0.0)
  cross_service_corr   — Pearson correlation with downstream error rates
                         (captures cascading failure patterns)

All features are normalised to [0, 1] before scoring to prevent high-magnitude
features (e.g., latency in ms) from dominating low-magnitude ones (error rate %).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional prometheus_api_client import — allows syntax checks without dep
# ---------------------------------------------------------------------------
try:
    from prometheus_api_client import PrometheusConnect
    _PROM_AVAILABLE = True
except ImportError:
    PrometheusConnect = None  # type: ignore[assignment,misc]
    _PROM_AVAILABLE = False
    logger.warning("[IF-Features] prometheus-api-client not installed")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Rolling window for mean/std computation (in seconds)
_ROLLING_WINDOW_SECONDS = 300  # 5 minutes

# Step resolution for range queries
_STEP_SECONDS = 15  # Match Prometheus scrape interval


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ServiceFeatureVector:
    """
    Feature vector for a single service at a point in time.

    All features normalised to roughly [0, 1] scale.
    """
    service_name: str
    namespace: str
    timestamp: float

    # Error rate features
    error_rate_pct: float = 0.0
    error_rate_delta: float = 0.0
    error_rate_zscore: float = 0.0

    # Latency features
    latency_p95_ms: float = 0.0
    latency_p95_delta: float = 0.0
    latency_p95_zscore: float = 0.0

    # Throughput features
    throughput_rps: float = 0.0
    throughput_ratio: float = 1.0  # 1.0 = normal, 0.0 = silent death

    # Cross-service correlation
    cross_service_corr: float = 0.0

    def to_array(self) -> list[float]:
        """Return features as a flat list for numpy/sklearn input."""
        return [
            self.error_rate_pct / 100.0,        # normalise to [0,1]
            self.error_rate_delta / 10.0,        # ±10% change is extreme
            self.error_rate_zscore / 5.0,        # ±5σ is extreme
            min(self.latency_p95_ms / 5000.0, 1.0),  # 5000ms cap
            self.latency_p95_delta / 1000.0,     # ±1000ms change is extreme
            self.latency_p95_zscore / 5.0,
            min(self.throughput_rps / 1000.0, 1.0),  # 1000 RPS cap
            self.throughput_ratio,
            (self.cross_service_corr + 1.0) / 2.0,  # [-1,1] → [0,1]
        ]


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class PrometheusFeatureExtractor:
    """
    Extracts rolling feature vectors for all services from Prometheus.

    Parameters
    ----------
    prometheus_url: Prometheus server URL (default: in-cluster service URL)
    namespace:      K8s namespace to filter services (default: "default")
    """

    def __init__(
        self,
        prometheus_url: str = "http://prometheus.monitoring.svc.cluster.local:9090",
        namespace: str = "default",
    ) -> None:
        self._url = prometheus_url
        self._namespace = namespace
        self._client = None
        if _PROM_AVAILABLE:
            self._client = PrometheusConnect(url=prometheus_url, disable_ssl=True)

    def _query_scalar(self, promql: str) -> float:
        """Execute an instant query and return the scalar result."""
        if self._client is None:
            return 0.0
        try:
            result = self._client.custom_query(promql)
            if result and result[0].get("value"):
                return float(result[0]["value"][1])
        except Exception as exc:
            logger.debug("[IF-Features] Scalar query failed: %s — %s", promql, exc)
        return 0.0

    def _query_range_values(self, promql: str, duration_s: int = _ROLLING_WINDOW_SECONDS) -> np.ndarray:
        """Execute a range query and return values as a numpy array."""
        if self._client is None:
            return np.array([])
        try:
            end = time.time()
            start = end - duration_s
            result = self._client.custom_query_range(
                promql,
                start_time=start,
                end_time=end,
                step=f"{_STEP_SECONDS}s",
            )
            if result and result[0].get("values"):
                return np.array([float(v[1]) for v in result[0]["values"]])
        except Exception as exc:
            logger.debug("[IF-Features] Range query failed: %s — %s", promql, exc)
        return np.array([])

    def _compute_zscore_and_delta(
        self,
        values: np.ndarray,
        current: float,
    ) -> tuple[float, float]:
        """
        Compute z-score (current vs. rolling window) and delta (current vs. previous).
        Returns (zscore, delta) as floats. Safe for empty arrays.
        """
        if len(values) < 2:
            return 0.0, 0.0
        mean = float(np.mean(values))
        std = float(np.std(values))
        std = max(std, 0.001)  # Prevent division by zero
        zscore = (current - mean) / std
        delta = current - float(values[-2]) if len(values) >= 2 else 0.0
        return zscore, delta

    def extract_features_for_service(
        self,
        service_name: str,
    ) -> Optional[ServiceFeatureVector]:
        """
        Extract the full feature vector for a single service.

        Returns None if no Prometheus data is available for the service.
        """
        ns = self._namespace

        # ── Error rate ────────────────────────────────────────────────────
        err_rate_q = (
            f'100 * sum(rate(traces_span_metrics_calls_total{{status_code="STATUS_CODE_ERROR",'
            f'k8s_namespace_name="{ns}",service_name="{service_name}"}}[1m])) / '
            f'sum(rate(traces_span_metrics_calls_total{{'
            f'k8s_namespace_name="{ns}",service_name="{service_name}"}}[1m]))'
        )
        err_rate = self._query_scalar(err_rate_q)
        err_range = self._query_range_values(err_rate_q)
        err_zscore, err_delta = self._compute_zscore_and_delta(err_range, err_rate)

        # ── P95 Latency ───────────────────────────────────────────────────
        lat_q = (
            f'histogram_quantile(0.95, sum(rate('
            f'traces_span_metrics_duration_milliseconds_bucket{{'
            f'k8s_namespace_name="{ns}",service_name="{service_name}"}}[1m])) by (le))'
        )
        lat_p95 = self._query_scalar(lat_q)
        lat_range = self._query_range_values(lat_q)
        lat_zscore, lat_delta = self._compute_zscore_and_delta(lat_range, lat_p95)

        # ── Throughput ────────────────────────────────────────────────────
        throughput_q = (
            f'sum(rate(traces_span_metrics_calls_total{{'
            f'k8s_namespace_name="{ns}",service_name="{service_name}"}}[1m]))'
        )
        throughput = self._query_scalar(throughput_q)
        tp_range = self._query_range_values(throughput_q)
        max_tp = float(np.max(tp_range)) if len(tp_range) > 0 else 1.0
        tp_ratio = throughput / max_tp if max_tp > 0 else 1.0

        # ── Cross-service correlation ─────────────────────────────────────
        # Compute Pearson correlation between this service's error rate and
        # the aggregate error rate of all OTHER services in the same namespace.
        # High correlation → cascading failure pattern (upstream root cause).
        cross_corr = 0.0
        agg_err_q = (
            f'100 * sum(rate(traces_span_metrics_calls_total{{status_code="STATUS_CODE_ERROR",'
            f'k8s_namespace_name="{ns}"}}[1m])) / '
            f'sum(rate(traces_span_metrics_calls_total{{k8s_namespace_name="{ns}"}}[1m]))'
        )
        agg_range = self._query_range_values(agg_err_q)
        if len(err_range) > 5 and len(agg_range) > 5:
            min_len = min(len(err_range), len(agg_range))
            try:
                corr_matrix = np.corrcoef(err_range[-min_len:], agg_range[-min_len:])
                cross_corr = float(corr_matrix[0, 1])
                if np.isnan(cross_corr):
                    cross_corr = 0.0
            except Exception:
                cross_corr = 0.0

        if err_rate == 0.0 and lat_p95 == 0.0 and throughput == 0.0:
            return None  # No data for this service

        return ServiceFeatureVector(
            service_name=service_name,
            namespace=ns,
            timestamp=time.time(),
            error_rate_pct=err_rate,
            error_rate_delta=err_delta,
            error_rate_zscore=err_zscore,
            latency_p95_ms=lat_p95,
            latency_p95_delta=lat_delta,
            latency_p95_zscore=lat_zscore,
            throughput_rps=throughput,
            throughput_ratio=min(max(tp_ratio, 0.0), 1.0),
            cross_service_corr=cross_corr,
        )

    def extract_all_services(self) -> list[ServiceFeatureVector]:
        """
        Discover all services in the namespace from Prometheus metric labels
        and extract feature vectors for each.
        """
        services_q = (
            f'group by (service_name) (traces_span_metrics_calls_total{{'
            f'k8s_namespace_name="{self._namespace}"}})'
        )
        if self._client is None:
            return []
        try:
            result = self._client.custom_query(services_q)
            service_names = [
                r["metric"].get("service_name", "")
                for r in result
                if r["metric"].get("service_name")
            ]
        except Exception as exc:
            logger.warning("[IF-Features] Service discovery failed: %s", exc)
            return []

        vectors = []
        for svc in service_names:
            vec = self.extract_features_for_service(svc)
            if vec is not None:
                vectors.append(vec)

        return vectors
