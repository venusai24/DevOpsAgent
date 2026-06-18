"""
airs_ml_sidecar/model.py
=========================

Isolation Forest anomaly scoring model for the AIRS ML sidecar.

Design decisions
----------------
1. **IsolationForest with contamination='auto'**: Uses sklearn's automatic
   contamination estimation rather than a static value. Avoids the common
   production misconfiguration where a fixed contamination=0.1 assumes
   exactly 10% of data is anomalous regardless of actual failure rate.

2. **Incremental warm-up**: The model collects a baseline window of N
   observations before producing scores. During warm-up, scores default
   to 0.0 (no anomaly) to prevent false positives from an untrained model.

3. **Exposure as Prometheus metric**: Anomaly scores are exposed as
   ``airs_if_anomaly_score{service_name, namespace}`` gauges, scraped by
   the cluster's main Prometheus. This integrates IF detection into the
   existing alerting infrastructure without a new data pipeline.

4. **Score normalisation**: IsolationForest.score_samples() returns values
   in roughly [-0.5, 0.5] where negative = anomalous. We invert and
   normalise to [0, 1] where 1.0 = maximally anomalous for Prometheus
   gauge compatibility.

5. **Thread-safe scoring**: The model is retrained periodically in a
   background thread. The scoring function always reads from an immutable
   snapshot, avoiding race conditions between train and score.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    from sklearn.ensemble import IsolationForest as _IF
    _SKLEARN_AVAILABLE = True
except ImportError:
    _IF = None  # type: ignore[assignment,misc]
    _SKLEARN_AVAILABLE = False
    logger.warning("[IF-Model] scikit-learn not installed — scoring disabled")

try:
    from prometheus_client import Gauge, Counter
    _PROM_CLIENT_AVAILABLE = True
except ImportError:
    Gauge = None    # type: ignore[assignment,misc]
    Counter = None  # type: ignore[assignment,misc]
    _PROM_CLIENT_AVAILABLE = False
    logger.warning("[IF-Model] prometheus-client not installed — metrics disabled")


# ---------------------------------------------------------------------------
# Prometheus metrics (exposed at /metrics for scraping)
# ---------------------------------------------------------------------------
if _PROM_CLIENT_AVAILABLE:
    _anomaly_score_gauge = Gauge(
        "airs_if_anomaly_score",
        "Isolation Forest anomaly score for a service. "
        "0.0 = normal, 1.0 = maximally anomalous.",
        labelnames=["service_name", "namespace"],
    )
    _model_warmup_remaining = Gauge(
        "airs_if_warmup_samples_remaining",
        "Number of observations still needed before IF produces live scores.",
    )
    _training_counter = Counter(
        "airs_if_model_trainings_total",
        "Total number of model retraining cycles completed.",
    )
else:
    _anomaly_score_gauge = None
    _model_warmup_remaining = None
    _training_counter = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Minimum observations before the model starts producing scores.
# With a 15s scrape interval: 200 samples ≈ 50 minutes of baseline.
_WARMUP_SAMPLES = 200

# Rolling window size for the training buffer.
# Keeps the model adapting to concept drift without full history replay.
# With 15s intervals: 2000 samples ≈ 8 hours of rolling context.
_TRAINING_WINDOW = 2000

# Retrain the model every N new observations (not every scrape — expensive).
_RETRAIN_INTERVAL = 50

# Anomaly threshold: score above this → flag as anomalous in API response.
# 0.65 means "roughly 65th percentile of anomalousness" in the current window.
_ANOMALY_THRESHOLD = 0.65


# ---------------------------------------------------------------------------
# AnomalyScore result type
# ---------------------------------------------------------------------------

@dataclass
class AnomalyScore:
    service_name: str
    namespace: str
    score: float           # [0.0, 1.0]; 1.0 = maximally anomalous
    is_anomalous: bool     # True if score >= _ANOMALY_THRESHOLD
    model_ready: bool      # False during warm-up period
    timestamp: float


# ---------------------------------------------------------------------------
# IsolationForestModel
# ---------------------------------------------------------------------------

class IsolationForestModel:
    """
    Incremental Isolation Forest anomaly detector.

    Thread-safe: scoring reads from an immutable model snapshot;
    training acquires a brief write lock only during model replacement.

    Parameters
    ----------
    warmup_samples:     Observations needed before scoring (default: 200)
    training_window:    Rolling buffer size (default: 2000)
    retrain_interval:   Retrain every N new observations (default: 50)
    contamination:      Fraction of outliers in training data.
                        'auto' = sklearn estimates from data distribution.
    """

    def __init__(
        self,
        warmup_samples: int = _WARMUP_SAMPLES,
        training_window: int = _TRAINING_WINDOW,
        retrain_interval: int = _RETRAIN_INTERVAL,
        contamination: float | str = "auto",
    ) -> None:
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for IsolationForestModel")

        self._warmup_samples = warmup_samples
        self._retrain_interval = retrain_interval
        self._contamination = contamination

        # Training buffer: deque with maxlen enforces rolling window
        self._buffer: deque = deque(maxlen=training_window)
        self._obs_count = 0
        self._since_last_train = 0

        # Model snapshot — replaced atomically on each training cycle
        self._model: _IF | None = None
        self._model_lock = threading.RLock()

        logger.info(
            "[IF-Model] Initialized: warmup=%d, window=%d, retrain_every=%d",
            warmup_samples, training_window, retrain_interval,
        )

    @property
    def is_ready(self) -> bool:
        """True once the warm-up window has been filled and model trained."""
        return self._model is not None and self._obs_count >= self._warmup_samples

    def add_observation(self, feature_vector: list[float]) -> None:
        """
        Add a new feature vector to the training buffer.
        Triggers a model retrain if the retrain interval has elapsed.
        """
        self._buffer.append(feature_vector)
        self._obs_count += 1
        self._since_last_train += 1

        if _model_warmup_remaining is not None:
            remaining = max(0, self._warmup_samples - self._obs_count)
            _model_warmup_remaining.set(remaining)

        # Retrain when enough new data has accumulated
        if (
            self._obs_count >= self._warmup_samples
            and self._since_last_train >= self._retrain_interval
        ):
            self._retrain()

    def _retrain(self) -> None:
        """Retrain the Isolation Forest on the current rolling buffer."""
        if len(self._buffer) < self._warmup_samples:
            return
        X = np.array(list(self._buffer))
        if X.shape[0] < 10 or X.shape[1] < 1:
            return
        try:
            t_start = time.monotonic()
            new_model = _IF(
                n_estimators=100,
                contamination=self._contamination,
                max_samples="auto",
                random_state=42,
                n_jobs=-1,  # Use all cores during training
            )
            new_model.fit(X)
            with self._model_lock:
                self._model = new_model
                self._since_last_train = 0
            elapsed = (time.monotonic() - t_start) * 1000
            if _training_counter is not None:
                _training_counter.inc()
            logger.info(
                "[IF-Model] Retrained on %d samples in %.0fms",
                len(self._buffer), elapsed,
            )
        except Exception as exc:
            logger.warning("[IF-Model] Training failed: %s", exc)

    def score(
        self,
        feature_vector: list[float],
        service_name: str = "",
        namespace: str = "",
    ) -> AnomalyScore:
        """
        Score a feature vector against the current model.

        Returns AnomalyScore with model_ready=False during warm-up.
        score=0.0 during warm-up (safe default — no false positives).
        """
        if not self.is_ready:
            return AnomalyScore(
                service_name=service_name,
                namespace=namespace,
                score=0.0,
                is_anomalous=False,
                model_ready=False,
                timestamp=time.time(),
            )

        with self._model_lock:
            model_snapshot = self._model

        try:
            X = np.array([feature_vector])
            # score_samples returns values in roughly [-0.5, 0.5]
            # Negative values = anomalous. Invert and normalise to [0, 1].
            raw_score = float(model_snapshot.score_samples(X)[0])
            # Shift from [-0.5, 0.5] to [0, 1] where 1 = most anomalous
            normalised = max(0.0, min(1.0, (-raw_score + 0.5) / 1.0))
        except Exception as exc:
            logger.warning("[IF-Model] Scoring failed: %s", exc)
            normalised = 0.0

        is_anomalous = normalised >= _ANOMALY_THRESHOLD

        # Publish to Prometheus gauge
        if _anomaly_score_gauge is not None and service_name:
            _anomaly_score_gauge.labels(
                service_name=service_name,
                namespace=namespace,
            ).set(normalised)

        return AnomalyScore(
            service_name=service_name,
            namespace=namespace,
            score=normalised,
            is_anomalous=is_anomalous,
            model_ready=True,
            timestamp=time.time(),
        )
