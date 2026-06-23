"""
Calibration Store — Module 1.4.

Loads and provides calibration thresholds for both the ConANN RAG engine
and the Conformal Risk Control engine.

Phase 1: Reads from calibration/defaults.json (hardcoded conservative defaults).
Phase 3: Will be replaced by output of scripts/calibrate.py (synthetic bootstrap).

The interface is unchanged between phases — only the backing data changes.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


class CalibrationStore:
    """
    Thread-safe, single-source calibration configuration.

    Loaded once at startup from calibration/defaults.json. Provides typed
    accessors for all calibration parameters used by risk and retrieval modules.
    """

    def __init__(self, calibration_path: Path) -> None:
        self._path = calibration_path
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load calibration data from JSON file."""
        if not self._path.exists():
            log.warning(
                "Calibration file not found at %s — using hardcoded emergency defaults",
                self._path,
            )
            self._data = self._emergency_defaults()
            return

        with open(self._path) as f:
            self._data = json.load(f)

        calibrated = self._data.get("_calibrated", False)
        version = self._data.get("_version", "unknown")
        if not calibrated:
            log.warning(
                "Using uncalibrated defaults (version=%s). "
                "Run scripts/calibrate.py to produce real calibration.",
                version,
            )
        else:
            log.info("Loaded calibration %s from %s", version, self._path)

    # ─── ConANN Calibration ───────────────────────────────────────────────────

    def get_collection_alpha(self, collection: str) -> float:
        """
        Get the coverage level α for a specific Qdrant collection.

        Lower α → stricter coverage guarantee (C3, C6).
        Higher α → more permissive (C5).
        """
        collections = self._data.get("conann", {}).get("collections", {})
        return collections.get(collection, {}).get(
            "alpha", self._data.get("conann", {}).get("default_alpha", 0.10)
        )

    def get_q_hat(self, collection: str) -> float:
        """
        Get the calibrated nonconformity quantile q̂_{1-α} for a collection.

        Used in ConANN radius expansion: s_i = 1 - cosim(q, c) ≤ q̂.
        Phase 1: conservative default of 0.50.
        """
        collections = self._data.get("conann", {}).get("collections", {})
        return collections.get(collection, {}).get("q_hat", 0.50)

    def get_max_expansions(self, collection: str) -> int:
        collections = self._data.get("conann", {}).get("collections", {})
        return collections.get(collection, {}).get("max_expansions", 5)

    def get_typical_k(self, collection: str) -> int:
        collections = self._data.get("conann", {}).get("collections", {})
        return collections.get(collection, {}).get("typical_k", 3)

    # ─── Risk Calibration ─────────────────────────────────────────────────────

    def get_nonconformity_weights(self) -> tuple[float, float]:
        """
        Returns (w1, w2) weights for the combined nonconformity score.

        s_k = w1 * LI_score + w2 * (1 - BERT_F1)
        Default: w1=0.40, w2=0.60.
        """
        weights = self._data.get("risk", {}).get("nonconformity_weights", {})
        w1 = weights.get("w1_li_score", 0.40)
        w2 = weights.get("w2_bert_f1", 0.60)
        return w1, w2

    def get_bert_score_threshold(self) -> float:
        """BERTScore F1 below this → mark interpretation as low quality."""
        return (
            self._data.get("risk", {})
            .get("bert_score", {})
            .get("low_quality_threshold", 0.30)
        )

    def get_bert_score_model(self) -> str:
        return (
            self._data.get("risk", {})
            .get("bert_score", {})
            .get("model_type", "microsoft/deberta-xlarge-mnli")
        )

    def get_sigmoid_scale(self) -> float:
        """Scale factor for sigmoid normalization of step risk."""
        return self._data.get("risk", {}).get("trajectory", {}).get("sigmoid_scale", 10.0)

    # ─── Context Calibration ──────────────────────────────────────────────────

    def get_scorer_weights(self) -> dict[str, float]:
        return self._data.get("context", {}).get(
            "scorer_weights",
            {
                "relevance": 0.30,
                "recency": 0.15,
                "causal_importance": 0.25,
                "uniqueness": 0.15,
                "diagnostic_value": 0.15,
            },
        )

    def get_recency_decay_lambda(self) -> float:
        """λ for recency exponential decay: score = exp(-λ * seconds_ago)."""
        return self._data.get("context", {}).get("recency_decay_lambda", 0.0002)

    def get_eviction_score_threshold(self) -> float:
        """Evidence with composite score below this is candidate for eviction."""
        return self._data.get("context", {}).get("eviction_score_threshold", 0.30)

    def get_eviction_pressure_threshold(self) -> float:
        """Active evidence utilization above this triggers eviction procedure."""
        return self._data.get("context", {}).get("eviction_pressure_threshold", 0.85)

    # ─── Emergency defaults ───────────────────────────────────────────────────

    @staticmethod
    def _emergency_defaults() -> dict[str, Any]:
        """Minimal hardcoded defaults used only if calibration file is missing."""
        return {
            "_version": "emergency",
            "_calibrated": False,
            "conann": {
                "default_alpha": 0.10,
                "collections": {},
            },
            "risk": {
                "nonconformity_weights": {"w1_li_score": 0.40, "w2_bert_f1": 0.60},
                "bert_score": {"low_quality_threshold": 0.30, "model_type": "microsoft/deberta-xlarge-mnli"},
                "trajectory": {"sigmoid_scale": 10.0},
            },
            "context": {
                "scorer_weights": {
                    "relevance": 0.30,
                    "recency": 0.15,
                    "causal_importance": 0.25,
                    "uniqueness": 0.15,
                    "diagnostic_value": 0.15,
                },
                "recency_decay_lambda": 0.0002,
                "eviction_score_threshold": 0.30,
                "eviction_pressure_threshold": 0.85,
            },
        }


# ─── Module-level singleton ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_calibration_store() -> CalibrationStore:
    """
    Return the global CalibrationStore singleton.

    Loaded once from calibration/defaults.json relative to the project root.
    Import and call this everywhere instead of constructing CalibrationStore directly.
    """
    from airs.config import settings
    return CalibrationStore(settings.calibration_dir / "defaults.json")
