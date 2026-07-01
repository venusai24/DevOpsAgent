"""Deduplication decision enumeration (Stage -1 classifier output)."""

from enum import Enum


class DedupDecision(str, Enum):
    """Classification of a new investigation relative to existing live rows.

    Used in investigation_registry.dedup_decision and
    InvestigationState.dedup_decision.
    """

    # No overlapping investigation exists in this cluster.
    NEW = "NEW"

    # Identical (cluster_id, symptom_signature) already live — never run.
    DUPLICATE = "DUPLICATE"

    # New investigation's window is fully contained within an existing one.
    SUBSET = "SUBSET"

    # New investigation's window fully contains one or more existing ones.
    SUPERSET = "SUPERSET"

    # Windows overlap but neither contains the other.
    PARTIAL_OVERLAP = "PARTIAL_OVERLAP"
