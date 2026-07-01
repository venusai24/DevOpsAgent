"""Root-cause confidence level enumeration (ADR-001 §Stage 7)."""

from enum import Enum


class ConfidenceLevel(str, Enum):
    """The four tiers of investigation confidence assigned at Stage 7.

    HIGH requires all four evidence sources to agree and no forced exits.
    MEDIUM is the ceiling when Stage 5.1b (trace degradation) or a forced
    early exit occurred.
    LOW is applied when a single-survivor hypothesis remains with score < 0.5
    or when the metric fallback quality was ``acceptable``.
    INCONCLUSIVE is returned when no hypothesis survives elimination.
    """

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INCONCLUSIVE = "INCONCLUSIVE"
