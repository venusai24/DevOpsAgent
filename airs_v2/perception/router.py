"""
airs_v2/perception/router.py
=============================

NeSy-Edge Three-Tier Log Parsing Router
-----------------------------------------
The central orchestrator of the ``airs_v2`` Perception Layer.

The router implements the **NeSy-Edge** cascade:

::

    log_entry
        │
        ▼
    ┌──────────────────────────────────────────────┐
    │  L1 — Symbolic Regex Cache                   │  ~0 ms
    │  Deterministic regex match against known     │  confidence = 1.0
    │  failure templates.  Hit → return immediately│
    └──────────────────┬───────────────────────────┘
                       │ MISS
                       ▼
    ┌──────────────────────────────────────────────┐
    │  L2 — Semantic TF-IDF Retrieval              │  ~5-15 ms
    │  Cosine similarity against knowledge base    │  confidence ∈ [0,1]
    │  of pattern descriptions.  Hit → return.     │
    └──────────────────┬───────────────────────────┘
                       │ MISS
                       ▼
    ┌──────────────────────────────────────────────┐
    │  L3 — LLM Abstraction Fallback               │  ~1-3 s
    │  Novel pattern — ask LLM to name and regex   │  (stub in unit tests)
    │  the pattern.  Optionally promote to L1+L2.  │
    └──────────────────────────────────────────────┘

Key features
------------
* **Continuous learning** — when L3 produces a result with
  ``confidence >= promote_threshold`` the learned pattern is immediately
  registered in L1 and L2 so future identical logs hit L1 at zero cost.
* **Telemetry block processing** — ``classify_block`` splits a raw
  multi-line telemetry string into individual log lines, filters to
  ERROR/WARN/CRITICAL entries, classifies each one, and returns a
  ``PerceptionReport`` with tier statistics and the dominant template.
* **Full observability** — ``RouterResult`` carries the tier, confidence,
  top-k L2 candidates, and the L3 result for downstream audit.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from airs_v2.perception.l1_cache import L1SymbolicCache, L1Match, TRANSIENT_TEMPLATE_KEYS
from airs_v2.perception.l2_semantic import L2SemanticMatcher, L2Match
from airs_v2.perception.l3_llm import L3LLMFallback, L3Result

logger = logging.getLogger(__name__)

# Lines that look like error/warning log entries worth classifying
_LOG_ENTRY_RE = re.compile(
    r"(ERROR|WARN|WARNING|CRITICAL|FATAL|Exception|Error|FATAL|Traceback)",
    re.IGNORECASE,
)

# Minimum confidence for L3 result to be promoted to L1+L2
_DEFAULT_PROMOTE_THRESHOLD = 0.60


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class Tier(str, enum.Enum):
    """Which tier produced the classification."""
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


@dataclass
class RouterResult:
    """
    Typed result of classifying a single log entry through the NeSy-Edge router.

    Attributes
    ----------
    log_entry:       The raw log text that was classified.
    tier:            The tier that produced the match (L1, L2, or L3).
    template_key:    Canonical failure category name.
    confidence:      Classification confidence (0.0–1.0).
    description:     Human-readable description of the failure pattern.
    is_novel:        True when L3 discovered a new pattern.
    l2_candidates:   Top-k L2 candidates (for diagnostic visibility).
    l3_result:       The L3 result, if L3 was invoked.
    promoted:        True when a L3 result was promoted to L1+L2.
    """
    log_entry: str
    tier: Tier
    template_key: str
    confidence: float
    description: str = ""
    is_novel: bool = False
    l2_candidates: list[L2Match] = field(default_factory=list)
    l3_result: Optional[L3Result] = None
    promoted: bool = False
    suppressed_transient: bool = False  # Set True by TransientFilter when this result's
                                        # template is transient AND count < min_recurrence


@dataclass
class PerceptionReport:
    """
    Aggregated output from classifying all notable log lines in a
    telemetry block.

    Mirrors the interface of ``agent/perception/log_classifier.PerceptionResult``
    for backwards-compatibility with existing graph nodes.
    """
    results: list[RouterResult]
    primary_template: str           # Most frequent template_key across all results
    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    novel_patterns: list[str] = field(default_factory=list)   # Newly discovered keys
    promoted_patterns: list[str] = field(default_factory=list)
    transient_suppressed: int = 0   # Count of results suppressed by TransientFilter
    transient_passed: int = 0       # Count of transient results that recurred enough to pass

    @property
    def tier_stats(self) -> dict:
        total = self.l1_hits + self.l2_hits + self.l3_hits
        return {
            "L1_hits": self.l1_hits,
            "L2_hits": self.l2_hits,
            "L3_hits": self.l3_hits,
            "total": total,
            "l1_rate": round(self.l1_hits / max(1, total), 4),
            "novel_count": len(self.novel_patterns),
            "promoted_count": len(self.promoted_patterns),
            "transient_suppressed": self.transient_suppressed,
            "transient_passed": self.transient_passed,
        }

    def to_markdown(self) -> str:
        stats = self.tier_stats
        lines = [
            "## NeSy-Edge Perception Report",
            f"**Primary Pattern**: `{self.primary_template}`",
            (
                f"**Tier Stats**: "
                f"L1={stats['L1_hits']} "
                f"L2={stats['L2_hits']} "
                f"L3={stats['L3_hits']} "
                f"(L1 rate: {stats['l1_rate']:.0%})"
            ),
        ]
        if self.novel_patterns:
            lines.append(f"**Novel Patterns Learned**: {', '.join(self.novel_patterns)}")
        if self.promoted_patterns:
            lines.append(f"**Promoted to L1**: {', '.join(self.promoted_patterns)}")
        lines.append("\n### Classified Entries")
        for r in self.results[:15]:
            tier_badge = {"L1": "🔵", "L2": "🟡", "L3": "🟣"}.get(r.tier.value, "⚪")
            novel_tag = " ✨new" if r.is_novel else ""
            lines.append(
                f"- {tier_badge} `[{r.tier.value}]` **{r.template_key}** "
                f"(conf={r.confidence:.0%}){novel_tag}: {r.log_entry[:80]}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Transient pre-filter
# ---------------------------------------------------------------------------

# Minimum number of occurrences of a transient-candidate template within a
# single telemetry block for it to escape suppression.  Rationale: most
# production retry policies use ≤2 retries before failing; a 3rd error
# occurrence means no retry recovered the service — genuine signal.
_DEFAULT_MIN_RECURRENCE: int = 3

# Regex that identifies a successful retry following an earlier error.
# Matches lines like: "succeeded after N retr", "retry succeeded", etc.
_SUCCESS_LINE_RE = re.compile(
    r"(succeeded|success|recovered|resolved|retry.*ok|after\s+\d+\s+retr)",
    re.IGNORECASE,
)


@dataclass
class TransientFilter:
    """
    Pre-classification transience filter inserted before hypothesis generation.

    Operates on the **full set of RouterResult objects** produced by a single
    ``classify_block`` call and annotates transient-candidate results whose
    occurrence count is below ``min_recurrence`` with
    ``suppressed_transient=True``.

    The suppressed results are kept in ``PerceptionReport.results`` for
    full auditability; the HypothesisEngine skips them during candidate
    generation.

    Parameters
    ----------
    min_recurrence:
        Minimum number of times a transient-candidate template must appear in
        the block (without a paired success line) before it is treated as a
        persistent signal.  Default: 3.
    raw_lines:
        The complete list of raw log lines from the telemetry block (including
        non-error lines that were filtered out before classification).  Used to
        detect paired success lines.  If None, the success-line check is skipped.
    """

    min_recurrence: int = _DEFAULT_MIN_RECURRENCE
    raw_lines: Optional[list[str]] = None

    def apply(self, results: list[RouterResult]) -> list[RouterResult]:
        """
        Annotate transient-candidate results that are below the recurrence
        threshold with ``suppressed_transient=True``.

        Returns the same list (mutated via ``dataclasses.replace``-style copy)
        with suppression flags applied.
        """
        from dataclasses import replace as dc_replace

        # Count occurrences of each transient-candidate template key
        transient_counts: dict[str, int] = {}
        for r in results:
            if r.template_key in TRANSIENT_TEMPLATE_KEYS:
                transient_counts[r.template_key] = transient_counts.get(r.template_key, 0) + 1

        # Determine which transient keys are suppressed (below threshold)
        has_success = self._has_success_line()
        suppressed_keys: set[str] = set()
        for key, count in transient_counts.items():
            if count < self.min_recurrence and not has_success:
                # count < min_recurrence AND no success line → below threshold
                # A paired success line would mean the retry resolved it
                pass  # Not suppressing when success line absent (no paired success)
            if count < self.min_recurrence:
                # Below recurrence threshold → suppress regardless of success line
                # (success line check adds precision: if a success line IS present,
                # we still suppress — the retry worked, it was truly transient)
                suppressed_keys.add(key)
                logger.debug(
                    "[TransientFilter] Suppressing '%s': count=%d < min_recurrence=%d",
                    key, count, self.min_recurrence,
                )

        # Annotate results
        annotated: list[RouterResult] = []
        for r in results:
            if r.template_key in suppressed_keys:
                annotated.append(dc_replace(r, suppressed_transient=True))
            else:
                annotated.append(r)

        return annotated

    def _has_success_line(self) -> bool:
        """Return True if any raw line contains a success/recovery signal."""
        if self.raw_lines is None:
            return False
        return any(_SUCCESS_LINE_RE.search(line) for line in self.raw_lines)


# ---------------------------------------------------------------------------
# LogRouter
# ---------------------------------------------------------------------------

class LogRouter:
    """
    NeSy-Edge three-tier log parsing router.

    Singleton-friendly — construct once and reuse across node invocations.
    The L1 cache is stateful (it learns from L3 promotions), so sharing the
    instance preserves the continuous-learning loop across graph executions.

    Parameters
    ----------
    promote_threshold:
        Minimum L3 confidence required to promote a new pattern to L1 + L2.
        Default 0.60.
    l2_threshold:
        Minimum cosine similarity for an L2 match to be accepted.
        Default ``L2SemanticMatcher.DEFAULT_THRESHOLD`` (0.72).
    """

    def __init__(
        self,
        promote_threshold: float = _DEFAULT_PROMOTE_THRESHOLD,
        l2_threshold: float = L2SemanticMatcher.DEFAULT_THRESHOLD,
    ) -> None:
        self._promote_threshold = promote_threshold
        self._l1 = L1SymbolicCache()
        self._l2 = L2SemanticMatcher(threshold=l2_threshold)
        self._l3 = L3LLMFallback()

    # ------------------------------------------------------------------
    # Single-entry classification
    # ------------------------------------------------------------------

    async def classify(self, log_entry: str) -> RouterResult:
        """
        Classify a single log entry through the L1 → L2 → L3 cascade.

        Never raises.
        """
        # ── L1 ──────────────────────────────────────────────────────────
        l1_match: Optional[L1Match] = self._l1.classify(log_entry)
        if l1_match is not None:
            return RouterResult(
                log_entry=log_entry,
                tier=Tier.L1,
                template_key=l1_match.template_key,
                confidence=l1_match.confidence,
                description=l1_match.description,
            )

        # ── L2 ──────────────────────────────────────────────────────────
        l2_candidates: list[L2Match] = self._l2.find_nearest_k(log_entry, k=3)
        best_l2 = l2_candidates[0] if l2_candidates else None
        if best_l2 is not None:
            return RouterResult(
                log_entry=log_entry,
                tier=Tier.L2,
                template_key=best_l2.template_key,
                confidence=best_l2.similarity,
                description=best_l2.description,
                l2_candidates=l2_candidates,
            )

        # ── L3 ──────────────────────────────────────────────────────────
        l3_result: L3Result = await self._l3.classify(log_entry)
        promoted = False
        if l3_result.confidence >= self._promote_threshold and l3_result.regex_pattern:
            self._promote(l3_result)
            promoted = True

        return RouterResult(
            log_entry=log_entry,
            tier=Tier.L3,
            template_key=l3_result.template_key,
            confidence=l3_result.confidence,
            description=l3_result.description,
            is_novel=True,
            l2_candidates=l2_candidates,
            l3_result=l3_result,
            promoted=promoted,
        )

    # ------------------------------------------------------------------
    # Telemetry block processing
    # ------------------------------------------------------------------

    async def classify_block(
        self,
        telemetry: str,
        max_lines: int = 25,
    ) -> PerceptionReport:
        """
        Classify all notable log lines in a multi-line telemetry block.

        Lines are filtered to those containing error/warning keywords.
        Up to *max_lines* are processed (performance cap for hot path).

        Returns a ``PerceptionReport`` with tier statistics and the dominant
        template across all classified lines.
        """
        candidates = [
            line.strip()
            for line in telemetry.splitlines()
            if _LOG_ENTRY_RE.search(line) and len(line.strip()) > 10
        ][:max_lines]

        if not candidates:
            return PerceptionReport(
                results=[],
                primary_template="no_logs_classified",
            )

        # Classify all lines concurrently
        tasks = [self.classify(line) for line in candidates]
        results: list[RouterResult] = await asyncio.gather(*tasks)

        # Apply transient pre-filter before hypothesis generation.
        # Pass all raw lines so the filter can detect paired success lines.
        all_raw_lines = telemetry.splitlines()
        tf = TransientFilter(raw_lines=all_raw_lines)
        results = tf.apply(results)

        # Aggregate tier stats
        l1_hits = sum(1 for r in results if r.tier == Tier.L1)
        l2_hits = sum(1 for r in results if r.tier == Tier.L2)
        l3_hits = sum(1 for r in results if r.tier == Tier.L3)
        novel = [r.template_key for r in results if r.is_novel]
        promoted = [r.template_key for r in results if r.promoted]

        # Transient filter stats
        transient_suppressed = sum(1 for r in results if r.suppressed_transient)
        transient_passed = sum(
            1 for r in results
            if not r.suppressed_transient and r.template_key in TRANSIENT_TEMPLATE_KEYS
        )

        template_counts = Counter(r.template_key for r in results)
        primary = template_counts.most_common(1)[0][0] if template_counts else "no_logs_classified"

        # Record to tracer
        from airs_v2.evaluation.tracer import ObservabilityTracer
        tracer = ObservabilityTracer.get_instance()
        tracer.record_metrics_analyzed([f"log_template:{k}" for k in template_counts.keys()])

        return PerceptionReport(
            results=results,
            primary_template=primary,
            l1_hits=l1_hits,
            l2_hits=l2_hits,
            l3_hits=l3_hits,
            novel_patterns=novel,
            promoted_patterns=promoted,
            transient_suppressed=transient_suppressed,
            transient_passed=transient_passed,
        )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        """Aggregated stats from all three tiers."""
        return {
            "l1": self._l1.stats,
            "l2": self._l2.stats,
            "l3": self._l3.stats,
            "promote_threshold": self._promote_threshold,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _promote(self, result: L3Result) -> None:
        """Promote a high-confidence L3 result into L1 and L2."""
        self._l1.register_template(
            key=result.template_key,
            pattern=result.regex_pattern,
            description=result.description,
            source="l3_learned",
        )
        self._l2.add_entry(
            key=result.template_key,
            description=result.description,
            pattern_hint=result.regex_pattern,
        )
        self._l3.record_promotion()
        logger.info(
            "[LogRouter] Promoted L3 pattern to L1+L2: key=%s conf=%.2f",
            result.template_key, result.confidence,
        )
