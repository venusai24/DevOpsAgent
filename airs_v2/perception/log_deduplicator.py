"""
airs_v2/perception/log_deduplicator.py
=======================================

TemplateDeduplicator — LogSieve-style semantic log deduplication.

Purpose
-------
Reduces LLM token consumption and prevents context dilution by collapsing
repeated log messages into templated, counted representations *before* the
forensic text is injected into the LLM context window.

This runs **after** ClickHouse query results are fetched (Option B from
the implementation plan), prior to ``ForensicQueryResult.to_text()``.

Algorithm
---------
1. Normalise each log message into a **canonical template** by stripping
   dynamic fields: decimal/hex integers, UUIDs, IPv4 addresses, quoted
   string literals, timestamps, and file paths with line numbers.

2. Group rows by (severity, container_name, canonical_template).

3. Emit one representative ``LogRow`` per group, annotated with the
   occurrence count. The representative is the **most recent** row
   (highest timestamp) so the LLM sees the latest incarnation.

4. For groups with count > 1 the message is prefixed with ``[×N] ``
   so the LLM knows this pattern recurred.

5. Novel rows (those whose template is seen only once) pass through
   unchanged, preserving all structurally unique failure signals.

Token savings
-------------
Empirical target: 30-42% reduction consistent with LogSieve benchmarks
(42% line reduction, 40% token reduction, cosine similarity 0.93).

Integration
-----------
Called from ``clickhouse_log_client.ForensicQueryResult.to_text()``
after the tiered query completes and before text rendering.

Usage
-----
    from airs_v2.perception.log_deduplicator import TemplateDeduplicator

    deduplicator = TemplateDeduplicator()
    deduplicated_rows, stats = deduplicator.deduplicate(log_rows)
"""

from __future__ import annotations

import re
import logging
from collections import defaultdict
from dataclasses import dataclass, replace as dc_replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid circular import — clickhouse_log_client imports us
    from clickhouse_log_client import LogRow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template normalisation patterns
# Applied in order — order matters; longer/more-specific patterns first.
# ---------------------------------------------------------------------------

_TEMPLATE_SUBS: list[tuple[re.Pattern, str]] = [
    # UUIDs (must precede generic hex)
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE), "?uuid"),
    # IPv4 addresses
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?\b"), "?ip"),
    # Timestamps within messages (ISO 8601 and common variants)
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"), "?ts"),
    # Java / Python file references: ClassName.java:42, module.py:123
    (re.compile(r"\b\w+\.(?:java|py|go|kt|scala|rb|js|ts):\d+\b"), "?srcref"),
    # Long hex strings (container IDs, pod UIDs, Git SHAs)
    (re.compile(r"\b[0-9a-f]{12,}\b", re.IGNORECASE), "?hex"),
    # Numeric IDs, port numbers, status codes, counts
    (re.compile(r"\b\d+\b"), "?n"),
    # Quoted strings (connection strings, SQL fragments, file paths in quotes)
    (re.compile(r'"[^"]{4,64}"'), "?str"),
    (re.compile(r"'[^']{4,64}'"), "?str"),
    # Absolute file paths
    (re.compile(r"/[\w./\-]+\.\w{1,8}"), "?path"),
    # Collapse multiple whitespace
    (re.compile(r"\s+"), " "),
]


def _canonicalise(message: str) -> str:
    """
    Reduce a raw log message to its canonical template string.

    All dynamic fields (numbers, IDs, paths, timestamps) are replaced with
    placeholder tokens so that semantically identical messages with different
    dynamic values collapse to the same template key.
    """
    text = message.strip()
    for pattern, replacement in _TEMPLATE_SUBS:
        text = pattern.sub(replacement, text)
    # Truncate to 200 chars — template keys beyond this are never unique
    return text[:200].strip()


# ---------------------------------------------------------------------------
# Deduplication statistics
# ---------------------------------------------------------------------------

@dataclass
class DeduplicationStats:
    """
    Metadata about a deduplication pass.

    Attributes
    ----------
    original_count:      Total log rows input to the deduplicator.
    deduplicated_count:  Rows after deduplication (unique templates + singletons).
    suppressed_count:    Rows collapsed into multi-occurrence representatives.
    dedup_ratio:         Fraction of rows eliminated (0.0 = no change, 1.0 = all collapsed).
    group_count:         Number of distinct template groups identified.
    """
    original_count: int
    deduplicated_count: int
    suppressed_count: int
    dedup_ratio: float
    group_count: int


# ---------------------------------------------------------------------------
# TemplateDeduplicator
# ---------------------------------------------------------------------------

class TemplateDeduplicator:
    """
    Groups semantically equivalent log rows and emits one representative
    per group, annotated with occurrence count.

    Thread-safe: stateless, all state lives in local variables per call.

    Parameters
    ----------
    max_representative_message_len:
        Truncate the representative message to this length in the output.
        Default 512 — balances context fidelity with token budget.
    min_occurrences_for_suppression:
        Groups with fewer than this many occurrences are NOT collapsed;
        each row is emitted individually. Default 2 — only suppress when
        a pattern genuinely repeats.
    """

    def __init__(
        self,
        max_representative_message_len: int = 512,
        min_occurrences_for_suppression: int = 2,
    ) -> None:
        self._max_msg_len = max_representative_message_len
        self._min_suppress = min_occurrences_for_suppression

    def deduplicate(
        self,
        rows: "list[LogRow]",
    ) -> "tuple[list[LogRow], DeduplicationStats]":
        """
        Deduplicate a list of ``LogRow`` objects.

        Returns
        -------
        (deduplicated_rows, stats)
            ``deduplicated_rows`` is sorted by timestamp ascending
            (preserving chronological order for the LLM).
            ``stats`` carries metadata for observability logging.
        """
        if not rows:
            return [], DeduplicationStats(
                original_count=0,
                deduplicated_count=0,
                suppressed_count=0,
                dedup_ratio=0.0,
                group_count=0,
            )

        original_count = len(rows)

        # ── Group rows by (severity, container_name, canonical_template) ──
        # Key: tuple of (severity_upper, container_name, template)
        groups: dict[tuple[str, str, str], list] = defaultdict(list)
        for row in rows:
            key = (
                row.severity.upper(),
                row.container_name,
                _canonicalise(row.message),
            )
            groups[key].append(row)

        # ── Emit representatives ──────────────────────────────────────────
        output_rows: list = []
        for key, group_rows in groups.items():
            count = len(group_rows)

            if count < self._min_suppress:
                # Below suppression threshold — emit individually, unchanged
                output_rows.extend(group_rows)
                continue

            # Sort group by timestamp descending to pick the most recent
            group_rows.sort(key=lambda r: r.timestamp, reverse=True)
            representative = group_rows[0]  # Most recent incarnation

            # Annotate message with occurrence count
            msg = representative.message
            if len(msg) > self._max_msg_len:
                msg = msg[: self._max_msg_len] + " …[truncated]"
            annotated_msg = f"[×{count}] {msg}"

            # Create annotated representative (immutable dataclass copy)
            annotated = dc_replace(representative, message=annotated_msg)
            output_rows.append(annotated)

        # Sort output chronologically (ascending) for LLM context coherence
        output_rows.sort(key=lambda r: r.timestamp)

        deduplicated_count = len(output_rows)
        suppressed_count = original_count - deduplicated_count
        dedup_ratio = suppressed_count / original_count if original_count > 0 else 0.0

        stats = DeduplicationStats(
            original_count=original_count,
            deduplicated_count=deduplicated_count,
            suppressed_count=suppressed_count,
            dedup_ratio=round(dedup_ratio, 4),
            group_count=len(groups),
        )

        logger.debug(
            "[TemplateDeduplicator] %d rows → %d unique templates "
            "(%.0f%% reduction, %d groups)",
            original_count,
            deduplicated_count,
            dedup_ratio * 100,
            len(groups),
        )

        return output_rows, stats


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

_default_deduplicator: TemplateDeduplicator | None = None


def deduplicate_log_rows(
    rows: "list[LogRow]",
) -> "tuple[list[LogRow], DeduplicationStats]":
    """
    Module-level convenience wrapper using a shared default deduplicator.

    Equivalent to ``TemplateDeduplicator().deduplicate(rows)`` but reuses
    a single instance across calls (the deduplicator is stateless, so this
    is safe and saves object creation overhead per query).
    """
    global _default_deduplicator
    if _default_deduplicator is None:
        _default_deduplicator = TemplateDeduplicator()
    return _default_deduplicator.deduplicate(rows)
