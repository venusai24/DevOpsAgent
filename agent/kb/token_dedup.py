"""
agent/kb/token_dedup.py

Log telemetry deduplication utility.

During cascading failures, investigate_node can accumulate thousands of
identical stack-trace lines across multiple tool-call rounds, rapidly
overflowing the LLM's context window and pushing critical root-cause
evidence out of its effective attention span.

This module compresses the telemetry string before it is passed to
extract_node and plan_node by:

  1. Splitting the telemetry into logical sections (the "---\n### Round N"
     or "---\n### [CACHED] Step N" headers written by investigate_node).
  2. Within each section, normalising log lines to remove volatile tokens
     (timestamps, UUIDs, pod names, IPs, hex addresses).
  3. Keeping the first occurrence of each normalised line and annotating it
     with a suppression count: "[×12 repetitions suppressed]".
  4. Hard-truncating the result at ``max_chars`` if it is still over the limit.

Usage:
    from agent.kb.token_dedup import deduplicate_telemetry

    compressed = deduplicate_telemetry(raw_telemetry, max_chars=16_000)
"""

from __future__ import annotations

import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CHARS: int = 16_000  # ~4 000 tokens at 4 chars/token
_MIN_LINE_LEN: int = 10           # Skip short separator / header lines


def deduplicate_telemetry(telemetry: str, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
    """
    Deduplicate repeated log lines in *telemetry* and hard-truncate to
    *max_chars* characters.

    Args:
        telemetry:  Raw concatenated telemetry string from state[\"telemetry\"].
        max_chars:  Maximum allowed length of the returned string.

    Returns:
        Compressed telemetry string, guaranteed to be <= max_chars characters.
        If compression was applied, a summary line is appended at the end.
    """
    if len(telemetry) <= max_chars:
        return telemetry

    original_len = len(telemetry)

    # Split into sections by the section-header markers investigate_node writes.
    # Pattern: "\n---\n### Round N" or "\n---\n### [CACHED] Step N"
    sections = re.split(r"(?=\n\n---\n###)", telemetry)

    compressed_sections = [_compress_section(sec) for sec in sections]
    result = "".join(compressed_sections)

    # Hard truncation if still over limit
    if len(result) > max_chars:
        result = (
            result[:max_chars]
            + f"\n\n[⚠ TELEMETRY TRUNCATED: {original_len:,} chars → {max_chars:,} chars"
            f" to prevent context-window overflow]"
        )

    compressed_len = len(result)
    if compressed_len < original_len:
        logger.info(
            "[token_dedup] Compressed telemetry %d → %d chars (%.0f%% reduction)",
            original_len,
            compressed_len,
            (1 - compressed_len / original_len) * 100,
        )

    return result


def _compress_section(section: str) -> str:
    """
    Deduplicate repeated log lines within a single telemetry section.

    Keeps the first occurrence of each normalised line and annotates it with
    a suppression count.  Short lines (headers, separators, blank lines) are
    always preserved verbatim.
    """
    lines = section.split("\n")

    # Two-pass approach:
    # Pass 1: count normalised occurrences
    counts: Counter[str] = Counter()
    for line in lines:
        norm = _normalise_line(line)
        if norm:
            counts[norm] += 1

    # Pass 2: emit each unique normalised line once, annotated with count
    output_lines: list[str] = []
    seen_norms: set[str] = set()

    for line in lines:
        norm = _normalise_line(line)

        # Always keep short / structural lines
        if not norm:
            output_lines.append(line)
            continue

        if norm not in seen_norms:
            seen_norms.add(norm)
            count = counts[norm]
            if count > 1:
                output_lines.append(
                    f"{line}  [×{count} occurrences, {count - 1} suppressed]"
                )
            else:
                output_lines.append(line)
        # else: duplicate — silently drop

    return "\n".join(output_lines)


def _normalise_line(line: str) -> str:
    """
    Normalise a log line for deduplication purposes.

    Strips volatile tokens so that two lines that are semantically identical
    but differ only in timestamp, pod name, IP, or UUID are treated as the
    same line.

    Returns an empty string for short or blank lines (< ``_MIN_LINE_LEN``
    chars after stripping), which are always emitted verbatim.
    """
    # Remove ISO 8601 timestamps: 2024-01-15T14:23:45.123Z or 2024/01/15 14:23:45
    normalised = re.sub(
        r"\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?Z?",
        "<ts>",
        line,
    )
    # Remove RFC 3339 short timestamps: 14:23:45.123
    normalised = re.sub(r"\b\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b", "<ts>", normalised)
    # Remove UUIDs
    normalised = re.sub(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        "<uuid>",
        normalised,
        flags=re.IGNORECASE,
    )
    # Remove IPv4 addresses (with optional port)
    normalised = re.sub(r"\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?", "<ip>", normalised)
    # Remove hex addresses / memory locations
    normalised = re.sub(r"0x[0-9a-fA-F]+", "<hex>", normalised)
    # Remove Kubernetes-style pod name suffixes: -7d9f8b-xkzp2
    normalised = re.sub(r"-[a-z0-9]{5,10}-[a-z0-9]{5}\b", "-<pod-suffix>", normalised)
    # Remove bare integers (e.g. PID, port numbers, retry counts)
    normalised = re.sub(r"\b\d{4,}\b", "<n>", normalised)
    # Collapse whitespace
    normalised = re.sub(r"\s+", " ", normalised).strip()

    if len(normalised) < _MIN_LINE_LEN:
        return ""  # Treat as structural line — always emit verbatim

    return normalised
