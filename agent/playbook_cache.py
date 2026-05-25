"""
agent/playbook_cache.py

A keyword-fingerprint cache that stores validated investigation playbooks
(ordered sequences of tool calls) for known incident patterns.

Design goals:
  - Zero architectural refactoring: wires into investigate_node only.
  - No new graph nodes, edges, or state schema changes required.
  - On cache HIT: replays tool calls deterministically, skipping LLM invocation.
  - On cache MISS: LLM runs normally; if the investigation fully succeeds
    (reaches INVESTIGATION_COMPLETE) the tool call sequence is written back
    to the cache as a new playbook for future use.
  - Cache is a plain JSON file (agent/playbook_cache.json) so it can be
    inspected, edited, and committed to version control.

Fingerprinting strategy:
  - Extracts a small set of canonical symptom keywords from the alert title
    and description (e.g. "connection pool", "oom", "dns").
  - Combines them into a frozenset-based fingerprint string to ensure
    order-independent matching.
  - This keeps matching deterministic and human-auditable without requiring
    embedding models or vector databases.

Cache entry format (one key in playbook_cache.json):
  {
    "oom+eviction": {
      "keywords": ["oom", "eviction", "memory"],
      "tool_sequence": [
        {"tool": "get_metrics", "args": {"query": "memory_utilization", "time_range": "last_15m"}},
        {"tool": "get_logs",    "args": {"service": "{service}",       "time_range": "last_15m"}}
      ],
      "hit_count": 3
    }
  }

  The "{service}" placeholder in args is substituted at runtime with the
  actual service name from the triage result, so the same playbook works
  across different microservices exhibiting the same symptom pattern.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File location — sibling of this module file
# ---------------------------------------------------------------------------

_CACHE_PATH: Path = Path(__file__).parent / "playbook_cache.json"

# ---------------------------------------------------------------------------
# Canonical symptom keyword sets
# ---------------------------------------------------------------------------

# Maps a canonical symptom label to a set of trigger keywords.
# Any single keyword match is sufficient to contribute the label to the fingerprint.
_SYMPTOM_KEYWORDS: dict[str, list[str]] = {
    "connection_pool": [
        "connection pool", "pool exhausted", "pool saturation", "queuepool",
        "postgresql", "pg_stat_activity", "db connections", "database connection",
    ],
    "oom": [
        "oom", "oomkilled", "out of memory", "outofmemoryerror",
        "java heap", "evicted", "exit code 137", "memory limit",
        "heap space", "memory usage",
    ],
    "dns": [
        "dns", "coredns", "name resolution", "gaierror", "nxdomain",
        "dns throttling", "dns timeout", "dns rate limit",
    ],
    "high_error_rate": [
        "error rate", "5xx", "http 500", "http 503", "request failure",
        "spike in errors",
    ],
    "disk_full": [
        "disk", "no space left", "storage full", "filesystem", "inode",
        "write failure", "disk exhaustion",
    ],
    "tls_cert": [
        "certificate", "tls", "ssl", "cert expir", "handshake failure",
        "pkix", "x509",
    ],
    "rate_limited": [
        "429", "rate limit", "too many requests", "throttl", "upstream rejection",
        "circuit breaker",
    ],
    "redis_oom": [
        "redis", "maxmemory", "eviction", "volatile-lru", "cache oom",
        "redisexception",
    ],
}


def _build_fingerprint(alert_title: str, alert_description: str = "") -> str:
    """
    Extract canonical symptom labels from the alert text and return a
    deterministic fingerprint string (sorted labels joined with '+').

    Example:
        alert_title = "CRITICAL: Pod OOMKilled — auth-service"
        → matched labels: {"oom"}
        → fingerprint: "oom"

    Returns an empty string if no known symptoms are detected.
    """
    combined = (alert_title + " " + alert_description).lower()
    matched_labels: set[str] = set()

    for label, keywords in _SYMPTOM_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            matched_labels.add(label)

    if not matched_labels:
        logger.debug("[playbook_cache] No symptom labels matched for fingerprint.")
        return ""

    fingerprint = "+".join(sorted(matched_labels))
    logger.debug("[playbook_cache] Fingerprint: %r from labels=%s", fingerprint, matched_labels)
    return fingerprint


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _load_cache() -> dict[str, Any]:
    """Load the JSON cache file. Returns empty dict if file does not exist."""
    if not _CACHE_PATH.exists():
        return {}
    try:
        with _CACHE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("[playbook_cache] Failed to load cache: %s", exc)
        return {}


def _save_cache(cache: dict[str, Any]) -> None:
    """Persist the cache dictionary back to JSON. Silently skips on I/O error."""
    try:
        with _CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        logger.debug("[playbook_cache] Cache saved to %s", _CACHE_PATH)
    except OSError as exc:
        logger.warning("[playbook_cache] Failed to save cache: %s", exc)


# ---------------------------------------------------------------------------
# Public API used by investigate_node
# ---------------------------------------------------------------------------

def lookup_playbook(
    alert_title: str,
    alert_description: str = "",
) -> tuple[str, list[dict[str, Any]]] | None:
    """
    Look up a cached playbook based on the alert fingerprint.

    Returns:
        (fingerprint, tool_sequence) if a playbook is found, otherwise None.
        The returned tool_sequence is a list of dicts:
            [{"tool": "get_metrics", "args": {...}}, ...]
    """
    fingerprint = _build_fingerprint(alert_title, alert_description)
    if not fingerprint:
        return None

    cache = _load_cache()
    entry = cache.get(fingerprint)

    if not entry:
        logger.info("[playbook_cache] MISS for fingerprint=%r", fingerprint)
        return None

    logger.info(
        "[playbook_cache] HIT for fingerprint=%r  hit_count=%d",
        fingerprint,
        entry.get("hit_count", 0),
    )
    # Increment hit counter and persist
    entry["hit_count"] = entry.get("hit_count", 0) + 1
    cache[fingerprint] = entry
    _save_cache(cache)

    return fingerprint, entry["tool_sequence"]


def record_successful_playbook(
    alert_title: str,
    alert_description: str,
    tool_sequence: list[dict[str, Any]],
) -> None:
    """
    Write a successfully completed tool call sequence back into the cache.

    Only called by investigate_node when the LLM signals INVESTIGATION_COMPLETE,
    ensuring only verified, successful sequences are cached.

    Args:
        alert_title:      Raw alert title from state["raw_alert"]["title"].
        alert_description: Raw alert description from state["raw_alert"]["description"].
        tool_sequence:    Ordered list of {"tool": str, "args": dict} dicts collected
                          during the live LLM investigation run.
    """
    fingerprint = _build_fingerprint(alert_title, alert_description)
    if not fingerprint or not tool_sequence:
        return

    cache = _load_cache()

    if fingerprint in cache:
        # Playbook already exists — do not overwrite with a potentially
        # different sequence from this run. Keep the original validated sequence.
        logger.debug(
            "[playbook_cache] Fingerprint %r already exists, skipping write.", fingerprint
        )
        return

    # Extract unique symptom labels for human-readable metadata
    combined = (alert_title + " " + alert_description).lower()
    matched_keywords: list[str] = [
        kw
        for label_kws in _SYMPTOM_KEYWORDS.values()
        for kw in label_kws
        if kw in combined
    ]

    cache[fingerprint] = {
        "keywords": sorted(set(matched_keywords)),
        "tool_sequence": tool_sequence,
        "hit_count": 0,
    }
    _save_cache(cache)
    logger.info(
        "[playbook_cache] NEW playbook written for fingerprint=%r  steps=%d",
        fingerprint,
        len(tool_sequence),
    )


def substitute_service(tool_sequence: list[dict[str, Any]], service: str) -> list[dict[str, Any]]:
    """
    Replace the "{service}" placeholder in cached tool args with the actual
    service name resolved at runtime by the triage node.

    This allows a single cached playbook (e.g. for OOM incidents) to be
    applied to different microservices without creating separate cache entries.

    Args:
        tool_sequence: Raw sequence from the cache (may contain "{service}").
        service:       Canonical service name from state["triage_result"].service.

    Returns:
        A new list of dicts with "{service}" substituted.
    """
    result = []
    for step in tool_sequence:
        substituted_args = {
            k: v.replace("{service}", service) if isinstance(v, str) else v
            for k, v in step["args"].items()
        }
        result.append({"tool": step["tool"], "args": substituted_args})
    return result
