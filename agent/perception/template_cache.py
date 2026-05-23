"""
agent/perception/template_cache.py

L1 Symbolic Cache — Deterministic regex-based log classification.

The L1 cache is the first and fastest tier in the neuro-symbolic perception
pipeline. It matches incoming log entries against a registry of known error
patterns using compiled regular expressions, achieving near-zero compute cost.

When the L3 LLM extracts a new template from a novel log pattern, it is
registered here for future fast-path matching (continuous learning loop).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TemplateMatch:
    """Result of an L1 template cache lookup."""
    template_key: str
    pattern: str
    confidence: float = 1.0
    tier: str = "L1"


# ---------------------------------------------------------------------------
# Built-in known error templates
# ---------------------------------------------------------------------------
# Each entry: (key, regex_pattern, description)

_BUILTIN_TEMPLATES: list[tuple[str, str]] = [
    (
        "connection_pool_exhausted",
        r"(QueuePool\s+limit|pool\s+exhausted|connection\s+pool.*100|connections.*in\s+use|pool\s+timed\s+out)",
    ),
    (
        "oom_killed",
        r"(OOMKilled|OutOfMemoryError|java\.lang\.OutOfMemoryError|Exit\s+code\s+137|evicting.*OOM|heap\s+space)",
    ),
    (
        "dns_resolution_failure",
        r"(name\s+resolution|gaierror|CoreDNS.*timeout|CoreDNS.*rate.limit|Errno\s+-3|socket\.gaierror)",
    ),
    (
        "disk_space_exhausted",
        r"(No\s+space\s+left\s+on\s+device|disk.*100%|partition.*99%|IOException.*space|write.ahead\s+log.*disk)",
    ),
    (
        "tls_cert_expired",
        r"(CERTIFICATE_VERIFY_FAILED|CertificateExpiredException|PKIX\s+path|SSL.*handshake|NotAfter|validity\s+check\s+failed)",
    ),
    (
        "upstream_rate_limited",
        r"(429\s+Too\s+Many\s+Requests|Retry-After|rate.limit|exhausted.*retries|circuit.breaker.*open)",
    ),
    (
        "database_query_timeout",
        r"(statement\s+timeout|query\s+exceeded.*execution|canceling\s+statement|SQLException.*timeout|max\s+connections.*reached)",
    ),
    (
        "redis_oom_eviction",
        r"(maxmemory.*limit\s+reached|OOM\s+command\s+not\s+allowed|used\s+memory.*maxmemory|eviction.*volatile)",
    ),
    (
        "pod_crash_loop",
        r"(CrashLoopBackOff|Back-off\s+restarting|container.*OOMKilled|Liveness\s+probe\s+failed)",
    ),
    (
        "http_upstream_unavailable",
        r"(503\s+Service\s+Unavailable|upstream\s+connect\s+error|connection\s+refused|no\s+healthy\s+upstream)",
    ),
    (
        "transaction_leak",
        r"(Long-running\s+transaction|connection\s+leak|open\s+for\s+\d+\s+seconds|possible\s+connection\s+leak)",
    ),
    (
        "thread_pool_exhausted",
        r"(Thread\s+pool\s+exhausted|Request\s+queue.*maximum|Rejecting\s+new\s+incoming|thread.*blocked)",
    ),
]


class TemplateCache:
    """
    L1 Symbolic Cache for log pattern recognition.

    Provides O(n_templates) deterministic classification of log entries
    with no network calls, no model inference, and no memory allocation
    beyond the compiled regex objects.

    New templates learned by the L3 LLM layer are registered at runtime
    via register_template(), closing the continuous learning loop.
    """

    def __init__(self) -> None:
        # Map of template_key -> compiled regex
        self._registry: dict[str, re.Pattern] = {}
        self._hit_counts: dict[str, int] = {}  # For observability
        self._total_queries: int = 0
        self._l1_hits: int = 0

        # Register all built-in templates
        for key, pattern in _BUILTIN_TEMPLATES:
            self.register_template(key, pattern)

    def register_template(self, key: str, pattern: str) -> None:
        """
        Register a new or updated regex template.

        Called at initialization for built-in templates, and at runtime
        when the L3 LLM discovers a novel log pattern.

        Args:
            key: Unique template identifier (snake_case).
            pattern: Raw regex pattern string.
        """
        try:
            self._registry[key] = re.compile(pattern, re.IGNORECASE)
            self._hit_counts[key] = self._hit_counts.get(key, 0)
            logger.debug("[L1Cache] Registered template: %s", key)
        except re.error as exc:
            logger.warning("[L1Cache] Invalid regex for template '%s': %s", key, exc)

    def classify(self, log_entry: str) -> Optional[TemplateMatch]:
        """
        Attempt to classify a log entry against the known template registry.

        Args:
            log_entry: A single log line or concatenated log block.

        Returns:
            TemplateMatch if a pattern matches, None if L2/L3 lookup is needed.
        """
        self._total_queries += 1
        for key, pattern in self._registry.items():
            if pattern.search(log_entry):
                self._hit_counts[key] = self._hit_counts.get(key, 0) + 1
                self._l1_hits += 1
                return TemplateMatch(
                    template_key=key,
                    pattern=pattern.pattern,
                    confidence=1.0,
                    tier="L1",
                )
        return None

    @property
    def stats(self) -> dict:
        """Return L1 cache hit statistics for observability."""
        hit_rate = self._l1_hits / max(1, self._total_queries)
        return {
            "total_queries": self._total_queries,
            "l1_hits": self._l1_hits,
            "hit_rate": round(hit_rate, 3),
            "registered_templates": len(self._registry),
            "top_patterns": sorted(
                self._hit_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }

    @property
    def template_keys(self) -> list[str]:
        """Return all registered template keys."""
        return list(self._registry.keys())
