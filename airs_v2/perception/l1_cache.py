"""
airs_v2/perception/l1_cache.py
==============================

L1 — Symbolic Regex Cache
--------------------------
The fastest tier in the NeSy-Edge log parsing router.

Every incoming log line is matched against a registry of compiled regular
expressions that represent **known, named failure patterns**.  A match is
deterministic (confidence = 1.0) and takes O(N_templates) time with no
memory allocation beyond the pre-compiled regex objects.

Improvements over v1 (agent/perception/template_cache.py)
----------------------------------------------------------
* **Priority ordering** — templates are sorted by hit-count descending so
  the most frequently triggered patterns are tried first.
* **Provenance metadata** — every template carries a ``source`` tag
  ("builtin" | "l3_learned" | "user") for audit trails.
* **Runtime promotion** — templates learned by L3 are tagged and tracked
  separately from built-ins, letting the evaluation harness measure the
  continuous-learning loop quantitatively.
* **Snapshot / restore** — ``snapshot()`` / ``restore()`` enable test
  isolation without constructing a new instance.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Known failure templates
# ---------------------------------------------------------------------------
# Format: (key, regex_pattern, description)
# Patterns are case-insensitive and compiled once at import time.

BUILTIN_TEMPLATES: list[tuple[str, str, str]] = [
    (
        "connection_pool_exhausted",
        r"(QueuePool\s+limit|connection\s+pool\s+exhausted|connection\s+pool.*100|connections.*in\s+use|pool\s+timed\s+out|too\s+many\s+clients)",
        "Database/HTTP connection pool fully saturated",
    ),
    (
        "oom_killed",
        r"(OOMKilled|OutOfMemoryError|java\.lang\.OutOfMemoryError|Exit\s+code\s+137|evicting.*OOM|heap\s+space|killed.*memory\s+limit)",
        "Process killed by kernel OOM killer or JVM heap exhausted",
    ),
    (
        "dns_resolution_failure",
        r"(name\s+resolution|gaierror|CoreDNS.*timeout|CoreDNS.*rate.limit|Errno\s+-3|socket\.gaierror|NXDOMAIN|SERVFAIL)",
        "DNS lookup failed or timed out",
    ),
    (
        "disk_space_exhausted",
        r"(No\s+space\s+left\s+on\s+device|disk.*100%|partition.*99%|IOException.*space|write.ahead\s+log.*disk|ENOSPC)",
        "Filesystem at capacity",
    ),
    (
        "tls_cert_expired",
        r"(CERTIFICATE_VERIFY_FAILED|CertificateExpiredException|PKIX\s+path|SSL.*handshake|NotAfter|validity\s+check\s+failed|certificate\s+has\s+expired)",
        "TLS/X.509 certificate expired or invalid",
    ),
    (
        "upstream_rate_limited",
        r"(429\s+Too\s+Many\s+Requests|Retry-After|rate.limit|exhausted.*retries|circuit.breaker.*open|backoff.*exceeded)",
        "Upstream service returning 429 or circuit breaker open",
    ),
    (
        "database_query_timeout",
        r"(statement\s+timeout|query\s+exceeded.*execution|canceling\s+statement|SQLException.*timeout|max\s+connections.*reached|lock\s+wait\s+timeout)",
        "Database query exceeded timeout threshold",
    ),
    (
        "redis_oom_eviction",
        r"(maxmemory.*limit\s+reached|OOM\s+command\s+not\s+allowed|used\s+memory.*maxmemory|eviction.*volatile|MISCONF.*maxmemory)",
        "Redis maxmemory policy triggering evictions",
    ),
    (
        "pod_crash_loop",
        r"(CrashLoopBackOff|Back-off\s+restarting|container.*OOMKilled|Liveness\s+probe\s+failed|pod.*restarting.*\d+\s+times)",
        "Kubernetes pod in CrashLoopBackOff",
    ),
    (
        "http_upstream_unavailable",
        r"(503\s+Service\s+Unavailable|upstream\s+connect\s+error|connection\s+refused|no\s+healthy\s+upstream|ECONNREFUSED)",
        "Upstream HTTP service unreachable",
    ),
    (
        "transaction_leak",
        r"(Long-running\s+transaction|connection\s+leak|open\s+for\s+\d+\s+seconds|possible\s+connection\s+leak|idle\s+in\s+transaction)",
        "Database connection held open indefinitely (leak)",
    ),
    (
        "thread_pool_exhausted",
        r"(Thread\s+pool\s+exhausted|Request\s+queue.*maximum|Rejecting\s+new\s+incoming|thread.*blocked|ExecutorService.*rejected|threadpool.*full)",
        "Application thread pool saturated",
    ),
    (
        "kafka_consumer_lag",
        r"(consumer\s+group.*lag|kafka.*offset.*behind|messages\s+pending.*\d{4,}|consumer.*not\s+keeping\s+up)",
        "Kafka consumer group falling behind on partition offsets",
    ),
    (
        "grpc_deadline_exceeded",
        r"(DEADLINE_EXCEEDED|rpc\s+error.*DeadlineExceeded|context\s+deadline\s+exceeded|grpc.*timeout)",
        "gRPC call exceeded deadline",
    ),
    (
        "network_partition",
        r"(network\s+partition|split.brain|unreachable\s+node|raft.*leader\s+lost|etcd.*cluster\s+unavailable)",
        "Network partition or leader election failure",
    ),
]


# ---------------------------------------------------------------------------
# Transient template classification
# ---------------------------------------------------------------------------
# Keys in this set are known to produce single-occurrence error logs during
# normal service operation (e.g., a retry that succeeded, a DNS blip, a
# brief 503 during rolling restart).  The router's TransientFilter suppresses
# these when occurrence count < min_recurrence AND no paired success line is
# found in the same telemetry block.
#
# NOT included here:
#   - database_query_timeout  →  Threshold-Contextual (dual-nature)
#   - connection_pool_exhausted, oom_killed, disk_space_exhausted, etc.
#     →  Persistent / Fatal (single occurrence is sufficient signal)
TRANSIENT_TEMPLATE_KEYS: frozenset[str] = frozenset({
    "upstream_rate_limited",      # 429 / circuit-breaker open — retry window, not an incident
    "grpc_deadline_exceeded",     # Single RPC timeout — often a network hiccup
    "dns_resolution_failure",     # Temporary DNS server unavailability
    "http_upstream_unavailable",  # Single 503 during rolling restart is normal
})


@dataclass
class L1Match:
    """Result of a successful L1 symbolic cache lookup."""
    template_key: str
    pattern: str
    description: str
    source: str          # "builtin" | "l3_learned" | "user"
    confidence: float = 1.0
    tier: str = "L1"
    is_transient: bool = False   # True when template_key is in TRANSIENT_TEMPLATE_KEYS


@dataclass
class _TemplateEntry:
    """Internal registry record."""
    key: str
    compiled: re.Pattern
    pattern_str: str
    description: str
    source: str
    hit_count: int = 0
    is_transient: bool = False   # Populated from TRANSIENT_TEMPLATE_KEYS at registration


class L1SymbolicCache:
    """
    Deterministic regex cache for known log failure patterns.

    Thread-safe for reads.  ``register_template`` acquires no lock —
    callers that run the L3 learning loop concurrently should ensure
    they call it from the same async task or add their own guard.
    """

    def __init__(self) -> None:
        # Ordered dict preserves insertion order; we re-sort by hit_count lazily.
        self._registry: OrderedDict[str, _TemplateEntry] = OrderedDict()
        self._total_queries: int = 0
        self._total_hits: int = 0
        self._learned_count: int = 0

        for key, pattern, desc in BUILTIN_TEMPLATES:
            self._add(key, pattern, desc, source="builtin")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, log_entry: str) -> Optional[L1Match]:
        """
        Attempt to classify *log_entry* against the known template registry.

        Returns an ``L1Match`` on success, ``None`` when L2 should be tried.
        Templates are iterated highest-hit-count first for cache efficiency.
        """
        self._total_queries += 1
        for entry in self._sorted_entries():
            if entry.compiled.search(log_entry):
                entry.hit_count += 1
                self._total_hits += 1
                return L1Match(
                    template_key=entry.key,
                    pattern=entry.pattern_str,
                    description=entry.description,
                    source=entry.source,
                    confidence=1.0,
                    tier="L1",
                    is_transient=entry.is_transient,
                )
        return None

    def register_template(
        self,
        key: str,
        pattern: str,
        description: str = "",
        source: str = "l3_learned",
    ) -> bool:
        """
        Register a new or updated regex template.

        Called at initialisation for built-ins, and by the L3 learning loop
        for novel patterns.

        Returns:
            True if registration succeeded, False on invalid regex.
        """
        try:
            self._add(key, pattern, description, source)
            logger.debug("[L1Cache] Registered template: key=%s source=%s", key, source)
            if source == "l3_learned":
                self._learned_count += 1
            return True
        except re.error as exc:
            logger.warning("[L1Cache] Invalid regex for '%s': %s", key, exc)
            return False

    def snapshot(self) -> dict:
        """Return a serialisable snapshot of the current registry for test isolation."""
        return {
            k: (e.pattern_str, e.description, e.source, e.hit_count)
            for k, e in self._registry.items()
        }

    def restore(self, snap: dict) -> None:
        """Restore the registry from a previous snapshot."""
        self._registry.clear()
        self._total_hits = 0
        self._total_queries = 0
        self._learned_count = 0
        for k, (pattern, desc, src, hits) in snap.items():
            self._add(k, pattern, desc, src)
            self._registry[k].hit_count = hits

    @property
    def stats(self) -> dict:
        """Observability stats consumed by the PerceptionReport."""
        return {
            "total_queries": self._total_queries,
            "total_hits": self._total_hits,
            "hit_rate": round(self._total_hits / max(1, self._total_queries), 4),
            "registered_templates": len(self._registry),
            "learned_templates": self._learned_count,
            "top_patterns": [
                {"key": e.key, "hits": e.hit_count}
                for e in sorted(
                    self._registry.values(), key=lambda x: x.hit_count, reverse=True
                )[:5]
            ],
        }

    @property
    def template_keys(self) -> list[str]:
        return list(self._registry.keys())

    @property
    def builtin_keys(self) -> list[str]:
        return [k for k, e in self._registry.items() if e.source == "builtin"]

    @property
    def learned_keys(self) -> list[str]:
        return [k for k, e in self._registry.items() if e.source == "l3_learned"]

    def is_transient_key(self, key: str) -> bool:
        """
        Return True if *key* is registered in ``TRANSIENT_TEMPLATE_KEYS``.

        Can be called by the router's TransientFilter and SymbolicValidator
        without holding a reference to the TRANSIENT_TEMPLATE_KEYS constant
        directly, making it easy to extend via ``register_template`` in future.
        """
        entry = self._registry.get(key)
        return entry.is_transient if entry is not None else False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add(self, key: str, pattern: str, description: str, source: str) -> None:
        compiled = re.compile(pattern, re.IGNORECASE)
        self._registry[key] = _TemplateEntry(
            key=key,
            compiled=compiled,
            pattern_str=pattern,
            description=description,
            source=source,
            hit_count=self._registry[key].hit_count if key in self._registry else 0,
            is_transient=key in TRANSIENT_TEMPLATE_KEYS,
        )

    def _sorted_entries(self) -> list[_TemplateEntry]:
        """Return entries sorted by hit_count descending (hot-path first)."""
        return sorted(self._registry.values(), key=lambda e: e.hit_count, reverse=True)
