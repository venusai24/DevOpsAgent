"""
agent/reasoning/incident_store.py

PostgreSQL-backed historical incident database for the Case-Based Reasoning (CBR) engine.

Each resolved incident is stored as a HistoricalCase with:
  - A normalized feature vector for cosine-similarity matching
  - The complete remediation plan that resolved it
  - Outcome metrics (MTTR, resolution status)

The store is seeded with the 7 incident scenarios from mock_enterprise/fixtures.json
so the CBR engine has immediate historical context on first run.

Falls back to an in-memory list when DATABASE_URL is not configured (demo mode).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

# Attempt Postgres import — falls back gracefully for demo mode
try:
    import psycopg
    HAS_PSYCOPG = True
except ImportError:
    psycopg = None  # type: ignore
    HAS_PSYCOPG = False
    logger.warning("[incident_store] psycopg not available — using in-memory store (demo mode).")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class HistoricalCase:
    """A fully resolved incident that serves as a reusable CBR case."""
    incident_id: str
    service: str
    severity: str
    root_cause_category: str          # e.g., "connection_pool_exhaustion"
    symptom_vector: list[float]        # Normalized feature vector (11 dimensions)
    telemetry_fingerprint: str         # SHA-256 hash of key symptoms for fast dedup
    remediation_steps: list[dict]      # Ordered list of RemediationStep dicts
    rollback_command: str
    outcome: str                       # "resolved" | "escalated" | "rolled_back"
    mttr_minutes: int
    resolved_at: datetime
    postmortem_summary: str
    cbr_confidence: float = 1.0        # Confidence score when retrieved as CBR match
    similarity_score: float = 0.0      # Populated during retrieval

    def to_dict(self) -> dict:
        """Serialize to a JSON-serializable dict for storage and transport."""
        return {
            "incident_id": self.incident_id,
            "service": self.service,
            "severity": self.severity,
            "root_cause_category": self.root_cause_category,
            "symptom_vector": self.symptom_vector,
            "telemetry_fingerprint": self.telemetry_fingerprint,
            "remediation_steps": self.remediation_steps,
            "rollback_command": self.rollback_command,
            "outcome": self.outcome,
            "mttr_minutes": self.mttr_minutes,
            "resolved_at": self.resolved_at.isoformat(),
            "postmortem_summary": self.postmortem_summary,
        }

    def summary_markdown(self) -> str:
        """One-paragraph markdown summary for LLM context injection."""
        steps_preview = "; ".join(
            s.get("action", "")[:60] for s in self.remediation_steps[:3]
        )
        return (
            f"**Case {self.incident_id}** | Service: `{self.service}` | "
            f"Severity: {self.severity} | Category: `{self.root_cause_category}` | "
            f"Similarity: {self.similarity_score:.0%} | MTTR: {self.mttr_minutes}m | "
            f"Outcome: {self.outcome}\n"
            f"  Steps: {steps_preview}\n"
            f"  Rollback: `{self.rollback_command[:80]}`"
        )


# ---------------------------------------------------------------------------
# Feature vector dimensions (must stay in sync with cbr_engine.py)
# ---------------------------------------------------------------------------
# Index 0:  error_rate_pct (0–100)
# Index 1:  connection_pool_pct (0–100)
# Index 2:  cpu_utilization_pct (0–100)
# Index 3:  memory_utilization_pct (0–100)
# Index 4:  dns_latency_ms (0–10000, clipped)
# Index 5:  disk_utilization_pct (0–100)
# Index 6:  ssl_error_count (0–1000, clipped)
# Index 7:  upstream_429_count (0–1000, clipped)
# Index 8:  has_oom_signal (0 or 1)
# Index 9:  has_timeout_signal (0 or 1)
# Index 10: service_tier (1, 2, or 3)
VECTOR_DIM = 11


# ---------------------------------------------------------------------------
# Seeded historical cases derived from fixtures.json scenarios
# ---------------------------------------------------------------------------

_SEED_CASES: list[HistoricalCase] = [
    HistoricalCase(
        incident_id="PD-20240519-0042",
        service="payments-service",
        severity="P0",
        root_cause_category="connection_pool_exhaustion",
        symptom_vector=[47.2, 100.0, 33.9, 20.0, 0.0, 30.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        telemetry_fingerprint=hashlib.sha256(b"connection_pool_exhaustion:payments-service").hexdigest()[:16],
        remediation_steps=[
            {"order": 1, "action": "Kill long-running leaked transaction",
             "command": "kubectl exec -n prod deployment/payments-service -- psql -c \"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE query_start < now() - interval '5 minutes';\"",
             "risk": "medium"},
            {"order": 2, "action": "Rolling restart to drain connection pool",
             "command": "kubectl rollout restart deployment/payments-service -n prod",
             "risk": "medium"},
            {"order": 3, "action": "Roll back to previous stable image",
             "command": "kubectl rollout undo deployment/payments-service -n prod",
             "risk": "low"},
        ],
        rollback_command="kubectl rollout undo deployment/payments-service -n prod",
        outcome="resolved",
        mttr_minutes=12,
        resolved_at=datetime(2024, 5, 19, 14, 35, 0),
        postmortem_summary="Connection leak in payment handler exhausted DB pool. Fixed by terminating stale TX and rolling back deployment.",
    ),
    HistoricalCase(
        incident_id="PD-20240520-1045",
        service="auth-service",
        severity="P1",
        root_cause_category="oom_killed",
        symptom_vector=[15.0, 5.0, 32.0, 99.0, 0.0, 20.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        telemetry_fingerprint=hashlib.sha256(b"oom_killed:auth-service").hexdigest()[:16],
        remediation_steps=[
            {"order": 1, "action": "Increase memory limit for auth-service pods",
             "command": "kubectl set resources deployment/auth-service -n prod --limits=memory=2Gi",
             "risk": "low"},
            {"order": 2, "action": "Rolling restart to apply new memory limits",
             "command": "kubectl rollout restart deployment/auth-service -n prod",
             "risk": "low"},
        ],
        rollback_command="kubectl rollout undo deployment/auth-service -n prod",
        outcome="resolved",
        mttr_minutes=8,
        resolved_at=datetime(2024, 5, 20, 11, 0, 0),
        postmortem_summary="Java heap exhaustion caused OOMKill. Increased memory limit to 2Gi and restarted.",
    ),
    HistoricalCase(
        incident_id="PD-20240520-2055",
        service="inventory-service",
        severity="P1",
        root_cause_category="dns_resolution_failure",
        symptom_vector=[85.0, 5.0, 20.0, 30.0, 5005.0, 20.0, 0.0, 0.0, 0.0, 1.0, 2.0],
        telemetry_fingerprint=hashlib.sha256(b"dns_failure:inventory-service").hexdigest()[:16],
        remediation_steps=[
            {"order": 1, "action": "Restart CoreDNS pods to clear rate-limiting state",
             "command": "kubectl rollout restart deployment/coredns -n kube-system",
             "risk": "low"},
            {"order": 2, "action": "Increase CoreDNS cache TTL to reduce resolution frequency",
             "command": "kubectl edit configmap coredns -n kube-system",
             "risk": "medium"},
        ],
        rollback_command="kubectl rollout undo deployment/coredns -n kube-system",
        outcome="resolved",
        mttr_minutes=15,
        resolved_at=datetime(2024, 5, 20, 21, 15, 0),
        postmortem_summary="CoreDNS rate limiting caused name resolution failures for external API. Fixed by restarting CoreDNS.",
    ),
    HistoricalCase(
        incident_id="PD-20260520-1001",
        service="user-profile-service",
        severity="P1",
        root_cause_category="redis_oom_eviction",
        symptom_vector=[30.0, 5.0, 25.0, 100.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        telemetry_fingerprint=hashlib.sha256(b"redis_oom:user-profile-service").hexdigest()[:16],
        remediation_steps=[
            {"order": 1, "action": "Flush expired Redis keys to free memory",
             "command": "kubectl exec -n prod redis-cache-statefulset-0 -- redis-cli FLUSHDB ASYNC",
             "risk": "medium"},
            {"order": 2, "action": "Increase Redis maxmemory limit",
             "command": "kubectl exec -n prod redis-cache-statefulset-0 -- redis-cli CONFIG SET maxmemory 4gb",
             "risk": "low"},
        ],
        rollback_command="kubectl exec -n prod redis-cache-statefulset-0 -- redis-cli CONFIG SET maxmemory 2gb",
        outcome="resolved",
        mttr_minutes=10,
        resolved_at=datetime(2026, 5, 20, 19, 30, 0),
        postmortem_summary="Redis OOM eviction caused cache misses which propagated to 500 errors. Flushed expired keys and increased limit.",
    ),
    HistoricalCase(
        incident_id="PD-20260520-1002",
        service="order-processing-api",
        severity="P0",
        root_cause_category="disk_space_exhaustion",
        symptom_vector=[40.0, 10.0, 45.0, 60.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        telemetry_fingerprint=hashlib.sha256(b"disk_full:order-processing-api").hexdigest()[:16],
        remediation_steps=[
            {"order": 1, "action": "Rotate and compress old application logs",
             "command": "kubectl exec -n prod order-api-node-abc12-87x2 -- find /var/log/app -name '*.log' -mtime +1 -exec gzip {} \\;",
             "risk": "low"},
            {"order": 2, "action": "Delete compressed logs older than 7 days",
             "command": "kubectl exec -n prod order-api-node-abc12-87x2 -- find /var/log/app -name '*.gz' -mtime +7 -delete",
             "risk": "low"},
        ],
        rollback_command="echo 'Disk cleanup is non-reversible but safe'",
        outcome="resolved",
        mttr_minutes=5,
        resolved_at=datetime(2026, 5, 20, 19, 28, 0),
        postmortem_summary="Log directory filled disk. Compressed and pruned old logs to restore write capacity.",
    ),
]


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length numeric vectors."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# IncidentStore
# ---------------------------------------------------------------------------


class IncidentStore:
    """
    Historical incident database for the CBR engine.

    Storage backends:
      - **PostgreSQL** (production): Persists cases across restarts. Requires
        DATABASE_URL to be configured in the environment.
      - **In-memory list** (demo mode): Used when DATABASE_URL is absent. Pre-seeded
        with cases from mock_enterprise/fixtures.json scenarios.

    The store is a singleton — use ``IncidentStore.get_instance()`` rather than
    constructing directly.
    """

    _instance: Optional["IncidentStore"] = None

    def __init__(self) -> None:
        self._memory_store: list[HistoricalCase] = list(_SEED_CASES)  # seed in-memory
        self._pg_conn: Any = None
        self._initialized = False

    @classmethod
    def get_instance(cls) -> "IncidentStore":
        """Return the singleton IncidentStore instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def initialize(self) -> None:
        """Set up the PostgreSQL schema (if configured) or use in-memory store."""
        if self._initialized:
            return
        if settings.DATABASE_URL and HAS_PSYCOPG:
            await self._init_postgres()
        else:
            logger.info("[IncidentStore] Running in in-memory demo mode (%d seed cases).", len(self._memory_store))
        self._initialized = True

    async def _init_postgres(self) -> None:
        """Create the incidents table in PostgreSQL if it does not exist."""
        try:
            conn = await psycopg.AsyncConnection.connect(settings.DATABASE_URL)
            async with conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS historical_cases (
                        incident_id            TEXT PRIMARY KEY,
                        service                TEXT NOT NULL,
                        severity               TEXT NOT NULL,
                        root_cause_category    TEXT NOT NULL,
                        symptom_vector         JSONB NOT NULL,
                        telemetry_fingerprint  TEXT NOT NULL,
                        remediation_steps      JSONB NOT NULL,
                        rollback_command       TEXT NOT NULL,
                        outcome                TEXT NOT NULL,
                        mttr_minutes           INTEGER NOT NULL,
                        resolved_at            TIMESTAMPTZ NOT NULL,
                        postmortem_summary     TEXT NOT NULL,
                        created_at             TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                await conn.commit()
            # Seed pre-existing cases
            for case in _SEED_CASES:
                await self._upsert_to_postgres(case)
            logger.info("[IncidentStore] PostgreSQL initialized with %d seed cases.", len(_SEED_CASES))
        except Exception as exc:
            logger.warning("[IncidentStore] PostgreSQL init failed (%s) — falling back to in-memory.", exc)

    async def _upsert_to_postgres(self, case: HistoricalCase) -> None:
        """Insert or update a case in PostgreSQL."""
        if not (settings.DATABASE_URL and HAS_PSYCOPG):
            return
        try:
            conn = await psycopg.AsyncConnection.connect(settings.DATABASE_URL)
            async with conn:
                await conn.execute("""
                    INSERT INTO historical_cases
                        (incident_id, service, severity, root_cause_category, symptom_vector,
                         telemetry_fingerprint, remediation_steps, rollback_command, outcome,
                         mttr_minutes, resolved_at, postmortem_summary)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (incident_id) DO NOTHING
                """, (
                    case.incident_id, case.service, case.severity, case.root_cause_category,
                    json.dumps(case.symptom_vector), case.telemetry_fingerprint,
                    json.dumps(case.remediation_steps), case.rollback_command,
                    case.outcome, case.mttr_minutes, case.resolved_at, case.postmortem_summary,
                ))
                await conn.commit()
        except Exception as exc:
            logger.warning("[IncidentStore] PostgreSQL upsert failed: %s", exc)

    async def store_case(self, case: HistoricalCase) -> None:
        """Persist a newly resolved case. Writes to PostgreSQL if available, plus in-memory."""
        # Deduplicate by incident_id
        existing_ids = {c.incident_id for c in self._memory_store}
        if case.incident_id not in existing_ids:
            self._memory_store.append(case)
        await self._upsert_to_postgres(case)
        logger.info("[IncidentStore] Stored case %s (category=%s)", case.incident_id, case.root_cause_category)

    async def search_similar(
        self,
        query_vector: list[float],
        top_k: int = 5,
        min_similarity: float = 0.5,
    ) -> list[HistoricalCase]:
        """
        Retrieve the most similar historical cases by cosine similarity.

        Args:
            query_vector: The normalized feature vector for the current incident.
            top_k: Maximum number of cases to return.
            min_similarity: Minimum cosine similarity threshold (0.0–1.0).

        Returns:
            Ranked list of HistoricalCase instances with similarity_score populated.
        """
        await self.initialize()
        scored: list[tuple[float, HistoricalCase]] = []
        for case in self._memory_store:
            sim = _cosine_similarity(query_vector, case.symptom_vector)
            if sim >= min_similarity:
                c = HistoricalCase(**{**case.__dict__, "similarity_score": sim})
                scored.append((sim, c))
        scored.sort(key=lambda t: t[0], reverse=True)
        results = [c for _, c in scored[:top_k]]
        logger.info(
            "[IncidentStore] CBR search: %d/%d cases above threshold=%.2f",
            len(results), len(self._memory_store), min_similarity,
        )
        return results

    async def get_by_service(self, service: str) -> list[HistoricalCase]:
        """Return all historical cases for a specific service."""
        await self.initialize()
        return [c for c in self._memory_store if c.service.lower() == service.lower()]

    async def get_by_category(self, category: str) -> list[HistoricalCase]:
        """Return all historical cases matching a root cause category."""
        await self.initialize()
        return [c for c in self._memory_store if c.root_cause_category == category]

    @property
    def total_cases(self) -> int:
        """Total number of cases in the store."""
        return len(self._memory_store)
