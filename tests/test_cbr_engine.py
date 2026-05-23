"""
tests/test_cbr_engine.py

Unit tests for the Case-Based Reasoning (CBR) engine.

Tests cover:
  - Feature vector extraction from raw telemetry
  - _cosine_similarity (module-level helper in incident_store)
  - IncidentStore: seed cases, search_similar, store_case, deduplication
  - CBREngine.retrieve: top-k, min_similarity filter, sort order
  - CBREngine.reuse: service/namespace substitution, step count preserved
  - CBREngine.retain: store delegation, dedup via store_case
  - CBREngine.format_candidates_markdown: empty vs populated
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime

import pytest

from agent.reasoning.incident_store import (
    HistoricalCase,
    IncidentStore,
    _cosine_similarity,
    VECTOR_DIM,
    _SEED_CASES,
)
from agent.reasoning.cbr_engine import CBREngine, ScoredCase, AdaptedPlan


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_case(
    incident_id: str,
    service: str = "payments-service",
    category: str = "connection_pool_exhaustion",
    vector: list[float] | None = None,
    mttr: int = 12,
) -> HistoricalCase:
    """Create a HistoricalCase with sensible defaults for testing."""
    if vector is None:
        vector = [0.0] * VECTOR_DIM
    return HistoricalCase(
        incident_id=incident_id,
        service=service,
        severity="P1",
        root_cause_category=category,
        symptom_vector=vector,
        telemetry_fingerprint=f"fp-{incident_id}",
        remediation_steps=[
            {
                "order": 1,
                "action": "Rolling restart",
                "command": f"kubectl rollout restart deployment/{service} -n prod",
                "risk": "low",
            },
            {
                "order": 2,
                "action": "Scale replicas",
                "command": f"kubectl scale deployment/{service} --replicas=5 -n prod",
                "risk": "medium",
            },
        ],
        rollback_command=f"kubectl rollout undo deployment/{service} -n prod",
        outcome="resolved",
        mttr_minutes=mttr,
        resolved_at=datetime.utcnow(),
        postmortem_summary="Connection pool exhausted due to long-running transactions.",
    )


@pytest.fixture(autouse=True)
def reset_store_singleton():
    """Reset IncidentStore singleton before each test for isolation."""
    IncidentStore._instance = None
    yield
    IncidentStore._instance = None


@pytest.fixture
def store() -> IncidentStore:
    """Fresh IncidentStore pre-loaded with 3 synthetic cases (no seed cases)."""
    s = IncidentStore()
    s._memory_store = [
        make_case("INC-001", vector=[1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.33]),
        make_case("INC-002", vector=[0.9, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.33]),
        make_case(
            "INC-003",
            service="order-service",
            category="oom_killed",
            vector=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.67],
        ),
    ]
    s._initialized = True
    return s


@pytest.fixture
def engine(store) -> CBREngine:
    """CBREngine wired to the test IncidentStore."""
    IncidentStore._instance = store
    return CBREngine()


# ---------------------------------------------------------------------------
# Tests: _cosine_similarity (module-level)
# ---------------------------------------------------------------------------

def test_cosine_similarity_identical():
    """Identical vectors → similarity == 1.0."""
    vec = [1.0, 0.5, 0.3] + [0.0] * 8
    sim = _cosine_similarity(vec, vec)
    assert abs(sim - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    """Completely orthogonal vectors → similarity == 0.0."""
    v1 = [1.0] + [0.0] * 10
    v2 = [0.0, 1.0] + [0.0] * 9
    sim = _cosine_similarity(v1, v2)
    assert abs(sim) < 1e-6


def test_cosine_similarity_zero_vector():
    """Zero query vector → 0.0 (no NaN or divide-by-zero)."""
    v_zero = [0.0] * VECTOR_DIM
    v_case = [1.0] + [0.0] * (VECTOR_DIM - 1)
    sim = _cosine_similarity(v_zero, v_case)
    assert sim == 0.0


def test_cosine_similarity_length_mismatch():
    """Mismatched lengths → 0.0 (safe fallback)."""
    sim = _cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])
    assert sim == 0.0


def test_cosine_similarity_partial_overlap():
    """Partial overlap produces value between 0 and 1."""
    v1 = [1.0, 1.0] + [0.0] * 9
    v2 = [1.0, 0.0] + [0.0] * 9
    sim = _cosine_similarity(v1, v2)
    assert 0.0 < sim < 1.0


# ---------------------------------------------------------------------------
# Tests: IncidentStore seed data
# ---------------------------------------------------------------------------

def test_seed_cases_exist():
    """IncidentStore is pre-seeded with production cases."""
    assert len(_SEED_CASES) >= 3


def test_incident_store_memory_seeded():
    """Fresh IncidentStore._memory_store contains seed cases."""
    s = IncidentStore()
    assert s.total_cases >= len(_SEED_CASES)


@pytest.mark.asyncio
async def test_store_case_adds_entry(store):
    """store_case() adds a new case to the in-memory store."""
    initial = store.total_cases
    new_case = make_case("INC-NEW-001")
    await store.store_case(new_case)
    assert store.total_cases == initial + 1


@pytest.mark.asyncio
async def test_store_case_deduplicates(store):
    """store_case() does not add a duplicate incident_id."""
    initial = store.total_cases
    duplicate = make_case("INC-001")  # Already in store fixture
    await store.store_case(duplicate)
    assert store.total_cases == initial  # Count unchanged


@pytest.mark.asyncio
async def test_search_similar_returns_ranked(store):
    """search_similar() returns cases sorted by descending similarity."""
    query = [1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.33]
    results = await store.search_similar(query, top_k=3, min_similarity=0.0)
    sims = [c.similarity_score for c in results]
    assert sims == sorted(sims, reverse=True), "Results should be sorted by descending similarity"


@pytest.mark.asyncio
async def test_search_similar_min_similarity_filter(store):
    """Cases below min_similarity are excluded from results."""
    # OOM case (INC-003) is orthogonal to a connection-pool query
    query = [1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.33]
    results = await store.search_similar(query, top_k=5, min_similarity=0.9)
    ids = {c.incident_id for c in results}
    assert "INC-003" not in ids, "Dissimilar OOM case should be filtered out"


@pytest.mark.asyncio
async def test_search_similar_top_k(store):
    """search_similar() returns at most top_k results."""
    query = [0.5] * VECTOR_DIM
    results = await store.search_similar(query, top_k=2, min_similarity=0.0)
    assert len(results) <= 2


@pytest.mark.asyncio
async def test_search_similar_populates_similarity_score(store):
    """Returned cases have similarity_score populated."""
    query = [1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.33]
    results = await store.search_similar(query, top_k=3, min_similarity=0.0)
    for case in results:
        assert 0.0 <= case.similarity_score <= 1.0


@pytest.mark.asyncio
async def test_get_by_service(store):
    """get_by_service() filters by service name case-insensitively."""
    results = await store.get_by_service("payments-service")
    assert all(c.service.lower() == "payments-service" for c in results)
    assert len(results) >= 2  # INC-001 and INC-002


@pytest.mark.asyncio
async def test_get_by_category(store):
    """get_by_category() filters by root_cause_category."""
    results = await store.get_by_category("oom_killed")
    assert all(c.root_cause_category == "oom_killed" for c in results)
    assert len(results) >= 1  # INC-003


# ---------------------------------------------------------------------------
# Tests: CBREngine.extract_feature_vector
# ---------------------------------------------------------------------------

def test_extract_feature_vector_length(engine):
    """Extracted vector has exactly VECTOR_DIM dimensions."""
    vec = engine.extract_feature_vector("connection pool exhausted timeout", "svc", service_tier=1)
    assert len(vec) == VECTOR_DIM


def test_extract_feature_vector_connection_pool_signal(engine):
    """Connection pool telemetry activates timeout signal (dim 9)."""
    telemetry = "QueuePool limit reached, pool exhausted, connection timed out"
    vec = engine.extract_feature_vector(telemetry, "payments-service", service_tier=1)
    assert vec[9] > 0.0, "Timeout signal (dim 9) should be active for pool exhaustion"


def test_extract_feature_vector_oom_signal(engine):
    """OOM telemetry activates the OOM binary signal (dim 8)."""
    telemetry = "OOMKilled — container exceeded memory limit (Exit code 137)"
    vec = engine.extract_feature_vector(telemetry, "auth-service", service_tier=1)
    assert vec[8] == 1.0, "OOM binary signal (dim 8) should be 1.0"


def test_extract_feature_vector_error_rate_numeric(engine):
    """Error rate in telemetry is extracted to dim 0."""
    telemetry = "error rate 47.2% exceeds threshold"
    vec = engine.extract_feature_vector(telemetry, "svc", service_tier=2)
    assert vec[0] > 0.0, "Error rate (dim 0) should be > 0"
    assert vec[0] <= 1.0, "Normalized error rate should be <= 1.0"


def test_extract_feature_vector_tier_encoding(engine):
    """Service tier is encoded in the last dimension (dim 10)."""
    vec_t1 = engine.extract_feature_vector("", "svc", service_tier=1)
    vec_t3 = engine.extract_feature_vector("", "svc", service_tier=3)
    assert vec_t1[10] < vec_t3[10], "Higher tier number → higher dim-10 value"


def test_extract_feature_vector_all_bounded(engine):
    """All vector values are in [0.0, 1.0] range."""
    telemetry = "error rate 99% connection pool 100% cpu 95% OOMKilled timeout"
    vec = engine.extract_feature_vector(telemetry, "svc", service_tier=1)
    for i, v in enumerate(vec):
        assert 0.0 <= v <= 1.0, f"Dim {i} out of bounds: {v}"


# ---------------------------------------------------------------------------
# Tests: CBREngine.retrieve
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrieve_returns_scored_cases(engine):
    """retrieve() returns a list of ScoredCase objects."""
    query = [1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.33]
    results = await engine.retrieve(query, top_k=3, min_similarity=0.0)
    assert all(isinstance(r, ScoredCase) for r in results)


@pytest.mark.asyncio
async def test_retrieve_top_k(engine):
    """retrieve() respects top_k limit."""
    query = [1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.33]
    results = await engine.retrieve(query, top_k=2, min_similarity=0.0)
    assert len(results) <= 2


@pytest.mark.asyncio
async def test_retrieve_sorted_by_similarity(engine):
    """retrieve() returns results sorted by descending similarity."""
    query = [1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.33]
    results = await engine.retrieve(query, top_k=3, min_similarity=0.0)
    sims = [r.similarity for r in results]
    assert sims == sorted(sims, reverse=True)


@pytest.mark.asyncio
async def test_retrieve_min_similarity_excludes_low_matches(engine):
    """INC-003 (OOM vector) should be excluded when querying with high min_similarity."""
    query = [1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.33]
    results = await engine.retrieve(query, top_k=5, min_similarity=0.95)
    ids = {r.case.incident_id for r in results}
    assert "INC-003" not in ids


@pytest.mark.asyncio
async def test_retrieve_empty_store(engine):
    """retrieve() on empty store returns empty list."""
    engine._store._memory_store = []
    results = await engine.retrieve([0.5] * VECTOR_DIM, top_k=3)
    assert results == []


@pytest.mark.asyncio
async def test_retrieve_scored_case_has_rank(engine):
    """ScoredCase.rank field is populated correctly (1-indexed)."""
    query = [1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.33]
    results = await engine.retrieve(query, top_k=3, min_similarity=0.0)
    if results:
        assert results[0].rank == 1
        if len(results) > 1:
            assert results[1].rank == 2


# ---------------------------------------------------------------------------
# Tests: CBREngine.reuse
# ---------------------------------------------------------------------------

def test_reuse_returns_adapted_plan(engine):
    """reuse() returns an AdaptedPlan object."""
    case = make_case("INC-001", service="payments-service")
    plan = engine.reuse("auth-service", "staging", case)
    assert isinstance(plan, AdaptedPlan)


def test_reuse_adapts_service_name(engine):
    """reuse() substitutes source service name with current service in commands."""
    case = make_case("INC-001", service="payments-service")
    plan = engine.reuse("auth-service", "staging", case)
    for step in plan.adapted_steps:
        cmd = step.get("command") or ""
        if "payments-service" in cmd:
            pytest.fail(f"Source service name not adapted: {cmd}")


def test_reuse_adapts_namespace(engine):
    """reuse() substitutes -n prod with the current namespace."""
    case = make_case("INC-001", service="payments-service")
    plan = engine.reuse("payments-service", "staging", case)
    for step in plan.adapted_steps:
        cmd = step.get("command") or ""
        assert "-n prod" not in cmd, f"Namespace not adapted in: {cmd}"
        if "kubectl" in cmd:
            assert "-n staging" in cmd


def test_reuse_preserves_step_count(engine):
    """reuse() preserves the exact number of original remediation steps."""
    case = make_case("INC-001")
    plan = engine.reuse("new-service", "prod", case)
    assert len(plan.adapted_steps) == len(case.remediation_steps)


def test_reuse_sets_source_case_id(engine):
    """AdaptedPlan.source_case_id matches the original case's incident_id."""
    case = make_case("INC-042")
    plan = engine.reuse("new-service", "prod", case)
    assert plan.source_case_id == "INC-042"


def test_reuse_confidence_is_bounded(engine):
    """AdaptedPlan.confidence is in [0.0, 1.0]."""
    case = make_case("INC-001")
    case.similarity_score = 0.92
    plan = engine.reuse("svc", "prod", case)
    assert 0.0 <= plan.confidence <= 1.0


def test_reuse_to_markdown(engine):
    """AdaptedPlan.to_markdown() returns a non-empty markdown string."""
    case = make_case("INC-001")
    case.similarity_score = 0.85
    plan = engine.reuse("svc", "prod", case)
    md = plan.to_markdown()
    assert isinstance(md, str)
    assert "INC-001" in md
    assert len(md) > 50


# ---------------------------------------------------------------------------
# Tests: CBREngine.retain
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retain_adds_case(engine, store):
    """retain() adds a new case to the underlying store."""
    initial = store.total_cases
    new_case = make_case("INC-RETAIN-001")
    await engine.retain(new_case)
    assert store.total_cases == initial + 1


@pytest.mark.asyncio
async def test_retain_deduplicates(engine, store):
    """retain() does not add a case whose incident_id already exists."""
    initial = store.total_cases
    duplicate = make_case("INC-001")  # Already in the store fixture
    await engine.retain(duplicate)
    assert store.total_cases == initial


# ---------------------------------------------------------------------------
# Tests: format_candidates_markdown
# ---------------------------------------------------------------------------

def test_format_candidates_markdown_empty(engine):
    """Empty candidates list returns a 'no matches' string."""
    md = engine.format_candidates_markdown([])
    assert isinstance(md, str)
    assert "no" in md.lower() or "0" in md or "none" in md.lower()


def test_format_candidates_markdown_populated(engine):
    """Populated candidates include incident ID in the output."""
    case = make_case("INC-001")
    case.similarity_score = 0.95
    candidates = [ScoredCase(case=case, similarity=0.95, rank=1)]
    md = engine.format_candidates_markdown(candidates)
    assert "INC-001" in md
    assert isinstance(md, str)
    assert len(md) > 20
