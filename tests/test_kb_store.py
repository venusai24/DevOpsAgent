"""
tests/test_kb_store.py

Unit tests for the AIRS KB store (agent/kb/store.py) — Qdrant edition.

Test isolation strategy
-----------------------
1.  ``mock_qdrant`` fixture: patches ``agent.kb.store._get_qdrant_client``
    to return a Qdrant in-memory client (``QdrantClient(":memory:")``) so no
    Docker daemon or network is required.

2.  ``mock_embedder`` fixture: patches ``agent.kb.embedder.embed_text`` and
    ``agent.kb.embedder.embed_for_query`` to return a deterministic fixed
    vector ([0.1] * 384) instead of calling the HF Inference API.  Because
    every insert uses the same vector, every lookup will rank the inserted
    entry as the top-1 result — which is all the tests need.

3.  ``mock_llm_judge`` fixture: patches ``agent.kb.store._llm_judge_similarity``
    to return a fixed score (0.0) so the Qdrant cosine score drives all
    routing decisions without making live Groq calls.

All three fixtures are ``autouse=True`` so every test in this file runs in
full isolation with no external services required.
"""

from __future__ import annotations

import pytest
import asyncio

from qdrant_client import QdrantClient

from agent.state import KBEntry, KBRemediationStep, KBRetrievalResult


# ---------------------------------------------------------------------------
# Entry factory helpers (unchanged from original test file)
# ---------------------------------------------------------------------------

def _make_oom_entry(entry_id: str = "kb-test-001") -> KBEntry:
    return KBEntry(
        entry_id=entry_id,
        incident_taxonomy="Errors:OOM",
        pattern_type="exact",
        error_pattern="Exit Code 137",
        affected_services=["payment-service"],
        severity="P1",
        root_cause_narrative=(
            "One or more pods terminated with exit code 137 (OOMKilled). "
            "The container's memory limit is too low for the current workload."
        ),
        remediation_steps=[
            KBRemediationStep(
                order=1,
                action="Increase memory limit by 50%.",
                environment="kubectl",
                command="kubectl set resources deployment/{service} -n prod --limits=memory=2Gi",
                risk="medium",
            ),
            KBRemediationStep(
                order=2,
                action="Rolling restart.",
                environment="kubectl",
                command="kubectl rollout restart deployment/{service} -n prod",
                risk="low",
            ),
        ],
        rollback_command="kubectl rollout undo deployment/{service} -n prod",
        confidence_score=0.88,
    )


def _make_regex_entry(entry_id: str = "kb-test-002") -> KBEntry:
    return KBEntry(
        entry_id=entry_id,
        incident_taxonomy="Latency:Database",
        pattern_type="regex",
        error_pattern=r"connection.*timed?\s*out",
        affected_services=[],
        severity="P1",
        root_cause_narrative=(
            "Database connection attempts are timing out before a connection "
            "can be established."
        ),
        remediation_steps=[
            KBRemediationStep(
                order=1,
                action="Verify DB pod is running.",
                environment="kubectl",
                command="kubectl get pod -n prod -l app=postgresql",
                risk="low",
            ),
        ],
        rollback_command="kubectl rollout undo deployment/{service} -n prod",
        confidence_score=0.80,
    )


def _make_semantic_entry(entry_id: str = "kb-test-003") -> KBEntry:
    return KBEntry(
        entry_id=entry_id,
        incident_taxonomy="Saturation:ConnectionPool",
        pattern_type="semantic",
        error_pattern="database connection pool exhausted queue full saturation",
        affected_services=[],
        severity="P0",
        root_cause_narrative=(
            "The database connection pool reached its hard limit, causing all "
            "subsequent requests to queue indefinitely."
        ),
        remediation_steps=[
            KBRemediationStep(
                order=1,
                action="Restart service.",
                environment="kubectl",
                command="kubectl rollout restart deployment/{service} -n prod",
                risk="medium",
            ),
        ],
        rollback_command="kubectl rollout undo deployment/{service} -n prod",
        confidence_score=0.75,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_qdrant(monkeypatch):
    """
    Replace QdrantClient with an in-memory Qdrant instance per test.

    Qdrant's Python client supports ``QdrantClient(":memory:")`` for
    integration testing without a running server.  Each test gets a fresh
    in-memory instance because the fixture is autouse and function-scoped.
    """
    mem_client = QdrantClient(":memory:")

    import agent.kb.store as store_module
    monkeypatch.setattr(store_module, "_get_qdrant_client", lambda: mem_client)
    yield mem_client


@pytest.fixture(autouse=True)
def mock_embedder(monkeypatch):
    """
    Replace HF API embedding calls with a deterministic fixed vector.

    Returns [0.1] * 384 for every input so:
      - All entries are inserted with identical vectors (equidistant in space).
      - Any lookup will return the inserted entries as top-K candidates.
      - No HF network call is made.
    """
    fixed_vector = [0.1] * 384

    async def _fake_embed(text: str) -> list[float]:
        return fixed_vector

    import agent.kb.embedder as embedder_module
    monkeypatch.setattr(embedder_module, "embed_text", _fake_embed)
    monkeypatch.setattr(embedder_module, "embed_for_query", _fake_embed)

    # Also patch the references already imported into store.py
    import agent.kb.store as store_module
    monkeypatch.setattr(store_module, "embed_text", _fake_embed)
    monkeypatch.setattr(store_module, "embed_for_query", _fake_embed)

    yield fixed_vector


@pytest.fixture(autouse=True)
def mock_llm_judge(monkeypatch):
    """
    Short-circuit the Groq LLM reranker to return 0.0 for all candidates.

    This means the Qdrant cosine score (which will be 1.0 for an identical
    vector) drives all routing decisions, keeping tests deterministic and
    eliminating live Groq API calls.
    """
    async def _fake_judge(alert_text: str, entry: KBEntry) -> float:
        return 0.0

    import agent.kb.store as store_module
    monkeypatch.setattr(store_module, "_llm_judge_similarity", _fake_judge)
    yield


# ---------------------------------------------------------------------------
# Tests: kb_insert + kb_count
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_insert_single_entry():
    from agent.kb.store import kb_insert, kb_count
    entry = _make_oom_entry()
    assert kb_count() == 0
    await kb_insert(entry)
    assert kb_count() == 1


@pytest.mark.asyncio
async def test_insert_idempotent():
    """Inserting the same entry_id twice should not raise or duplicate (Qdrant upsert)."""
    from agent.kb.store import kb_insert, kb_count
    entry = _make_oom_entry("kb-dup-001")
    await kb_insert(entry)
    await kb_insert(entry)  # second insert — same entry_id → upsert, no duplicate
    assert kb_count() == 1


@pytest.mark.asyncio
async def test_insert_multiple_entries():
    from agent.kb.store import kb_insert, kb_count
    await kb_insert(_make_oom_entry("a"))
    await kb_insert(_make_regex_entry("b"))
    await kb_insert(_make_semantic_entry("c"))
    assert kb_count() == 3


# ---------------------------------------------------------------------------
# Tests: kb_lookup — hit and miss behaviour
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lookup_returns_inserted_entry():
    """
    After inserting an entry, a lookup should return it as the top match.

    Because the mock embedder returns the same vector for all texts, the
    single inserted entry will be the sole top-K candidate and its Qdrant
    cosine score will be 1.0 (identical vectors).
    With confidence_score=0.88, final_score = 1.0 * 0.88 = 0.88 > RAG_THRESHOLD.
    """
    from agent.kb.store import kb_insert, kb_lookup
    entry = _make_oom_entry()
    await kb_insert(entry)

    result: KBRetrievalResult = await kb_lookup(
        alert_title="CRITICAL: Pod OOMKilled — payment-service",
        alert_description="Exit Code 137 detected in last 5 minutes.",
    )

    assert result.entry is not None
    assert result.entry.entry_id == entry.entry_id
    assert result.retrieval_score >= 0.70
    assert result.match_type == "semantic"


@pytest.mark.asyncio
async def test_lookup_empty_collection_returns_none():
    """A lookup against an empty collection returns match_type='none'."""
    from agent.kb.store import kb_lookup
    result: KBRetrievalResult = await kb_lookup(
        alert_title="High latency detected",
        alert_description="p99 latency increased to 2.5s",
    )
    assert result.match_type == "none"
    assert result.entry is None
    assert result.bypass_llm is False


@pytest.mark.asyncio
async def test_lookup_bypass_threshold():
    """
    An entry with confidence_score >= 0.95 should set bypass_llm=True.

    final_score = qdrant_score (1.0) * confidence_score (0.97) = 0.97 >= 0.95.
    """
    from agent.kb.store import kb_insert, kb_lookup

    entry = _make_oom_entry()
    entry = entry.model_copy(update={"confidence_score": 0.97})
    await kb_insert(entry)

    result: KBRetrievalResult = await kb_lookup(
        alert_title="Exit Code 137 OOMKilled payment-service",
        alert_description="",
    )

    assert result.bypass_llm is True


@pytest.mark.asyncio
async def test_lookup_below_rag_threshold():
    """
    An entry with low confidence produces a final_score below KB_RAG_THRESHOLD.

    final_score = 1.0 * 0.50 = 0.50 < 0.70 → match_type='none'.
    """
    from agent.kb.store import kb_insert, kb_lookup

    entry = _make_oom_entry()
    entry = entry.model_copy(update={"confidence_score": 0.50})
    await kb_insert(entry)

    result: KBRetrievalResult = await kb_lookup(
        alert_title="Some random alert",
        alert_description="something happened",
    )

    assert result.match_type == "none"
    assert result.bypass_llm is False


@pytest.mark.asyncio
async def test_lookup_picks_best_among_multiple():
    """
    When multiple entries are inserted, the one with the highest
    final_score (confidence_score) should be returned as the best match.

    All entries share the same vector (mock), so qdrant_score=1.0 for all.
    final_score = 1.0 * confidence_score → highest confidence wins.
    """
    from agent.kb.store import kb_insert, kb_lookup

    low = _make_oom_entry("low-conf")
    low = low.model_copy(update={"confidence_score": 0.72})

    high = _make_regex_entry("high-conf")
    high = high.model_copy(update={"confidence_score": 0.95})

    await kb_insert(low)
    await kb_insert(high)

    result = await kb_lookup("database timeout error", "connection timed out")

    assert result.entry is not None
    assert result.entry.entry_id == "high-conf"


# ---------------------------------------------------------------------------
# Tests: kb_update_confidence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_confidence_success():
    from agent.kb.store import kb_insert, kb_update_confidence, kb_lookup

    entry = _make_oom_entry()
    original_score = entry.confidence_score  # 0.88
    await kb_insert(entry)

    await kb_update_confidence(entry.entry_id, success=True)

    # Re-fetch via lookup to verify payload was updated
    result = await kb_lookup("Exit Code 137", "")
    assert result.entry is not None
    assert result.entry.confidence_score == pytest.approx(
        original_score + 0.01, abs=0.001
    )


@pytest.mark.asyncio
async def test_update_confidence_failure():
    from agent.kb.store import kb_insert, kb_update_confidence, kb_lookup

    entry = _make_oom_entry()
    original_score = entry.confidence_score  # 0.88
    await kb_insert(entry)

    await kb_update_confidence(entry.entry_id, success=False)

    result = await kb_lookup("Exit Code 137", "")
    assert result.entry is not None
    assert result.entry.confidence_score == pytest.approx(
        original_score - 0.02, abs=0.001
    )


@pytest.mark.asyncio
async def test_update_confidence_bounded_at_max():
    from agent.kb.store import kb_insert, kb_update_confidence, kb_lookup

    entry = _make_oom_entry()
    entry = entry.model_copy(update={"confidence_score": 1.0})
    await kb_insert(entry)

    await kb_update_confidence(entry.entry_id, success=True)

    result = await kb_lookup("Exit Code 137", "")
    assert result.entry is not None
    assert result.entry.confidence_score <= 1.0


@pytest.mark.asyncio
async def test_update_confidence_bounded_at_zero():
    """
    Decrementing a 0.01 confidence score should floor at 0.0, not go negative.

    A confidence_score of 0.0 means final_score = qdrant_score * 0.0 = 0.0,
    which is below KB_RAG_THRESHOLD (0.70), so kb_lookup correctly returns
    match_type='none'.  We verify the floor bound by reading the payload
    directly via scroll rather than through kb_lookup.
    """
    from agent.kb.store import kb_insert, kb_update_confidence, _get_qdrant_client, _entry_id_to_point_id
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    entry = _make_oom_entry()
    entry = entry.model_copy(update={"confidence_score": 0.01})
    await kb_insert(entry)

    await kb_update_confidence(entry.entry_id, success=False)

    # Read the stored payload directly — score must be >= 0.0 (not negative)
    cfg_module = __import__("config", fromlist=["settings"])
    client = _get_qdrant_client()
    results, _ = client.scroll(
        collection_name=cfg_module.settings.QDRANT_COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="entry_id", match=MatchValue(value=entry.entry_id))]
        ),
        with_payload=True,
        with_vectors=False,
        limit=1,
    )
    assert results, "Entry should still exist in Qdrant after confidence update"
    stored_score = results[0].payload.get("confidence_score", -1.0)
    assert stored_score >= 0.0, f"confidence_score should be >= 0.0, got {stored_score}"



@pytest.mark.asyncio
async def test_update_confidence_unknown_entry_is_noop():
    """Updating a non-existent entry should not raise — it should log and return."""
    from agent.kb.store import kb_update_confidence
    # Should not raise
    await kb_update_confidence("does-not-exist", success=True)
