"""
tests/test_airs_v2_reasoning.py
================================

Integration test suite for Stage 3 — Reasoning Layer.

Test structure
--------------
TestSymbolicValidatorRules
    Unit tests for each of the five physical rules in isolation.
    Uses direct graph injection (no subprocess).

TestHypothesisEngine
    Unit tests for the HypothesisEngine template→node mapping and
    causal path construction.

TestCausalInvariants
    Parametrized invariant tests applied to EVERY valid hypothesis
    in every scenario:
    - No hallucinated connections
    - Blast radius containment
    - Causal path edge validity

TestSimulatedFailureScenarios
    Five end-to-end scenarios using ReasoningEngine(graph=g) (direct
    graph injection, no MCP subprocess):

    S1  connection_pool_exhausted  → payments-service
        Expected root cause: payments-db
        Must reject: auth-service, thirdparty-sso, inventory-db

    S2  Network bandwidth drop (novel L3 pattern)  → payments-service
        Expected: only topologically reachable nodes in candidates
        Must reject: auth-db, inventory-db, thirdparty-sso

    S3  redis_oom_eviction  → user-profile-service
        Expected root cause: redis-user-cache
        Must reject: auth-db, payments-db, inventory-db

    S4  tls_cert_expired  → auth-service
        Expected root cause: thirdparty-sso
        Must reject: payments-db, inventory-db

    S5  Mixed logs (valid + fabricated impossible candidates) → api-gateway
        Must reject: services ≥ 3 hops away from api-gateway

TestMCPIntegrationReasoning
    End-to-end scenario using ReasoningEngine() (spawns real MCP subprocess)
    to validate the full transport stack + reasoning pipeline.

All tests run in the genai conda environment.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from airs_v2.context.graph import ContextGraph, BLAST_RADIUS_DEPTH
from airs_v2.perception.router import PerceptionReport, RouterResult, Tier
from airs_v2.reasoning import (
    ReasoningEngine,
    IncidentAnalysis,
    CausalGraph,
    RootCauseHypothesis,
    RejectedHypothesis,
    CausalHypothesis,
    CausalEvidence,
    TopologyContext,
    SymbolicValidator,
    HypothesisEngine,
)
from airs_v2.reasoning.symbolic_validator import (
    R1_NODE_EXISTS,
    R2_BLAST_RADIUS,
    R3_PATH_VALID,
    R4_DIRECTION,
    R5_HEALTH_COHERENCE,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_PATH = (
    Path(__file__).resolve().parents[1]
    / "mock_enterprise"
    / "topology_fixtures.json"
)


@pytest.fixture(scope="module")
def graph() -> ContextGraph:
    return ContextGraph(fixtures_path=FIXTURES_PATH)


@pytest.fixture(scope="function")
def isolated_graph() -> ContextGraph:
    return ContextGraph(fixtures_path=FIXTURES_PATH)


def _make_report(
    log_entry: str,
    template_key: str,
    tier: str = "L1",
    confidence: float = 1.0,
    description: str = "",
) -> PerceptionReport:
    """Build a synthetic PerceptionReport from a single log event."""
    tier_enum = Tier(tier)
    r = RouterResult(
        log_entry=log_entry,
        tier=tier_enum,
        template_key=template_key,
        confidence=confidence,
        description=description,
        is_novel=(tier == "L3"),
    )
    return PerceptionReport(
        results=[r],
        primary_template=template_key,
        l1_hits=1 if tier == "L1" else 0,
        l2_hits=1 if tier == "L2" else 0,
        l3_hits=1 if tier == "L3" else 0,
    )


def _make_context(
    graph: ContextGraph,
    focal: str,
    health_overrides: Optional[dict[str, str]] = None,
) -> TopologyContext:
    """Build a TopologyContext for a focal service."""
    br = graph.blast_radius(focal)
    blast_set = set(br.affected_services)

    health_by_node: dict[str, str] = {}
    metrics_by_node: dict[str, float] = {}
    on_call_by_node: dict[str, list[str]] = {}

    for name in blast_set | {focal}:
        snap = graph.get_node(name)
        if snap:
            health_by_node[name] = snap.health_status
            on_call_by_node[name] = [snap.on_call] if snap.on_call else []
        m = graph.get_metrics_snapshot(name)
        if m:
            metrics_by_node[name] = m.connection_saturation

    if health_overrides:
        health_by_node.update(health_overrides)

    return TopologyContext(
        graph=graph,
        focal_service=focal,
        blast_radius_nodes=blast_set,
        health_by_node=health_by_node,
        metrics_by_node=metrics_by_node,
        on_call_by_node=on_call_by_node,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candidate_names(analysis: IncidentAnalysis) -> set[str]:
    return {c.candidate_node for c in analysis.root_cause_candidates}


def _rejected_names(analysis: IncidentAnalysis) -> set[str]:
    return {r.hypothesis.candidate_node for r in analysis.rejected_hypotheses}


def _all_path_nodes(analysis: IncidentAnalysis) -> set[str]:
    """All nodes mentioned in any causal_path of any valid candidate."""
    nodes: set[str] = set()
    for c in analysis.root_cause_candidates:
        nodes.update(c.causal_path)
    return nodes


# ===========================================================================
# TestSymbolicValidatorRules
# ===========================================================================


class TestSymbolicValidatorRules:
    """Unit tests for each symbolic rule in isolation."""

    @pytest.fixture(autouse=True)
    def setup(self, graph):
        self.graph = graph
        self.ctx = _make_context(graph, "payments-service")
        self.validator = SymbolicValidator(self.ctx)

    def _hyp(
        self,
        candidate: str,
        path: Optional[list[str]] = None,
        score: float = 0.9,
    ) -> CausalHypothesis:
        return CausalHypothesis(
            candidate_node=candidate,
            template_key="connection_pool_exhausted",
            evidence=[],
            raw_score=score,
            causal_path=path or [candidate, "payments-service"],
        )

    # ── R1 ──────────────────────────────────────────────────────────────

    def test_r1_rejects_nonexistent_node(self):
        h = self._hyp("completely-made-up-svc")
        _, rejected = self.validator.validate_all([h])
        assert len(rejected) == 1
        assert rejected[0].violated_rule == R1_NODE_EXISTS

    def test_r1_passes_for_known_node(self):
        h = self._hyp("payments-db")
        valid, rejected = self.validator.validate_all([h])
        assert len(valid) == 1
        assert len(rejected) == 0

    def test_r1_reason_contains_candidate_name(self):
        h = self._hyp("ghost-service-xyz")
        _, rejected = self.validator.validate_all([h])
        assert "ghost-service-xyz" in rejected[0].reason

    # ── R2 ──────────────────────────────────────────────────────────────

    def test_r2_rejects_node_outside_blast_radius(self):
        """auth-service is not in payments-service blast radius."""
        h = self._hyp("auth-service", path=["auth-service", "payments-service"])
        _, rejected = self.validator.validate_all([h])
        r2_rejections = [r for r in rejected if r.violated_rule == R2_BLAST_RADIUS]
        assert len(r2_rejections) == 1

    def test_r2_accepts_node_inside_blast_radius(self):
        """payments-db IS in payments-service blast radius."""
        h = self._hyp("payments-db")
        valid, rejected = self.validator.validate_all([h])
        assert "payments-db" in [v.candidate_node for v in valid]
        r2_rejections = [r for r in rejected if r.violated_rule == R2_BLAST_RADIUS]
        assert len(r2_rejections) == 0

    def test_r2_accepts_focal_service_itself(self):
        """Self-fault: focal service is always in scope."""
        h = self._hyp("payments-service", path=["payments-service"])
        valid, _ = self.validator.validate_all([h])
        assert any(v.candidate_node == "payments-service" for v in valid)

    def test_r2_reason_mentions_blast_radius(self):
        h = self._hyp("inventory-db", path=["inventory-db", "payments-service"])
        _, rejected = self.validator.validate_all([h])
        r2 = [r for r in rejected if r.violated_rule == R2_BLAST_RADIUS]
        assert len(r2) >= 1
        assert "blast radius" in r2[0].reason.lower()

    # ── R3 ──────────────────────────────────────────────────────────────

    def test_r3_rejects_invalid_edge_in_path(self):
        """payments-db → auth-service is not a real topology edge."""
        h = self._hyp(
            "payments-db",
            path=["payments-db", "auth-service", "payments-service"],
        )
        _, rejected = self.validator.validate_all([h])
        r3_rejections = [r for r in rejected if r.violated_rule == R3_PATH_VALID]
        assert len(r3_rejections) == 1

    def test_r3_accepts_valid_path(self):
        """payments-db → payments-service is a real dependency edge."""
        h = self._hyp("payments-db", path=["payments-db", "payments-service"])
        valid, rejected = self.validator.validate_all([h])
        r3_rejections = [r for r in rejected if r.violated_rule == R3_PATH_VALID]
        assert len(r3_rejections) == 0

    def test_r3_accepts_single_node_path(self):
        """Self-fault path with one node has no edges to check."""
        h = self._hyp("payments-service", path=["payments-service"])
        valid, rejected = self.validator.validate_all([h])
        r3_rejections = [r for r in rejected if r.violated_rule == R3_PATH_VALID]
        assert len(r3_rejections) == 0

    # ── R4 ──────────────────────────────────────────────────────────────

    def test_r4_rejects_downstream_only_node(self):
        """
        api-gateway is a CONSUMER of payments-service (api-gateway DEPENDS ON
        payments-service), so payments-service cannot reach api-gateway following
        dependency edges — api-gateway is NOT an upstream dependency.
        """
        # We need to put api-gateway in the blast radius context manually
        ctx = _make_context(self.graph, "payments-service")
        ctx.blast_radius_nodes.add("api-gateway")
        validator = SymbolicValidator(ctx)
        # Use single-node path so R3 passes and we hit R4
        h = self._hyp("api-gateway", path=["api-gateway"])
        _, rejected = validator.validate_all([h])
        r4_rejections = [r for r in rejected if r.violated_rule == R4_DIRECTION]
        assert len(r4_rejections) == 1

    def test_r4_accepts_upstream_dependency(self):
        """payments-db IS upstream of payments-service."""
        h = self._hyp("payments-db", path=["payments-db", "payments-service"])
        valid, _ = self.validator.validate_all([h])
        assert any(v.candidate_node == "payments-db" for v in valid)

    def test_r4_accepts_self_fault(self):
        """Focal service is always its own upstream (trivially)."""
        h = self._hyp("payments-service", path=["payments-service"])
        valid, rejected = self.validator.validate_all([h])
        r4_rejections = [r for r in rejected if r.violated_rule == R4_DIRECTION]
        assert len(r4_rejections) == 0

    # ── R5 ──────────────────────────────────────────────────────────────

    def test_r5_reduces_score_for_healthy_low_saturation(self):
        """R5 is advisory — it reduces score by 30%, does NOT reject."""
        ctx = _make_context(self.graph, "payments-service")
        ctx.health_by_node["payments-db"] = "healthy"
        ctx.metrics_by_node["payments-db"] = 0.10  # low saturation
        validator = SymbolicValidator(ctx)
        h = self._hyp("payments-db", score=1.0)
        valid, rejected = validator.validate_all([h])
        # Must NOT be rejected
        assert len(rejected) == 0 or all(
            r.violated_rule != R5_HEALTH_COHERENCE for r in rejected
        )
        # Score must be reduced
        assert len(valid) == 1
        assert valid[0].raw_score < 1.0  # 30% reduction applied

    def test_r5_does_not_reject(self):
        """R5 never causes a rejection even for perfectly healthy nodes."""
        ctx = _make_context(self.graph, "payments-service")
        ctx.health_by_node["payments-db"] = "healthy"
        ctx.metrics_by_node["payments-db"] = 0.0
        validator = SymbolicValidator(ctx)
        h = self._hyp("payments-db")
        valid, rejected = validator.validate_all([h])
        assert len(valid) == 1

    def test_r5_does_not_penalize_critical_node(self):
        """Nodes with critical health are not penalized by R5."""
        ctx = _make_context(self.graph, "payments-service")
        ctx.health_by_node["payments-db"] = "critical"
        ctx.metrics_by_node["payments-db"] = 1.0
        validator = SymbolicValidator(ctx)
        h = self._hyp("payments-db", score=0.9)
        valid, _ = validator.validate_all([h])
        assert len(valid) == 1
        assert valid[0].raw_score == pytest.approx(0.9, rel=0.01)

    # ── Rule ordering (fail-fast) ────────────────────────────────────────

    def test_fail_fast_r1_before_r2(self):
        """
        A nonexistent node must fail R1, not R2 — even if it would also
        be outside the blast radius.
        """
        h = self._hyp("invented-node-abc")
        _, rejected = self.validator.validate_all([h])
        assert rejected[0].violated_rule == R1_NODE_EXISTS

    def test_rejected_hypotheses_have_nonempty_reason(self):
        """Every RejectedHypothesis must carry a non-empty reason string."""
        hypotheses = [
            self._hyp("ghost-svc-1"),           # R1
            self._hyp("auth-service",            # R2
                      path=["auth-service", "payments-service"]),
            self._hyp("payments-db",             # valid
                      path=["payments-db", "payments-service"]),
        ]
        _, rejected = self.validator.validate_all(hypotheses)
        for r in rejected:
            assert r.violated_rule, "violated_rule must not be empty"
            assert r.reason, "reason must not be empty"
            assert len(r.reason) > 10, "reason must be descriptive"


# ===========================================================================
# TestHypothesisEngine
# ===========================================================================


class TestHypothesisEngine:
    """Unit tests for HypothesisEngine candidate generation."""

    @pytest.fixture(autouse=True)
    def setup(self, graph):
        self.graph = graph

    def test_database_template_generates_db_candidates(self):
        ctx = _make_context(self.graph, "payments-service")
        engine = HypothesisEngine(ctx)
        report = _make_report(
            "ERROR: connection pool exhausted — 100/100 connections in use",
            "connection_pool_exhausted",
            tier="L1",
            confidence=1.0,
        )
        candidates = engine.generate(report)
        candidate_names = {c.candidate_node for c in candidates}
        # payments-db is a database node in the blast radius
        assert "payments-db" in candidate_names

    def test_database_template_excludes_non_db_nodes(self):
        """connection_pool_exhausted should prefer db nodes, not service nodes."""
        ctx = _make_context(self.graph, "payments-service")
        engine = HypothesisEngine(ctx)
        report = _make_report(
            "ERROR: connection pool exhausted — 100/100",
            "connection_pool_exhausted",
            tier="L1",
            confidence=1.0,
        )
        candidates = engine.generate(report)
        # api-gateway is a service node, not a database → should not be in candidates
        # (unless it happens to match "any" type hint, which connection_pool doesn't)
        candidate_names = {c.candidate_node for c in candidates}
        # api-gateway is NOT a database → should be absent
        assert "api-gateway" not in candidate_names

    def test_l3_novel_generates_any_type_candidates(self):
        """Novel L3 patterns use 'any' hint → all blast radius nodes are candidates."""
        ctx = _make_context(self.graph, "payments-service")
        engine = HypothesisEngine(ctx)
        report = _make_report(
            "ERR: Network bandwidth utilization at 95%% — packets dropping on eth0",
            "network_bandwidth_drop",  # novel, not in L1 catalog
            tier="L3",
            confidence=0.65,
        )
        candidates = engine.generate(report)
        candidate_names = {c.candidate_node for c in candidates}
        # All db nodes in blast radius should be present (since type=any)
        assert "payments-db" in candidate_names

    def test_l3_scores_lower_than_l1(self):
        """L3 tier weight (0.50) produces lower raw scores than L1 (1.00)."""
        ctx = _make_context(self.graph, "payments-service")
        engine = HypothesisEngine(ctx)

        l1_report = _make_report(
            "ERROR: connection pool exhausted",
            "connection_pool_exhausted",
            tier="L1",
            confidence=1.0,
        )
        l3_report = _make_report(
            "ERR: Something weird happened",
            "network_bandwidth_drop",
            tier="L3",
            confidence=1.0,
        )

        l1_candidates = engine.generate(l1_report)
        l3_candidates = engine.generate(l3_report)

        # Find payments-db in each
        l1_db = next((c for c in l1_candidates if c.candidate_node == "payments-db"), None)
        l3_all = [c for c in l3_candidates if c.candidate_node in {"payments-db", "payments-service"}]

        if l1_db and l3_all:
            l3_max = max(c.raw_score for c in l3_all)
            assert l1_db.raw_score > l3_max or l1_db.raw_score == pytest.approx(1.0)

    def test_correlation_boost_applied(self):
        """Nodes in correlation partners get +0.15 boost."""
        ctx = _make_context(self.graph, "payments-service")
        br = self.graph.blast_radius("payments-service")
        blast_set = set(br.affected_services)
        # Pick a database node that's in blast radius to be the "correlated partner"
        db_partners = {
            n for n in blast_set
            if (snap := self.graph.get_node(n)) and snap.node_type == "database"
        }
        if not db_partners:
            pytest.skip("No database partners in blast radius")

        partner = next(iter(db_partners))

        # Without boost
        engine_no_boost = HypothesisEngine(ctx, correlation_partners=set())
        # With boost
        engine_boosted = HypothesisEngine(ctx, correlation_partners={partner})

        report = _make_report(
            "ERROR: connection pool exhausted",
            "connection_pool_exhausted",
            tier="L1",
            confidence=0.9,
        )

        no_boost_cands = engine_no_boost.generate(report)
        boosted_cands = engine_boosted.generate(report)

        no_boost_score = next(
            (c.raw_score for c in no_boost_cands if c.candidate_node == partner), None
        )
        boosted_score = next(
            (c.raw_score for c in boosted_cands if c.candidate_node == partner), None
        )

        if no_boost_score is not None and boosted_score is not None:
            assert boosted_score > no_boost_score

    def test_causal_path_is_reversed_dependency_path(self):
        """
        causal_path must run candidate → focal (reversed dependency direction).
        For payments-db → payments-service: the dependency graph has
        payments-service → payments-db, so the causal path = [payments-db, payments-service].
        """
        ctx = _make_context(self.graph, "payments-service")
        engine = HypothesisEngine(ctx)
        report = _make_report(
            "ERROR: connection pool exhausted",
            "connection_pool_exhausted",
            tier="L1",
            confidence=1.0,
        )
        candidates = engine.generate(report)
        db_cand = next((c for c in candidates if c.candidate_node == "payments-db"), None)
        assert db_cand is not None, "payments-db must be a candidate"
        # Path must start with payments-db (root cause)
        assert db_cand.causal_path[0] == "payments-db"
        # Path must end with payments-service (focal)
        assert db_cand.causal_path[-1] == "payments-service"

    def test_self_fault_path_is_single_node(self):
        """When candidate == focal service, causal_path = [focal_service]."""
        ctx = _make_context(self.graph, "payments-service")
        engine = HypothesisEngine(ctx)
        report = _make_report(
            "ERROR: CrashLoopBackOff",
            "pod_crash_loop",
            tier="L1",
            confidence=0.95,
        )
        candidates = engine.generate(report)
        focal_cand = next(
            (c for c in candidates if c.candidate_node == "payments-service"), None
        )
        assert focal_cand is not None, "payments-service (self-fault) must be a candidate"
        assert focal_cand.causal_path == ["payments-service"]

    def test_deduplication_keeps_highest_score(self):
        """Same candidate node from multiple events → only highest score retained."""
        tier_enum = Tier("L1")
        r1 = RouterResult(
            log_entry="ERROR: pool exhausted",
            tier=tier_enum,
            template_key="connection_pool_exhausted",
            confidence=0.8,
        )
        r2 = RouterResult(
            log_entry="ERROR: connection timeout",
            tier=tier_enum,
            template_key="database_query_timeout",
            confidence=1.0,
        )
        report = PerceptionReport(
            results=[r1, r2],
            primary_template="connection_pool_exhausted",
            l1_hits=2,
        )
        ctx = _make_context(self.graph, "payments-service")
        engine = HypothesisEngine(ctx)
        candidates = engine.generate(report)

        # payments-db should appear exactly once
        db_candidates = [c for c in candidates if c.candidate_node == "payments-db"]
        assert len(db_candidates) == 1
        # Score should be the highest across all matching results
        assert db_candidates[0].raw_score == pytest.approx(1.0, rel=0.01)


# ===========================================================================
# TestSimulatedFailureScenarios
# ===========================================================================


class TestSimulatedFailureScenarios:
    """
    Five end-to-end scenarios using ReasoningEngine(graph=g).
    Validates that the reasoning layer correctly identifies root causes
    without hallucinating impossible connections.
    """

    @pytest.fixture(autouse=True)
    def setup(self, isolated_graph):
        self.graph = isolated_graph

    # ── Scenario S1: payments-db connection pool exhausted ──────────────

    @pytest.mark.asyncio
    async def test_s1_identifies_payments_db_as_root_cause(self):
        """
        S1: payments-service reporting connection pool exhaustion.
        Root cause = payments-db (the DB it depends on, 100/100 connections).
        """
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(
            "ERROR: QueuePool limit of size 100 overflow 0 reached, connection timed out",
            "connection_pool_exhausted",
            tier="L1",
            confidence=1.0,
            description="Database/HTTP connection pool fully saturated",
        )
        analysis = await engine.analyze_incident(report, "payments-service")

        assert "payments-db" in _candidate_names(analysis), (
            "payments-db must be identified as a root cause candidate"
        )

    @pytest.mark.asyncio
    async def test_s1_rejects_auth_service(self):
        """S1: auth-service is NOT in payments-service blast radius → rejected."""
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(
            "ERROR: QueuePool limit of size 100 overflow 0 reached",
            "connection_pool_exhausted",
            tier="L1",
            confidence=1.0,
        )
        analysis = await engine.analyze_incident(report, "payments-service")

        assert "auth-service" not in _candidate_names(analysis), (
            "auth-service must be rejected — outside payments-service blast radius"
        )

    @pytest.mark.asyncio
    async def test_s1_rejects_thirdparty_sso(self):
        """S1: thirdparty-sso is not in payments-service blast radius."""
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(
            "ERROR: connection pool exhausted — 100/100 connections in use",
            "connection_pool_exhausted",
            tier="L1",
            confidence=1.0,
        )
        analysis = await engine.analyze_incident(report, "payments-service")

        assert "thirdparty-sso" not in _candidate_names(analysis)

    @pytest.mark.asyncio
    async def test_s1_rejects_inventory_db(self):
        """S1: inventory-db is outside the blast radius."""
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(
            "ERROR: too many clients already",
            "connection_pool_exhausted",
            tier="L1",
            confidence=1.0,
        )
        analysis = await engine.analyze_incident(report, "payments-service")
        assert "inventory-db" not in _candidate_names(analysis)

    @pytest.mark.asyncio
    async def test_s1_symbolic_path_is_fast(self):
        """S1: All L1 logs → SYMBOLIC_FAST path."""
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(
            "ERROR: connection pool exhausted",
            "connection_pool_exhausted",
            tier="L1",
            confidence=1.0,
        )
        analysis = await engine.analyze_incident(report, "payments-service")
        assert analysis.symbolic_path == "SYMBOLIC_FAST"

    @pytest.mark.asyncio
    async def test_s1_payments_db_causal_path_is_valid(self):
        """S1: causal path for payments-db must be [payments-db, payments-service]."""
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(
            "ERROR: QueuePool limit of size 100 overflow 0 reached",
            "connection_pool_exhausted",
            tier="L1",
            confidence=1.0,
        )
        analysis = await engine.analyze_incident(report, "payments-service")

        db_rch = next(
            (c for c in analysis.root_cause_candidates if c.candidate_node == "payments-db"),
            None,
        )
        assert db_rch is not None
        path = db_rch.causal_path
        assert path[0] == "payments-db"
        assert path[-1] == "payments-service"
        # Every consecutive pair must be a real dependency edge
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            assert self.graph._g.has_edge(b, a), (
                f"Causal path step ({a}→{b}) is not a real dependency edge"
            )

    @pytest.mark.asyncio
    async def test_s1_stale_metrics_warning_set(self):
        """S1: Every RootCauseHypothesis must have stale_metrics_warning=True."""
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(
            "ERROR: connection pool exhausted",
            "connection_pool_exhausted",
            tier="L1",
            confidence=1.0,
        )
        analysis = await engine.analyze_incident(report, "payments-service")
        for rch in analysis.root_cause_candidates:
            assert rch.stale_metrics_warning is True, (
                f"stale_metrics_warning must be True for {rch.candidate_node} "
                "— metrics are not live in Stage 2"
            )

    # ── Scenario S2: Network bandwidth drop (novel L3 pattern) ──────────

    @pytest.mark.asyncio
    async def test_s2_network_bandwidth_drop_no_hallucination(self):
        """
        S2: Novel network bandwidth drop log (L3 tier, unknown pattern).
        All candidates must be topologically reachable from payments-service.
        """
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(
            "ERR: Network bandwidth utilization at 95% — packets dropping on eth0",
            "network_bandwidth_drop",  # novel, not in L1 catalog
            tier="L3",
            confidence=0.65,
            description="Novel network saturation pattern",
        )
        analysis = await engine.analyze_incident(report, "payments-service")

        # All valid candidates must be reachable (R4 ensures this)
        for rch in analysis.root_cause_candidates:
            candidate = rch.candidate_node
            assert self.graph._g.has_node(candidate), (
                f"Hallucinated node '{candidate}' not in topology"
            )
            # Focal must be able to reach candidate (or candidate == focal)
            if candidate != "payments-service":
                import networkx as nx
                reachable = nx.has_path(
                    self.graph._g, "payments-service", candidate
                )
                assert reachable, (
                    f"HALLUCINATION: '{candidate}' is not an upstream dependency "
                    "of 'payments-service' — R4 should have rejected this"
                )

    @pytest.mark.asyncio
    async def test_s2_rejects_auth_db(self):
        """S2: auth-db is outside payments-service blast radius."""
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(
            "ERR: Network bandwidth utilization at 95%",
            "network_bandwidth_drop",
            tier="L3",
            confidence=0.65,
        )
        analysis = await engine.analyze_incident(report, "payments-service")
        assert "auth-db" not in _candidate_names(analysis)

    @pytest.mark.asyncio
    async def test_s2_rejects_inventory_db(self):
        """S2: inventory-db is outside blast radius."""
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(
            "ERR: Network bandwidth utilization at 95%",
            "network_bandwidth_drop",
            tier="L3",
            confidence=0.65,
        )
        analysis = await engine.analyze_incident(report, "payments-service")
        assert "inventory-db" not in _candidate_names(analysis)

    @pytest.mark.asyncio
    async def test_s2_symbolic_path_is_neural_full(self):
        """S2: L3 tier event → NEURAL_FULL symbolic path."""
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(
            "ERR: Network bandwidth utilization at 95%",
            "network_bandwidth_drop",
            tier="L3",
            confidence=0.65,
        )
        analysis = await engine.analyze_incident(report, "payments-service")
        assert analysis.symbolic_path == "NEURAL_FULL"

    @pytest.mark.asyncio
    async def test_s2_all_rejected_have_rule_and_reason(self):
        """S2: Every rejected hypothesis carries a violated_rule and reason."""
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(
            "ERR: Network bandwidth utilization at 95%",
            "network_bandwidth_drop",
            tier="L3",
            confidence=0.65,
        )
        analysis = await engine.analyze_incident(report, "payments-service")
        for rh in analysis.rejected_hypotheses:
            assert rh.violated_rule, f"Missing violated_rule for {rh.hypothesis.candidate_node}"
            assert rh.reason, f"Missing reason for {rh.hypothesis.candidate_node}"

    # ── Scenario S3: redis_oom_eviction → user-profile-service ──────────

    @pytest.mark.asyncio
    async def test_s3_identifies_redis_as_root_cause(self):
        """
        S3: user-profile-service reporting Redis OOM eviction.
        Expected root cause: redis-user-cache.
        """
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(
            "CRITICAL: OOM command not allowed when used memory > 'maxmemory'",
            "redis_oom_eviction",
            tier="L1",
            confidence=1.0,
            description="Redis maxmemory policy triggering evictions",
        )
        analysis = await engine.analyze_incident(report, "user-profile-service")

        assert "redis-user-cache" in _candidate_names(analysis), (
            "redis-user-cache must be identified as the root cause"
        )

    @pytest.mark.asyncio
    async def test_s3_rejects_auth_db(self):
        """S3: auth-db is not in user-profile-service blast radius."""
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(
            "CRITICAL: OOM command not allowed",
            "redis_oom_eviction",
            tier="L1",
            confidence=1.0,
        )
        analysis = await engine.analyze_incident(report, "user-profile-service")
        assert "auth-db" not in _candidate_names(analysis)

    @pytest.mark.asyncio
    async def test_s3_rejects_payments_db(self):
        """S3: payments-db is not in user-profile-service blast radius."""
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(
            "CRITICAL: maxmemory limit reached",
            "redis_oom_eviction",
            tier="L1",
            confidence=1.0,
        )
        analysis = await engine.analyze_incident(report, "user-profile-service")
        assert "payments-db" not in _candidate_names(analysis)

    @pytest.mark.asyncio
    async def test_s3_redis_causal_path_valid(self):
        """S3: redis-user-cache → user-profile-service path uses real edges."""
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(
            "CRITICAL: OOM command not allowed when used memory > maxmemory",
            "redis_oom_eviction",
            tier="L1",
            confidence=1.0,
        )
        analysis = await engine.analyze_incident(report, "user-profile-service")

        redis_rch = next(
            (c for c in analysis.root_cause_candidates
             if c.candidate_node == "redis-user-cache"),
            None,
        )
        assert redis_rch is not None, "redis-user-cache must be a valid candidate"
        path = redis_rch.causal_path
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            assert self.graph._g.has_edge(b, a), (
                f"S3 causal path step ({a}→{b}) not a real dependency edge"
            )

    # ── Scenario S4: tls_cert_expired → auth-service ────────────────────

    @pytest.mark.asyncio
    async def test_s4_identifies_thirdparty_sso_as_root_cause(self):
        """
        S4: auth-service reporting TLS certificate verification failure.
        Expected root cause: thirdparty-sso (auth-service depends on it).
        """
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(
            "ERROR: CERTIFICATE_VERIFY_FAILED — certificate has expired",
            "tls_cert_expired",
            tier="L1",
            confidence=1.0,
            description="TLS/X.509 certificate expired or invalid",
        )
        analysis = await engine.analyze_incident(report, "auth-service")

        assert "thirdparty-sso" in _candidate_names(analysis), (
            "thirdparty-sso (external auth dependency) must be identified"
        )

    @pytest.mark.asyncio
    async def test_s4_rejects_payments_db(self):
        """S4: payments-db is not in auth-service blast radius."""
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(
            "ERROR: CERTIFICATE_VERIFY_FAILED",
            "tls_cert_expired",
            tier="L1",
            confidence=1.0,
        )
        analysis = await engine.analyze_incident(report, "auth-service")
        assert "payments-db" not in _candidate_names(analysis)

    @pytest.mark.asyncio
    async def test_s4_rejects_inventory_db(self):
        """S4: inventory-db is not in auth-service blast radius."""
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(
            "ERROR: SSL handshake failed — certificate has expired",
            "tls_cert_expired",
            tier="L1",
            confidence=1.0,
        )
        analysis = await engine.analyze_incident(report, "auth-service")
        assert "inventory-db" not in _candidate_names(analysis)

    # ── Scenario S5: api-gateway mixed logs ──────────────────────────────

    @pytest.mark.asyncio
    async def test_s5_api_gateway_no_far_nodes(self):
        """
        S5: api-gateway as focal service.
        All valid candidates must be within depth-2 blast radius.
        Services farther than 2 hops must be rejected.
        """
        engine = ReasoningEngine(graph=self.graph)
        # Mixed: network issue + http upstream failure
        tier_enum = Tier("L1")
        results = [
            RouterResult(
                log_entry="503 Service Unavailable — upstream connect error",
                tier=tier_enum,
                template_key="http_upstream_unavailable",
                confidence=1.0,
                description="Upstream HTTP service unreachable",
            ),
            RouterResult(
                log_entry="ERROR: grpc DEADLINE_EXCEEDED calling payments-service",
                tier=tier_enum,
                template_key="grpc_deadline_exceeded",
                confidence=0.95,
                description="gRPC call exceeded deadline",
            ),
        ]
        report = PerceptionReport(
            results=results,
            primary_template="http_upstream_unavailable",
            l1_hits=2,
        )

        analysis = await engine.analyze_incident(report, "api-gateway")

        # Compute expected blast radius
        br = self.graph.blast_radius("api-gateway")
        blast_set = set(br.affected_services) | {"api-gateway"}

        for rch in analysis.root_cause_candidates:
            assert rch.candidate_node in blast_set or rch.candidate_node == "api-gateway", (
                f"HALLUCINATION: '{rch.candidate_node}' is outside api-gateway blast radius"
            )

    @pytest.mark.asyncio
    async def test_s5_causal_graph_has_valid_structure(self):
        """S5: CausalGraph must have focal service as a node."""
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(
            "503 Service Unavailable — upstream connect error",
            "http_upstream_unavailable",
            tier="L1",
            confidence=1.0,
        )
        analysis = await engine.analyze_incident(report, "api-gateway")

        causal_node_names = {n.name for n in analysis.causal_graph.nodes}
        assert "api-gateway" in causal_node_names, "Focal service must appear in CausalGraph"

    @pytest.mark.asyncio
    async def test_s5_analysis_markdown_not_empty(self):
        """S5: analysis_markdown must be non-empty and mention focal service."""
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(
            "503 Service Unavailable",
            "http_upstream_unavailable",
            tier="L1",
            confidence=1.0,
        )
        analysis = await engine.analyze_incident(report, "api-gateway")
        assert len(analysis.analysis_markdown) > 100
        assert "api-gateway" in analysis.analysis_markdown


# ===========================================================================
# TestCausalInvariants — applies to ALL scenarios
# ===========================================================================


class TestCausalInvariants:
    """
    Parametrized invariant tests that must hold for every scenario output.

    These are the primary anti-hallucination guarantees:
    - No node in any causal path may be absent from the topology
    - All causal path edges must be real dependency edges
    - No valid candidate may be outside depth-2 blast radius
    """

    SCENARIO_PARAMS = [
        pytest.param(
            "payments-service",
            "connection_pool_exhausted",
            "L1",
            1.0,
            "ERROR: connection pool exhausted — 100/100",
            id="S1_payments_pool",
        ),
        pytest.param(
            "payments-service",
            "network_bandwidth_drop",
            "L3",
            0.65,
            "ERR: Network bandwidth at 95%",
            id="S2_network_bandwidth",
        ),
        pytest.param(
            "user-profile-service",
            "redis_oom_eviction",
            "L1",
            1.0,
            "CRITICAL: OOM command not allowed",
            id="S3_redis_oom",
        ),
        pytest.param(
            "auth-service",
            "tls_cert_expired",
            "L1",
            1.0,
            "ERROR: CERTIFICATE_VERIFY_FAILED",
            id="S4_tls_cert",
        ),
        pytest.param(
            "api-gateway",
            "http_upstream_unavailable",
            "L1",
            0.95,
            "503 Service Unavailable",
            id="S5_api_gateway",
        ),
    ]

    @pytest.fixture(autouse=True)
    def setup(self, isolated_graph):
        self.graph = isolated_graph

    @pytest.mark.asyncio
    @pytest.mark.parametrize("focal,template,tier,conf,log", SCENARIO_PARAMS)
    async def test_no_hallucinated_nodes_in_any_path(
        self, focal, template, tier, conf, log
    ):
        """Every node in every causal_path must exist in the topology."""
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(log, template, tier=tier, confidence=conf)
        analysis = await engine.analyze_incident(report, focal)

        all_nodes = _all_path_nodes(analysis)
        for node_name in all_nodes:
            assert self.graph._g.has_node(node_name), (
                f"HALLUCINATION: Node '{node_name}' in causal_path "
                f"does not exist in topology (focal={focal})"
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("focal,template,tier,conf,log", SCENARIO_PARAMS)
    async def test_all_candidates_in_blast_radius(
        self, focal, template, tier, conf, log
    ):
        """Every valid RootCauseHypothesis.candidate_node must be in depth-2 blast radius."""
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(log, template, tier=tier, confidence=conf)
        analysis = await engine.analyze_incident(report, focal)

        br = self.graph.blast_radius(focal)
        blast_set = set(br.affected_services) | {focal}

        for rch in analysis.root_cause_candidates:
            assert rch.candidate_node in blast_set, (
                f"INVARIANT VIOLATION: '{rch.candidate_node}' is a valid candidate "
                f"but is NOT in the depth-{BLAST_RADIUS_DEPTH} blast radius of '{focal}'"
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("focal,template,tier,conf,log", SCENARIO_PARAMS)
    async def test_all_causal_path_edges_are_real(
        self, focal, template, tier, conf, log
    ):
        """Every edge in every causal_path must correspond to a real dependency."""
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(log, template, tier=tier, confidence=conf)
        analysis = await engine.analyze_incident(report, focal)

        for rch in analysis.root_cause_candidates:
            path = rch.causal_path
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                # a causes b: b depends_on a → g.has_edge(b, a)
                assert self.graph._g.has_edge(b, a), (
                    f"HALLUCINATED EDGE: '{a}' → '{b}' in causal_path "
                    f"of candidate '{rch.candidate_node}' (focal={focal}) "
                    f"is NOT a real dependency edge"
                )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("focal,template,tier,conf,log", SCENARIO_PARAMS)
    async def test_all_rejected_have_structured_reason(
        self, focal, template, tier, conf, log
    ):
        """Every RejectedHypothesis must carry a violated_rule and reason."""
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(log, template, tier=tier, confidence=conf)
        analysis = await engine.analyze_incident(report, focal)

        for rh in analysis.rejected_hypotheses:
            assert rh.violated_rule, (
                f"Missing violated_rule for rejected candidate "
                f"'{rh.hypothesis.candidate_node}' (focal={focal})"
            )
            assert rh.reason, (
                f"Missing reason for rejected candidate "
                f"'{rh.hypothesis.candidate_node}' (focal={focal})"
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("focal,template,tier,conf,log", SCENARIO_PARAMS)
    async def test_analysis_has_required_fields(
        self, focal, template, tier, conf, log
    ):
        """IncidentAnalysis must have all required fields populated."""
        engine = ReasoningEngine(graph=self.graph)
        report = _make_report(log, template, tier=tier, confidence=conf)
        analysis = await engine.analyze_incident(report, focal)

        assert analysis.focal_service == focal
        assert analysis.causal_graph is not None
        assert analysis.causal_graph.focal_service == focal
        assert analysis.symbolic_path in {"SYMBOLIC_FAST", "CBR_GUIDED", "NEURAL_FULL"}
        assert 0.0 <= analysis.overall_confidence <= 1.0
        assert len(analysis.analysis_markdown) > 0
        assert analysis.analyzed_at  # non-empty ISO timestamp


# ===========================================================================
# TestMCPIntegrationReasoning
# ===========================================================================


class TestMCPIntegrationReasoning:
    """
    End-to-end integration test: ReasoningEngine() spawns the real MCP server
    subprocess and calls its tools for topology data, then runs the full
    reasoning pipeline.
    """

    @pytest.mark.asyncio
    async def test_mcp_connection_pool_scenario(self):
        """
        Full MCP round-trip: connection pool exhausted → payments-service.
        ReasoningEngine() uses the subprocess transport.
        """
        engine = ReasoningEngine()  # No injected graph → uses MCP subprocess
        report = _make_report(
            "ERROR: QueuePool limit of size 100 overflow 0 reached, connection timed out",
            "connection_pool_exhausted",
            tier="L1",
            confidence=1.0,
        )
        analysis = await engine.analyze_incident(report, "payments-service")

        # Core correctness invariant
        assert "payments-db" in _candidate_names(analysis), (
            "MCP integration: payments-db must be identified via full subprocess pipeline"
        )
        assert "auth-service" not in _candidate_names(analysis), (
            "MCP integration: auth-service must be rejected (outside blast radius)"
        )

    @pytest.mark.asyncio
    async def test_mcp_analysis_markdown_roundtrip(self):
        """MCP mode: analysis_markdown must be populated and mention focal service."""
        engine = ReasoningEngine()
        report = _make_report(
            "CRITICAL: OOM command not allowed when used memory > maxmemory",
            "redis_oom_eviction",
            tier="L1",
            confidence=1.0,
        )
        analysis = await engine.analyze_incident(report, "user-profile-service")

        assert "user-profile-service" in analysis.analysis_markdown
        assert analysis.symbolic_path == "SYMBOLIC_FAST"
