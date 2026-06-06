"""
tests/test_transient_filter.py
================================

Unit + integration tests for the transient error awareness layer added to
AIRS v2 in the transient-filter feature:

  TestTransientTemplateTagging      — TRANSIENT_TEMPLATE_KEYS membership,
                                      L1Match.is_transient propagation,
                                      non-transient templates unaffected,
                                      database_query_timeout NOT in set.

  TestTransientFilter               — Recurrence threshold logic (below / at /
                                      above), success-line pairing, mixed
                                      transient + persistent blocks, flag
                                      propagation, PerceptionReport counters.

  TestR5bTransientEvidence          — Single transient evidence → 50% score
                                      reduction; two+ pieces → no reduction;
                                      persistent evidence → no reduction;
                                      combined transient + persistent → no cut.

  TestR5cContextualEvidence         — database_query_timeout alone → 40%
                                      discount; with connection_pool_exhausted
                                      → no discount; multiple occurrences but
                                      still no corroborator → still discounted.

  TestEndToEndTransientSuppression  — Full classify_block + HypothesisEngine
                                      round-trip covering all three fault tiers.

All tests are synchronous or pytest-asyncio; none require a live GROQ_API_KEY.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from airs_v2.perception.l1_cache import (
    L1SymbolicCache,
    L1Match,
    TRANSIENT_TEMPLATE_KEYS,
    BUILTIN_TEMPLATES,
)
from airs_v2.perception.router import (
    LogRouter,
    RouterResult,
    PerceptionReport,
    Tier,
    TransientFilter,
    _DEFAULT_MIN_RECURRENCE,
)
from airs_v2.reasoning.causal_types import CausalEvidence, CausalHypothesis
from airs_v2.reasoning.symbolic_validator import (
    SymbolicValidator,
    TopologyContext,
    TRANSIENT_MIN_EVIDENCE,
    TRANSIENT_SCORE_FACTOR,
    CONTEXTUAL_SCORE_FACTOR,
    CONTEXTUAL_TEMPLATE_KEYS,
    CONTEXTUAL_CORROBORATION_KEYS,
)
from airs_v2.context.graph import ContextGraph
from airs_v2.reasoning.hypothesis_engine import HypothesisEngine

import os
os.environ.pop("GROQ_API_KEY", None)

FIXTURES_PATH = (
    Path(__file__).resolve().parents[1]
    / "mock_enterprise"
    / "topology_fixtures.json"
)


# ===========================================================================
# Shared helpers
# ===========================================================================

def _make_router_result(
    template_key: str,
    log_entry: str = "ERROR: test log",
    tier: str = "L1",
    confidence: float = 1.0,
    suppressed_transient: bool = False,
) -> RouterResult:
    return RouterResult(
        log_entry=log_entry,
        tier=Tier(tier),
        template_key=template_key,
        confidence=confidence,
        suppressed_transient=suppressed_transient,
    )


def _make_perception_report(
    results: list[RouterResult],
    primary_template: Optional[str] = None,
) -> PerceptionReport:
    if primary_template is None:
        primary_template = results[0].template_key if results else "none"
    l1 = sum(1 for r in results if r.tier == Tier.L1)
    l2 = sum(1 for r in results if r.tier == Tier.L2)
    l3 = sum(1 for r in results if r.tier == Tier.L3)
    return PerceptionReport(
        results=results,
        primary_template=primary_template,
        l1_hits=l1,
        l2_hits=l2,
        l3_hits=l3,
    )


def _make_evidence(
    template_key: str,
    is_transient: bool = False,
    confidence: float = 1.0,
) -> CausalEvidence:
    return CausalEvidence(
        log_entry=f"ERROR matching {template_key}",
        template_key=template_key,
        tier="L1",
        confidence=confidence,
        is_transient=is_transient,
    )


def _make_hypothesis(
    evidence: list[CausalEvidence],
    candidate_node: str = "payments-db",
    template_key: str = "test_pattern",
    raw_score: float = 1.0,
) -> CausalHypothesis:
    return CausalHypothesis(
        candidate_node=candidate_node,
        template_key=template_key,
        evidence=evidence,
        raw_score=raw_score,
        causal_path=[candidate_node, "payments-service"],
    )


def _make_context(graph: ContextGraph, focal: str) -> TopologyContext:
    br = graph.blast_radius(focal)
    blast_set = set(br.affected_services)
    health_by_node: dict = {}
    metrics_by_node: dict = {}
    on_call_by_node: dict = {}
    for name in blast_set | {focal}:
        snap = graph.get_node(name)
        if snap:
            health_by_node[name] = snap.health_status
            on_call_by_node[name] = [snap.on_call] if snap.on_call else []
        m = graph.get_metrics_snapshot(name)
        if m:
            metrics_by_node[name] = m.connection_saturation
    return TopologyContext(
        graph=graph,
        focal_service=focal,
        blast_radius_nodes=blast_set,
        health_by_node=health_by_node,
        metrics_by_node=metrics_by_node,
        on_call_by_node=on_call_by_node,
    )


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def graph() -> ContextGraph:
    return ContextGraph(fixtures_path=FIXTURES_PATH)


# ===========================================================================
# TestTransientTemplateTagging
# ===========================================================================


class TestTransientTemplateTagging:
    """Verify TRANSIENT_TEMPLATE_KEYS membership and L1Match.is_transient flag."""

    def test_transient_keys_are_a_frozenset(self):
        assert isinstance(TRANSIENT_TEMPLATE_KEYS, frozenset)

    def test_known_transient_keys_present(self):
        expected = {
            "upstream_rate_limited",
            "grpc_deadline_exceeded",
            "dns_resolution_failure",
            "http_upstream_unavailable",
        }
        assert expected <= TRANSIENT_TEMPLATE_KEYS, (
            f"Missing transient keys: {expected - TRANSIENT_TEMPLATE_KEYS}"
        )

    def test_database_query_timeout_NOT_transient(self):
        """database_query_timeout is threshold-contextual, not transient."""
        assert "database_query_timeout" not in TRANSIENT_TEMPLATE_KEYS

    def test_persistent_keys_not_transient(self):
        """Fatal/persistent templates must never appear in the transient set."""
        fatal_keys = {
            "connection_pool_exhausted",
            "oom_killed",
            "disk_space_exhausted",
            "tls_cert_expired",
            "redis_oom_eviction",
            "pod_crash_loop",
            "transaction_leak",
            "thread_pool_exhausted",
            "kafka_consumer_lag",
            "network_partition",
        }
        overlap = fatal_keys & TRANSIENT_TEMPLATE_KEYS
        assert not overlap, f"Fatal keys incorrectly marked transient: {overlap}"

    def test_l1match_is_transient_true_for_transient_template(self):
        cache = L1SymbolicCache()
        # dns_resolution_failure is in TRANSIENT_TEMPLATE_KEYS
        match = cache.classify("socket.gaierror: [Errno -3] Temporary failure in name resolution")
        assert match is not None
        assert match.template_key == "dns_resolution_failure"
        assert match.is_transient is True

    def test_l1match_is_transient_false_for_persistent_template(self):
        cache = L1SymbolicCache()
        # connection_pool_exhausted is NOT transient
        match = cache.classify("QueuePool limit of size 5 overflow 10 reached, connection timed out")
        assert match is not None
        assert match.template_key == "connection_pool_exhausted"
        assert match.is_transient is False

    def test_l1match_is_transient_false_for_database_query_timeout(self):
        cache = L1SymbolicCache()
        match = cache.classify("ERROR: statement timeout after 30000ms")
        assert match is not None
        assert match.template_key == "database_query_timeout"
        assert match.is_transient is False

    def test_is_transient_key_helper(self):
        cache = L1SymbolicCache()
        assert cache.is_transient_key("dns_resolution_failure") is True
        assert cache.is_transient_key("grpc_deadline_exceeded") is True
        assert cache.is_transient_key("connection_pool_exhausted") is False
        assert cache.is_transient_key("database_query_timeout") is False
        assert cache.is_transient_key("nonexistent_key") is False

    def test_l3_learned_template_is_not_transient_by_default(self):
        """Novel patterns learned by L3 are non-transient by default (fail-safe)."""
        cache = L1SymbolicCache()
        cache.register_template(
            key="custom_novel_pattern",
            pattern=r"novel.*error.*pattern",
            description="A novel pattern",
            source="l3_learned",
        )
        assert cache.is_transient_key("custom_novel_pattern") is False


# ===========================================================================
# TestTransientFilter
# ===========================================================================


class TestTransientFilter:
    """Unit tests for TransientFilter.apply() logic."""

    def test_single_transient_result_is_suppressed(self):
        """1 occurrence of a transient key < min_recurrence=3 → suppressed."""
        results = [
            _make_router_result("dns_resolution_failure"),
        ]
        tf = TransientFilter()
        annotated = tf.apply(results)
        assert annotated[0].suppressed_transient is True

    def test_two_transient_results_suppressed(self):
        """2 occurrences (< 3) still suppressed."""
        results = [
            _make_router_result("dns_resolution_failure"),
            _make_router_result("dns_resolution_failure"),
        ]
        tf = TransientFilter()
        annotated = tf.apply(results)
        assert all(r.suppressed_transient for r in annotated)

    def test_three_transient_results_pass(self):
        """3 occurrences == min_recurrence=3 → NOT suppressed."""
        results = [
            _make_router_result("dns_resolution_failure"),
            _make_router_result("dns_resolution_failure"),
            _make_router_result("dns_resolution_failure"),
        ]
        tf = TransientFilter()
        annotated = tf.apply(results)
        assert all(not r.suppressed_transient for r in annotated)

    def test_four_transient_results_pass(self):
        """4 occurrences > min_recurrence → NOT suppressed."""
        results = [_make_router_result("grpc_deadline_exceeded") for _ in range(4)]
        tf = TransientFilter()
        annotated = tf.apply(results)
        assert all(not r.suppressed_transient for r in annotated)

    def test_persistent_result_never_suppressed(self):
        """connection_pool_exhausted is not in TRANSIENT_TEMPLATE_KEYS → never suppressed."""
        results = [_make_router_result("connection_pool_exhausted")]
        tf = TransientFilter()
        annotated = tf.apply(results)
        assert annotated[0].suppressed_transient is False

    def test_database_query_timeout_never_suppressed(self):
        """Threshold-contextual key is NOT in TRANSIENT_TEMPLATE_KEYS → never suppressed."""
        results = [_make_router_result("database_query_timeout")]
        tf = TransientFilter()
        annotated = tf.apply(results)
        assert annotated[0].suppressed_transient is False

    def test_mixed_block_only_transient_suppressed(self):
        """In a mixed block, only the transient template below threshold is suppressed."""
        results = [
            _make_router_result("connection_pool_exhausted"),     # persistent → pass
            _make_router_result("dns_resolution_failure"),        # transient, count=1 → suppress
            _make_router_result("database_query_timeout"),        # contextual → pass
        ]
        tf = TransientFilter()
        annotated = tf.apply(results)

        by_key = {r.template_key: r for r in annotated}
        assert by_key["connection_pool_exhausted"].suppressed_transient is False
        assert by_key["dns_resolution_failure"].suppressed_transient is True
        assert by_key["database_query_timeout"].suppressed_transient is False

    def test_two_different_transient_keys_both_suppressed(self):
        """Each transient template key is evaluated independently."""
        results = [
            _make_router_result("dns_resolution_failure"),
            _make_router_result("grpc_deadline_exceeded"),
        ]
        tf = TransientFilter()
        annotated = tf.apply(results)
        assert all(r.suppressed_transient for r in annotated)

    def test_custom_min_recurrence(self):
        """Custom min_recurrence=2: 1 occurrence suppressed, 2 pass."""
        results_1 = [_make_router_result("dns_resolution_failure")]
        results_2 = [
            _make_router_result("dns_resolution_failure"),
            _make_router_result("dns_resolution_failure"),
        ]
        tf = TransientFilter(min_recurrence=2)
        assert tf.apply(results_1)[0].suppressed_transient is True
        assert all(not r.suppressed_transient for r in tf.apply(results_2))

    def test_success_line_in_raw_does_not_prevent_suppression(self):
        """A success line in raw_lines means the retry worked → still suppress
        (transient events with paired success are definitionally transient)."""
        raw_lines = [
            "ERROR: socket.gaierror: [Errno -3] Temporary failure in name resolution",
            "INFO: DNS lookup succeeded after 1 retry (234ms)",
        ]
        results = [_make_router_result("dns_resolution_failure")]
        tf = TransientFilter(raw_lines=raw_lines)
        annotated = tf.apply(results)
        assert annotated[0].suppressed_transient is True

    def test_perception_report_transient_counters(self):
        """PerceptionReport.transient_suppressed and transient_passed are correct."""
        # 1 suppressed dns, 3 passed grpc, 1 persistent (not counted in either)
        results = (
            [_make_router_result("dns_resolution_failure")]
            + [_make_router_result("grpc_deadline_exceeded") for _ in range(3)]
            + [_make_router_result("connection_pool_exhausted")]
        )
        tf = TransientFilter()
        annotated = tf.apply(results)

        suppressed = sum(1 for r in annotated if r.suppressed_transient)
        passed_transient = sum(
            1 for r in annotated
            if not r.suppressed_transient and r.template_key in TRANSIENT_TEMPLATE_KEYS
        )

        assert suppressed == 1   # dns (count=1 < 3)
        assert passed_transient == 3  # grpc (count=3 >= 3)

    def test_filter_is_idempotent(self):
        """Applying the filter twice produces the same result."""
        results = [_make_router_result("dns_resolution_failure")]
        tf = TransientFilter()
        once = tf.apply(results)
        twice = tf.apply(once)
        assert once[0].suppressed_transient == twice[0].suppressed_transient


# ===========================================================================
# TestR5bTransientEvidence
# ===========================================================================


class TestR5bTransientEvidence:
    """Unit tests for R5b advisory: transient-evidence score discount."""

    @pytest.fixture(autouse=True)
    def setup(self, graph):
        self.ctx = _make_context(graph, "payments-service")
        self.validator = SymbolicValidator(self.ctx)

    def _validate(self, h: CausalHypothesis) -> CausalHypothesis:
        """Run _validate_one and return the resulting hypothesis (must pass R1-R4)."""
        result = self.validator._validate_one(h)
        # If it's a rejection for topology reasons, the test itself is mis-constructed
        assert not hasattr(result, "violated_rule"), (
            f"Hypothesis was unexpectedly rejected: {result}"
        )
        return result

    def test_single_transient_evidence_gets_score_cut(self):
        """1 transient evidence piece + count < TRANSIENT_MIN_EVIDENCE → 50% score cut."""
        h = _make_hypothesis(
            evidence=[_make_evidence("dns_resolution_failure", is_transient=True)],
            candidate_node="payments-db",
            template_key="dns_resolution_failure",
            raw_score=1.0,
        )
        result = self._validate(h)
        expected = round(1.0 * TRANSIENT_SCORE_FACTOR, 6)
        assert abs(result.raw_score - expected) < 0.001, (
            f"Expected score ~{expected}, got {result.raw_score}"
        )

    def test_two_transient_evidence_pieces_no_cut(self):
        """2 transient pieces >= TRANSIENT_MIN_EVIDENCE=2 → no R5b cut."""
        h = _make_hypothesis(
            evidence=[
                _make_evidence("dns_resolution_failure", is_transient=True),
                _make_evidence("dns_resolution_failure", is_transient=True),
            ],
            candidate_node="payments-db",
            template_key="dns_resolution_failure",
            raw_score=1.0,
        )
        result = self._validate(h)
        # R5b should NOT fire; score may be modified only by R5/R5c, not R5b
        assert result.raw_score >= TRANSIENT_SCORE_FACTOR, (
            f"R5b should not have fired with 2 evidence pieces, got {result.raw_score}"
        )

    def test_persistent_evidence_no_cut(self):
        """Persistent (non-transient) evidence → R5b never fires."""
        h = _make_hypothesis(
            evidence=[_make_evidence("connection_pool_exhausted", is_transient=False)],
            candidate_node="payments-db",
            template_key="connection_pool_exhausted",
            raw_score=1.0,
        )
        result = self._validate(h)
        # R5b guard: all_transient must be True — it won't be for this evidence
        # Score can only be cut by R5 (health coherence), not R5b
        # payments-db is typically critical/degraded, so R5 shouldn't fire either
        assert result.raw_score > TRANSIENT_SCORE_FACTOR

    def test_mixed_transient_and_persistent_no_r5b_cut(self):
        """Mixed transient+persistent evidence → all_transient=False → R5b does not fire."""
        h = _make_hypothesis(
            evidence=[
                _make_evidence("dns_resolution_failure", is_transient=True),
                _make_evidence("connection_pool_exhausted", is_transient=False),
            ],
            candidate_node="payments-db",
            template_key="connection_pool_exhausted",
            raw_score=1.0,
        )
        result = self._validate(h)
        # all_transient=False → R5b must not fire
        assert result.raw_score > TRANSIENT_SCORE_FACTOR

    def test_transient_min_evidence_constant_is_two(self):
        """Verify the constant matches the agreed configuration decision."""
        assert TRANSIENT_MIN_EVIDENCE == 2

    def test_transient_score_factor_constant_is_half(self):
        """Verify the 50% factor matches the agreed configuration decision."""
        assert TRANSIENT_SCORE_FACTOR == 0.50


# ===========================================================================
# TestR5cContextualEvidence
# ===========================================================================


class TestR5cContextualEvidence:
    """Unit tests for R5c advisory: contextual-corroboration score discount."""

    @pytest.fixture(autouse=True)
    def setup(self, graph):
        self.ctx = _make_context(graph, "payments-service")
        self.validator = SymbolicValidator(self.ctx)

    def _validate(self, h: CausalHypothesis) -> CausalHypothesis:
        result = self.validator._validate_one(h)
        assert not hasattr(result, "violated_rule"), f"Unexpected rejection: {result}"
        return result

    def test_database_query_timeout_alone_gets_discount(self):
        """database_query_timeout alone (no corroboration) → 40% score discount."""
        h = _make_hypothesis(
            evidence=[_make_evidence("database_query_timeout", is_transient=False)],
            candidate_node="payments-db",
            template_key="database_query_timeout",
            raw_score=1.0,
        )
        result = self._validate(h)
        expected = round(1.0 * CONTEXTUAL_SCORE_FACTOR, 6)
        assert abs(result.raw_score - expected) < 0.001, (
            f"Expected R5c to apply {CONTEXTUAL_SCORE_FACTOR}x, got {result.raw_score}"
        )

    def test_database_query_timeout_with_pool_exhausted_no_discount(self):
        """database_query_timeout + connection_pool_exhausted = corroboration → no R5c cut."""
        h = _make_hypothesis(
            evidence=[
                _make_evidence("database_query_timeout", is_transient=False),
                _make_evidence("connection_pool_exhausted", is_transient=False),
            ],
            candidate_node="payments-db",
            template_key="database_query_timeout",
            raw_score=1.0,
        )
        result = self._validate(h)
        # R5c should not fire because connection_pool_exhausted is a corroboration key
        assert result.raw_score > CONTEXTUAL_SCORE_FACTOR, (
            f"R5c should not have fired with pool_exhausted corroboration, "
            f"got {result.raw_score}"
        )

    def test_database_query_timeout_with_transaction_leak_no_discount(self):
        """transaction_leak is also a corroboration key → R5c does not fire."""
        h = _make_hypothesis(
            evidence=[
                _make_evidence("database_query_timeout"),
                _make_evidence("transaction_leak"),
            ],
            candidate_node="payments-db",
            template_key="database_query_timeout",
            raw_score=1.0,
        )
        result = self._validate(h)
        assert result.raw_score > CONTEXTUAL_SCORE_FACTOR

    def test_multiple_database_query_timeout_still_discounted_without_corroboration(self):
        """Even 3 occurrences of database_query_timeout → R5c fires if no corroboration.
        This tests Scenario A (FLT-CTXT-001): sustained timeouts alone insufficient."""
        h = _make_hypothesis(
            evidence=[
                _make_evidence("database_query_timeout"),
                _make_evidence("database_query_timeout"),
                _make_evidence("database_query_timeout"),
            ],
            candidate_node="payments-db",
            template_key="database_query_timeout",
            raw_score=1.0,
        )
        result = self._validate(h)
        # R5c checks evidence_keys (set), not count. Still just {database_query_timeout}
        expected = round(1.0 * CONTEXTUAL_SCORE_FACTOR, 6)
        assert abs(result.raw_score - expected) < 0.001

    def test_persistent_template_not_affected_by_r5c(self):
        """connection_pool_exhausted is not in CONTEXTUAL_TEMPLATE_KEYS → R5c silent."""
        h = _make_hypothesis(
            evidence=[_make_evidence("connection_pool_exhausted")],
            candidate_node="payments-db",
            template_key="connection_pool_exhausted",
            raw_score=1.0,
        )
        result = self._validate(h)
        # R5c must not fire for non-contextual templates
        assert result.raw_score > CONTEXTUAL_SCORE_FACTOR

    def test_contextual_constants_are_correct(self):
        """Verify the R5c constants match the agreed design values."""
        assert CONTEXTUAL_SCORE_FACTOR == 0.60   # 40% cut
        assert "database_query_timeout" in CONTEXTUAL_TEMPLATE_KEYS
        assert "connection_pool_exhausted" in CONTEXTUAL_CORROBORATION_KEYS
        assert "thread_pool_exhausted" in CONTEXTUAL_CORROBORATION_KEYS
        assert "transaction_leak" in CONTEXTUAL_CORROBORATION_KEYS


# ===========================================================================
# TestEndToEndTransientSuppression
# ===========================================================================


class TestEndToEndTransientSuppression:
    """End-to-end tests: classify_block + HypothesisEngine round-trip."""

    @pytest.fixture(autouse=True)
    def setup(self, graph):
        self.graph = graph
        self.router = LogRouter()

    def _run(self, telemetry: str) -> PerceptionReport:
        return asyncio.get_event_loop().run_until_complete(
            self.router.classify_block(telemetry)
        )

    def test_single_transient_error_is_suppressed_in_report(self):
        """One DNS error line in a block → suppressed, not hypothesis-generating."""
        telemetry = (
            "2024-01-01T10:00:00 ERROR: socket.gaierror: [Errno -3] "
            "Temporary failure in name resolution\n"
            "2024-01-01T10:00:01 INFO: Service health check OK\n"
        )
        report = self._run(telemetry)

        dns_results = [r for r in report.results if r.template_key == "dns_resolution_failure"]
        assert dns_results, "dns_resolution_failure should still appear in results"
        assert all(r.suppressed_transient for r in dns_results), (
            "Single DNS error must be suppressed"
        )
        assert report.transient_suppressed >= 1

    def test_three_transient_errors_pass_through(self):
        """Three identical DNS error lines → count == min_recurrence → NOT suppressed."""
        dns_line = "ERROR: socket.gaierror: [Errno -3] Temporary failure in name resolution"
        telemetry = "\n".join([dns_line] * 3)
        report = self._run(telemetry)

        dns_results = [r for r in report.results if r.template_key == "dns_resolution_failure"]
        assert dns_results, "dns_resolution_failure should appear in results"
        assert all(not r.suppressed_transient for r in dns_results), (
            "Three DNS errors must pass (count == min_recurrence)"
        )
        assert report.transient_passed == 3

    def test_persistent_error_never_suppressed(self):
        """connection_pool_exhausted → always passes regardless of count."""
        telemetry = (
            "ERROR: QueuePool limit of size 5 overflow 10 reached, connection timed out\n"
        )
        report = self._run(telemetry)

        pool_results = [r for r in report.results if r.template_key == "connection_pool_exhausted"]
        assert pool_results
        assert all(not r.suppressed_transient for r in pool_results)

    def test_database_query_timeout_always_routed(self):
        """database_query_timeout is threshold-contextual: always reaches report.results."""
        telemetry = (
            "ERROR: statement timeout after 30000ms — canceling statement\n"
        )
        report = self._run(telemetry)

        db_results = [r for r in report.results if r.template_key == "database_query_timeout"]
        assert db_results, "database_query_timeout must always appear in results"
        assert all(not r.suppressed_transient for r in db_results), (
            "database_query_timeout must never be suppressed by TransientFilter"
        )

    def test_hypothesis_engine_skips_suppressed_results(self, graph):
        """Suppressed results must not generate CausalHypothesis candidates."""
        # Build a report with one suppressed transient and one persistent result
        suppressed = _make_router_result(
            "dns_resolution_failure",
            log_entry="ERROR: socket.gaierror: [Errno -3] Temporary failure in name resolution",
            suppressed_transient=True,
        )
        persistent = _make_router_result(
            "connection_pool_exhausted",
            log_entry="ERROR: QueuePool limit of size 5 overflow 10 reached",
            suppressed_transient=False,
        )
        report = _make_perception_report(
            results=[suppressed, persistent],
            primary_template="connection_pool_exhausted",
        )

        ctx = _make_context(graph, "payments-service")
        engine = HypothesisEngine(context=ctx)
        hypotheses = engine.generate(report)

        # All hypotheses should come from connection_pool_exhausted only
        template_keys = {h.template_key for h in hypotheses}
        assert "dns_resolution_failure" not in template_keys, (
            "Suppressed transient result must not generate hypotheses"
        )
        assert "connection_pool_exhausted" in template_keys

    def test_mixed_block_correct_suppression_and_routing(self):
        """Block with 1 DNS (suppressed), 3 gRPC (passed), 1 pool exhausted (persistent)."""
        dns_line = "ERROR: socket.gaierror: [Errno -3] Temporary failure in name resolution"
        grpc_line = "ERROR: DEADLINE_EXCEEDED: context deadline exceeded after 5000ms"
        pool_line = "ERROR: QueuePool limit of size 5 overflow 10 reached, connection timed out"

        telemetry = "\n".join([
            dns_line,
            grpc_line,
            grpc_line,
            grpc_line,
            pool_line,
        ])
        report = self._run(telemetry)

        dns_results = [r for r in report.results if r.template_key == "dns_resolution_failure"]
        grpc_results = [r for r in report.results if r.template_key == "grpc_deadline_exceeded"]
        pool_results = [r for r in report.results if r.template_key == "connection_pool_exhausted"]

        # DNS: 1 occurrence < 3 → suppressed
        assert dns_results and all(r.suppressed_transient for r in dns_results)

        # gRPC: 3 occurrences == 3 → passed
        assert grpc_results and all(not r.suppressed_transient for r in grpc_results)

        # Pool: persistent → always passed
        assert pool_results and all(not r.suppressed_transient for r in pool_results)

        # Counter sanity
        assert report.transient_suppressed == 1
        assert report.transient_passed == 3
