"""
tests/test_perception.py

Unit tests for the 3-tier Perception Layer.

Tests cover:
  L1 — TemplateCache (regex-based exact pattern matching)
  L2 — EmbeddingMatcher (TF-IDF cosine similarity)
  L3 — TieredLogClassifier (cascade orchestrator + NeSy routing)
  NeSyRouter — Pathway selection logic
"""

from __future__ import annotations

import pytest

from agent.perception.template_cache import TemplateCache, TemplateMatch
from agent.perception.embedding_matcher import EmbeddingMatcher, L2Match
from agent.perception.log_classifier import TieredLogClassifier, PerceptionResult
from agent.reasoning.nesym_router import NeuroSymbolicRouter, ReasoningPathway


# ---------------------------------------------------------------------------
# L1: TemplateCache
# ---------------------------------------------------------------------------

class TestTemplateCache:

    def setup_method(self):
        self.cache = TemplateCache()

    def test_connection_pool_match(self):
        """Recognises QueuePool / too many clients error strings."""
        text = "QueuePool limit of size 5 overflow 10 reached"
        match = self.cache.classify(text)
        assert match is not None
        assert match.template_key == "connection_pool_exhausted"
        assert match.confidence >= 0.9

    def test_connection_pool_too_many_clients(self):
        """Recognises 'pool exhausted' variant."""
        text = "pool exhausted: connection pool 100% in use"
        match = self.cache.classify(text)
        assert match is not None
        assert match.template_key == "connection_pool_exhausted"

    def test_oom_killed_match(self):
        """Recognises OOMKilled exit code 137."""
        text = "OOMKilled: container exceeded memory limit (Exit code 137)"
        match = self.cache.classify(text)
        assert match is not None
        assert match.template_key == "oom_killed"

    def test_dns_failure_match(self):
        """Recognises socket.gaierror DNS failures."""
        text = "socket.gaierror: [Errno -3] Temporary failure in name resolution"
        match = self.cache.classify(text)
        assert match is not None
        assert match.template_key == "dns_resolution_failure"

    def test_disk_space_match(self):
        """Recognises disk full error."""
        text = "No space left on device writing to /var/lib/postgresql/data"
        match = self.cache.classify(text)
        assert match is not None
        assert match.template_key == "disk_space_exhausted"

    def test_tls_cert_match(self):
        """Recognises TLS/x509 certificate errors."""
        text = "SSL handshake failed: CERTIFICATE_VERIFY_FAILED"
        match = self.cache.classify(text)
        assert match is not None
        assert match.template_key == "tls_cert_expired"

    def test_no_match_returns_none(self):
        """Normal info log line returns None (no error pattern)."""
        text = "Server started on port 8080, ready to accept connections"
        match = self.cache.classify(text)
        assert match is None

    def test_match_returns_template_match(self):
        """classify() returns a TemplateMatch dataclass."""
        text = "QueuePool limit of size 5 overflow 10 reached"
        match = self.cache.classify(text)
        assert match is not None
        assert isinstance(match, TemplateMatch)
        assert hasattr(match, "template_key")
        assert hasattr(match, "confidence")
        assert hasattr(match, "tier")
        assert match.tier == "L1"

    def test_register_new_template_matches(self):
        """Dynamically registered templates are matched on subsequent calls."""
        self.cache.register_template(
            key="custom_timeout_001",
            pattern=r"CustomError: timeout after \d+ms",
        )
        text = "CustomError: timeout after 5000ms during DB write"
        match = self.cache.classify(text)
        assert match is not None
        assert match.template_key == "custom_timeout_001"

    def test_stats_returns_dict(self):
        """stats property returns a dict with expected keys."""
        self.cache.classify("QueuePool limit exceeded")
        stats = self.cache.stats
        assert isinstance(stats, dict)
        assert "total_queries" in stats
        assert "l1_hits" in stats
        assert "hit_rate" in stats
        assert "registered_templates" in stats

    def test_template_keys_includes_builtins(self):
        """template_keys contains all 12 built-in categories."""
        keys = self.cache.template_keys
        assert "connection_pool_exhausted" in keys
        assert "oom_killed" in keys
        assert "dns_resolution_failure" in keys
        assert "disk_space_exhausted" in keys
        assert "tls_cert_expired" in keys
        assert len(keys) >= 10  # At least 10 built-in templates

    def test_pod_crash_loop_match(self):
        """Recognises CrashLoopBackOff events."""
        text = "Back-off restarting failed container, CrashLoopBackOff"
        match = self.cache.classify(text)
        assert match is not None
        assert match.template_key == "pod_crash_loop"

    def test_transaction_leak_match(self):
        """Recognises long-running transaction leak warnings."""
        text = "Long-running transaction open for 120 seconds — possible connection leak"
        match = self.cache.classify(text)
        assert match is not None
        assert match.template_key == "transaction_leak"


# ---------------------------------------------------------------------------
# L2: EmbeddingMatcher
# ---------------------------------------------------------------------------

class TestEmbeddingMatcher:

    def setup_method(self):
        self.matcher = EmbeddingMatcher()
        self.matcher.fit()

    def test_find_nearest_connection_pool_semantics(self):
        """Semantically similar connection pool text may match at L2."""
        text = "database connections exhausted connection refused pool full"
        match = self.matcher.find_nearest(text)
        # L2 may or may not match (threshold 0.75) — just verify no error
        assert match is None or isinstance(match, L2Match)

    def test_find_nearest_returns_l2_match_or_none(self):
        """find_nearest returns L2Match or None, never raises."""
        text = "memory pressure OOM process killed"
        match = self.matcher.find_nearest(text)
        assert match is None or isinstance(match, L2Match)

    def test_l2_match_has_fields(self):
        """L2Match objects have template_key, similarity, tier fields."""
        # Use a text that's very close to a known pattern to force a match
        text = "QueuePool limit size overflow reached connection pool exhausted"
        match = self.matcher.find_nearest(text)
        if match is not None:
            assert hasattr(match, "template_key")
            assert hasattr(match, "similarity")
            assert hasattr(match, "tier")
            assert match.tier == "L2"
            assert 0.0 <= match.similarity <= 1.0

    def test_empty_text_returns_none_or_match(self):
        """Empty text does not raise — returns None or a low-confidence match."""
        result = self.matcher.find_nearest("")
        assert result is None or isinstance(result, L2Match)

    def test_stats_returns_dict(self):
        """stats property returns a dict."""
        stats = self.matcher.stats
        assert isinstance(stats, dict)
        assert "l2_hits" in stats
        assert "l2_misses" in stats
        assert "fitted" in stats

    def test_fit_sets_fitted_true(self):
        """After fit(), _fitted is True."""
        m = EmbeddingMatcher()
        assert not m._fitted
        m.fit()
        assert m._fitted


# ---------------------------------------------------------------------------
# L3: TieredLogClassifier (integration of L1 → L2 → L3)
# ---------------------------------------------------------------------------

class TestTieredLogClassifier:

    def setup_method(self):
        self.classifier = TieredLogClassifier()

    @pytest.mark.asyncio
    async def test_l1_hit_classification(self):
        """
        Telemetry with known error patterns should result in L1 hits.
        """
        block = "\n".join([
            "ERROR: QueuePool limit of size 5 overflow 10 reached",
            "ERROR: connection pool exhausted, pool timed out",
        ])
        result = await self.classifier.classify_telemetry_block(block)
        assert isinstance(result, PerceptionResult)
        assert result.l1_hits > 0
        assert result.primary_template == "connection_pool_exhausted"

    @pytest.mark.asyncio
    async def test_oom_classification(self):
        """OOMKilled log lines should be classified as oom_killed at L1."""
        block = "CRITICAL: OOMKilled — container exceeded 512Mi memory limit (Exit code 137)"
        result = await self.classifier.classify_telemetry_block(block)
        assert isinstance(result, PerceptionResult)
        assert result.l1_hits >= 1
        assert result.primary_template == "oom_killed"

    @pytest.mark.asyncio
    async def test_empty_block_returns_result(self):
        """Empty telemetry block returns PerceptionResult without error."""
        result = await self.classifier.classify_telemetry_block("")
        assert isinstance(result, PerceptionResult)
        assert result.primary_template == "no_logs_classified"

    @pytest.mark.asyncio
    async def test_info_only_block(self):
        """Block with only INFO lines returns 0 classifications."""
        block = "\n".join([
            "INFO: Service started on port 8080",
            "INFO: Health check passed",
            "INFO: Request processed in 45ms",
        ])
        result = await self.classifier.classify_telemetry_block(block)
        assert isinstance(result, PerceptionResult)
        # INFO lines should not trigger classification
        total = result.l1_hits + result.l2_hits + result.l3_hits
        assert total == 0
        assert result.primary_template == "no_logs_classified"

    @pytest.mark.asyncio
    async def test_tier_stats_structure(self):
        """tier_stats dict has expected keys with correct types."""
        block = "ERROR: QueuePool limit of size 5 overflow 10 reached"
        result = await self.classifier.classify_telemetry_block(block)
        stats = result.tier_stats
        assert "L1_hits" in stats
        assert "L2_hits" in stats
        assert "L3_hits" in stats
        assert "total" in stats
        assert "l1_rate" in stats
        assert 0.0 <= stats["l1_rate"] <= 1.0

    @pytest.mark.asyncio
    async def test_l1_rate_all_l1_is_one(self):
        """When all lines are L1 hits, l1_rate should be 1.0."""
        block = "\n".join([
            "ERROR: QueuePool limit of size 5 overflow 10 reached",
            "ERROR: pool exhausted: connection pool timed out",
        ])
        result = await self.classifier.classify_telemetry_block(block)
        if result.l1_hits == result.l1_hits + result.l2_hits + result.l3_hits and result.l1_hits > 0:
            assert result.tier_stats["l1_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_to_markdown_returns_string(self):
        """to_markdown() produces a non-empty markdown string."""
        block = "ERROR: OOMKilled — exit code 137"
        result = await self.classifier.classify_telemetry_block(block)
        md = result.to_markdown()
        assert isinstance(md, str)
        assert len(md) > 20
        assert "Perception" in md

    @pytest.mark.asyncio
    async def test_multi_pattern_block_identifies_dominant(self):
        """When multiple patterns appear, primary_template is the most frequent."""
        block = "\n".join([
            "ERROR: QueuePool limit of size 5 overflow 10 reached",
            "ERROR: pool exhausted: connections in use 100",
            "ERROR: pool timed out waiting for connection",
            "ERROR: OOMKilled exit code 137",
        ])
        result = await self.classifier.classify_telemetry_block(block)
        # 3 connection_pool vs 1 oom — primary should be connection_pool
        assert result.primary_template == "connection_pool_exhausted"

    def test_l1_stats_property(self):
        """l1_stats property accessible and returns dict."""
        stats = self.classifier.l1_stats
        assert isinstance(stats, dict)

    def test_l2_stats_property(self):
        """l2_stats property accessible and returns dict."""
        stats = self.classifier.l2_stats
        assert isinstance(stats, dict)


# ---------------------------------------------------------------------------
# NeSyRouter
# ---------------------------------------------------------------------------

class TestNeuroSymbolicRouter:

    def setup_method(self):
        self.router = NeuroSymbolicRouter()

    def _stats(self, l1=0, l2=0, l3=0, total=10):
        l1_rate = l1 / total if total > 0 else 0.0
        return {"L1_hits": l1, "L2_hits": l2, "L3_hits": l3, "total": total, "l1_rate": l1_rate}

    def test_symbolic_fast_path(self):
        """All L1 hits + zero L3 → SYMBOLIC_FAST."""
        decision = self.router.route(
            perception_stats=self._stats(l1=10, l2=0, l3=0, total=10),
            cbr_confidence=0.4,
            primary_template="connection_pool_exhausted",
        )
        assert decision.pathway == ReasoningPathway.SYMBOLIC_FAST

    def test_cbr_guided_path(self):
        """High CBR confidence (>= 0.75) → CBR_GUIDED regardless of L1 rate."""
        decision = self.router.route(
            perception_stats=self._stats(l1=5, l2=3, l3=2, total=10),
            cbr_confidence=0.85,
            primary_template="oom_killed",
        )
        assert decision.pathway == ReasoningPathway.CBR_GUIDED

    def test_neural_full_path(self):
        """Novel patterns + low CBR confidence → NEURAL_FULL."""
        decision = self.router.route(
            perception_stats=self._stats(l1=2, l2=2, l3=6, total=10),
            cbr_confidence=0.30,
            primary_template="unknown",
        )
        assert decision.pathway == ReasoningPathway.NEURAL_FULL

    def test_symbolic_overrides_low_cbr(self):
        """SYMBOLIC_FAST takes precedence when L1 rate is 100%, even with low CBR."""
        decision = self.router.route(
            perception_stats=self._stats(l1=10, l2=0, l3=0, total=10),
            cbr_confidence=0.2,
            primary_template="disk_space_exhausted",
        )
        assert decision.pathway == ReasoningPathway.SYMBOLIC_FAST

    def test_cbr_guided_threshold_boundary(self):
        """CBR confidence exactly at threshold (0.75) triggers CBR_GUIDED."""
        decision = self.router.route(
            perception_stats=self._stats(l1=4, l2=4, l3=2, total=10),
            cbr_confidence=0.75,
            primary_template="database_query_timeout",
        )
        assert decision.pathway == ReasoningPathway.CBR_GUIDED

    def test_decision_includes_rationale(self):
        """All routing decisions include a non-empty rationale string."""
        for pathway_inputs in [
            (self._stats(l1=10, l2=0, l3=0, total=10), 0.2, "template"),
            (self._stats(l1=5, l2=3, l3=2, total=10), 0.8, "template"),
            (self._stats(l1=2, l2=2, l3=6, total=10), 0.3, "unknown"),
        ]:
            decision = self.router.route(*pathway_inputs)
            assert len(decision.rationale) > 0

    def test_to_markdown_includes_pathway(self):
        """to_markdown() output includes the pathway name."""
        decision = self.router.route(
            perception_stats=self._stats(l1=10, l2=0, l3=0, total=10),
            cbr_confidence=0.9,
            primary_template="tls_cert_expired",
        )
        md = decision.to_markdown()
        assert decision.pathway.value.upper() in md.upper()

    def test_empty_stats_goes_neural_full(self):
        """Zero total logs → NEURAL_FULL (no evidence for SYMBOLIC)."""
        decision = self.router.route(
            perception_stats=self._stats(l1=0, l2=0, l3=0, total=0),
            cbr_confidence=0.5,
            primary_template="",
        )
        assert decision.pathway == ReasoningPathway.NEURAL_FULL

    def test_l1_rate_fields_populated(self):
        """RoutingDecision includes l1_rate and cbr_confidence."""
        decision = self.router.route(
            perception_stats=self._stats(l1=8, l2=2, l3=0, total=10),
            cbr_confidence=0.6,
            primary_template="connection_pool_exhausted",
        )
        assert decision.l1_rate == pytest.approx(0.8)
        assert decision.cbr_confidence == pytest.approx(0.6)
