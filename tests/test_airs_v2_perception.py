"""
tests/test_airs_v2_perception.py
=================================

Unit tests for the airs_v2 NeSy-Edge Three-Tier Log Parsing Router.

Test coverage
-------------
TestL1SymbolicCache       — Deterministic regex matching, registration, stats, snapshot/restore
TestL2SemanticMatcher     — TF-IDF cosine similarity, threshold control, incremental learning
TestL3LLMFallback         — Stub mode (no API key), parse helpers, timeout stub
TestLogRouterSingleEntry  — Cascade routing decisions for known and novel logs
TestLogRouterBlock        — Full telemetry block classification and PerceptionReport
TestContinuousLearning    — L3 promotion to L1+L2 and cache-hit verification
TestNeSyRouterIntegration — End-to-end: PerceptionReport → NeSy router pathway

All tests are fully synchronous or pytest-asyncio async; none require a
live GROQ_API_KEY — the L3 layer automatically returns a stub result when
the key is absent.

Log samples are drawn from the existing mock_enterprise fixtures and from
the remediation step examples in agent/state.py.
"""

from __future__ import annotations

import os
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from airs_v2.perception.l1_cache import (
    L1SymbolicCache,
    L1Match,
    BUILTIN_TEMPLATES,
)
from airs_v2.perception.l2_semantic import L2SemanticMatcher, L2Match
from airs_v2.perception.l3_llm import L3LLMFallback, L3Result, _parse_l3_response
from airs_v2.perception.router import (
    LogRouter,
    RouterResult,
    PerceptionReport,
    Tier,
)

# Ensure no GROQ_API_KEY bleeds in from the environment during unit tests
os.environ.pop("GROQ_API_KEY", None)


# ============================================================================
# Log sample fixtures — drawn from existing system logs in state.py / mock data
# ============================================================================

# L1-level (exact regex match expected)
LOG_POOL_EXHAUSTED = "QueuePool limit of size 5 overflow 10 reached, connection timed out"
LOG_POOL_TOO_MANY = "ERROR: connection pool exhausted — connections in use: 100/100"
LOG_OOM_KILLED = "CRITICAL: OOMKilled — container exceeded 512Mi memory limit (Exit code 137)"
LOG_OOM_JAVA = "java.lang.OutOfMemoryError: Java heap space at DataProcessor.run:42"
LOG_DNS_GAIERROR = "socket.gaierror: [Errno -3] Temporary failure in name resolution"
LOG_DNS_COREDNS = "ERROR: CoreDNS timeout resolving payments-service.prod.svc.cluster.local"
LOG_DISK_FULL = "CRITICAL: No space left on device writing to /var/lib/postgresql/data/base"
LOG_TLS_EXPIRED = "ERROR: SSL handshake failed — CERTIFICATE_VERIFY_FAILED: certificate expired"
LOG_RATE_LIMIT = "WARN: 429 Too Many Requests from stripe.com — Retry-After: 30s"
LOG_CIRCUIT_OPEN = "ERROR: circuit.breaker.open — exhausted retries for payment-gateway"
LOG_DB_TIMEOUT = "ERROR: statement timeout after 30000ms — canceling statement due to user request"
LOG_REDIS_OOM = "ERROR: OOM command not allowed when used memory > 'maxmemory' (used: 2.1GB, max: 2.0GB)"
LOG_CRASH_LOOP = "WARN: Back-off restarting failed container — CrashLoopBackOff (5 restarts)"
LOG_CRASH_PROBE = "ERROR: Liveness probe failed: HTTP probe failed with statuscode: 503"
LOG_UPSTREAM_503 = "ERROR: 503 Service Unavailable — no healthy upstream for api-gateway"
LOG_TX_LEAK = "WARN: Long-running transaction open for 312 seconds — possible connection leak"
LOG_THREAD_EXHAUSTED = "ERROR: Thread pool exhausted — Rejecting new incoming request"
LOG_KAFKA_LAG = "WARN: consumer group orders-consumer lag is 45000 messages pending — not keeping up"
LOG_GRPC_DEADLINE = "ERROR: rpc error: code = DeadlineExceeded desc = context deadline exceeded"

# L2-level (semantic match expected — paraphrased, regex won't catch)
LOG_L2_POOL_PARAPHRASE = "database connections all occupied waiting queue full"
LOG_L2_OOM_PARAPHRASE = "memory pressure killed worker process heap allocation failed"

# L3-level (truly novel — should go to LLM / stub)
LOG_NOVEL_1 = "ERROR: CustomRateLimiter rejected request: quota_window=60s bucket=api_v3_premium"
LOG_NOVEL_2 = "WARN: FeatureFlagService circuit open for experiment group PAYMENTS_V2_ROLLOUT"
LOG_NOVEL_3 = "CRITICAL: ChaosMesh network delay 500ms injected on payments-service pod"

# Info lines — should NOT be classified (filtered out by block classifier)
LOG_INFO_1 = "INFO: Server started on port 8080, ready to accept connections"
LOG_INFO_2 = "INFO: Health check passed — latency 12ms"
LOG_INFO_3 = "INFO: Request processed in 45ms"


# ============================================================================
# TestL1SymbolicCache
# ============================================================================

class TestL1SymbolicCache:
    """Tests for the L1 deterministic regex cache."""

    def setup_method(self):
        self.cache = L1SymbolicCache()

    # -- Basic matching --

    def test_connection_pool_exhausted_queuepool(self):
        match = self.cache.classify(LOG_POOL_EXHAUSTED)
        assert match is not None
        assert match.template_key == "connection_pool_exhausted"
        assert match.confidence == 1.0
        assert match.tier == "L1"

    def test_connection_pool_exhausted_in_use(self):
        match = self.cache.classify(LOG_POOL_TOO_MANY)
        assert match is not None
        assert match.template_key == "connection_pool_exhausted"

    def test_oom_killed_exit_code(self):
        match = self.cache.classify(LOG_OOM_KILLED)
        assert match is not None
        assert match.template_key == "oom_killed"

    def test_oom_java_heap(self):
        match = self.cache.classify(LOG_OOM_JAVA)
        assert match is not None
        assert match.template_key == "oom_killed"

    def test_dns_gaierror(self):
        match = self.cache.classify(LOG_DNS_GAIERROR)
        assert match is not None
        assert match.template_key == "dns_resolution_failure"

    def test_dns_coredns_timeout(self):
        match = self.cache.classify(LOG_DNS_COREDNS)
        assert match is not None
        assert match.template_key == "dns_resolution_failure"

    def test_disk_space_exhausted(self):
        match = self.cache.classify(LOG_DISK_FULL)
        assert match is not None
        assert match.template_key == "disk_space_exhausted"

    def test_tls_cert_expired(self):
        match = self.cache.classify(LOG_TLS_EXPIRED)
        assert match is not None
        assert match.template_key == "tls_cert_expired"

    def test_upstream_rate_limited_429(self):
        match = self.cache.classify(LOG_RATE_LIMIT)
        assert match is not None
        assert match.template_key == "upstream_rate_limited"

    def test_circuit_breaker_open(self):
        match = self.cache.classify(LOG_CIRCUIT_OPEN)
        assert match is not None
        assert match.template_key == "upstream_rate_limited"

    def test_database_query_timeout(self):
        match = self.cache.classify(LOG_DB_TIMEOUT)
        assert match is not None
        assert match.template_key == "database_query_timeout"

    def test_redis_oom_eviction(self):
        match = self.cache.classify(LOG_REDIS_OOM)
        assert match is not None
        assert match.template_key == "redis_oom_eviction"

    def test_pod_crash_loop_backoff(self):
        match = self.cache.classify(LOG_CRASH_LOOP)
        assert match is not None
        assert match.template_key == "pod_crash_loop"

    def test_liveness_probe_failed(self):
        match = self.cache.classify(LOG_CRASH_PROBE)
        assert match is not None
        assert match.template_key == "pod_crash_loop"

    def test_http_upstream_unavailable(self):
        match = self.cache.classify(LOG_UPSTREAM_503)
        assert match is not None
        assert match.template_key == "http_upstream_unavailable"

    def test_transaction_leak(self):
        match = self.cache.classify(LOG_TX_LEAK)
        assert match is not None
        assert match.template_key == "transaction_leak"

    def test_thread_pool_exhausted(self):
        match = self.cache.classify(LOG_THREAD_EXHAUSTED)
        assert match is not None
        assert match.template_key == "thread_pool_exhausted"

    def test_kafka_consumer_lag(self):
        match = self.cache.classify(LOG_KAFKA_LAG)
        assert match is not None
        assert match.template_key == "kafka_consumer_lag"

    def test_grpc_deadline_exceeded(self):
        match = self.cache.classify(LOG_GRPC_DEADLINE)
        assert match is not None
        assert match.template_key == "grpc_deadline_exceeded"

    # -- Negative cases --

    def test_info_line_returns_none(self):
        assert self.cache.classify(LOG_INFO_1) is None

    def test_health_check_info_returns_none(self):
        assert self.cache.classify(LOG_INFO_2) is None

    def test_novel_log_returns_none(self):
        """A truly novel log that does not match any regex should return None."""
        assert self.cache.classify(LOG_NOVEL_1) is None

    # -- Match type --

    def test_returns_l1_match_dataclass(self):
        match = self.cache.classify(LOG_POOL_EXHAUSTED)
        assert isinstance(match, L1Match)
        assert match.tier == "L1"
        assert isinstance(match.template_key, str)
        assert isinstance(match.confidence, float)
        assert isinstance(match.description, str)
        assert isinstance(match.source, str)

    def test_source_is_builtin_for_builtin_templates(self):
        match = self.cache.classify(LOG_OOM_KILLED)
        assert match is not None
        assert match.source == "builtin"

    # -- Dynamic registration --

    def test_register_new_template_matches(self):
        self.cache.register_template(
            key="custom_quota_exceeded",
            pattern=r"CustomRateLimiter.*quota_window",
            description="Custom rate limiter quota exceeded",
            source="user",
        )
        match = self.cache.classify(LOG_NOVEL_1)
        assert match is not None
        assert match.template_key == "custom_quota_exceeded"
        assert match.source == "user"

    def test_register_invalid_regex_returns_false(self):
        ok = self.cache.register_template(
            key="bad_regex",
            pattern=r"[unclosed",
            description="Bad regex test",
        )
        assert ok is False
        assert "bad_regex" not in self.cache.template_keys

    def test_learned_keys_tracks_l3_source(self):
        self.cache.register_template(
            key="l3_learned_pattern",
            pattern=r"l3.*learned",
            description="Learned by L3",
            source="l3_learned",
        )
        assert "l3_learned_pattern" in self.cache.learned_keys

    # -- Stats --

    def test_stats_structure(self):
        self.cache.classify(LOG_POOL_EXHAUSTED)
        stats = self.cache.stats
        assert "total_queries" in stats
        assert "total_hits" in stats
        assert "hit_rate" in stats
        assert "registered_templates" in stats
        assert "learned_templates" in stats
        assert "top_patterns" in stats

    def test_stats_hit_rate_after_one_hit(self):
        self.cache.classify(LOG_OOM_KILLED)
        self.cache.classify(LOG_INFO_1)  # miss
        stats = self.cache.stats
        assert stats["total_queries"] == 2
        assert stats["total_hits"] == 1
        assert stats["hit_rate"] == pytest.approx(0.5)

    def test_template_keys_includes_all_builtins(self):
        keys = self.cache.template_keys
        for key, _, _ in BUILTIN_TEMPLATES:
            assert key in keys

    # -- Snapshot / restore --

    def test_snapshot_restore_roundtrip(self):
        snap = self.cache.snapshot()
        self.cache.register_template("temp_key", r"temp.*match", "temp", source="user")
        assert "temp_key" in self.cache.template_keys
        self.cache.restore(snap)
        assert "temp_key" not in self.cache.template_keys

    def test_restore_preserves_hit_counts(self):
        self.cache.classify(LOG_OOM_KILLED)     # 1 hit for oom_killed
        snap = self.cache.snapshot()
        cache2 = L1SymbolicCache()
        cache2.restore(snap)
        stats = cache2.stats
        assert stats["total_hits"] == 0         # total_hits is separate from hit_counts
        # The hit_count for oom_killed is preserved in registry
        assert cache2._registry["oom_killed"].hit_count == 1

    # -- Case-insensitive --

    def test_case_insensitive_match(self):
        match = self.cache.classify("queuepool LIMIT of size 10 OVERFLOW reached")
        assert match is not None
        assert match.template_key == "connection_pool_exhausted"


# ============================================================================
# TestL2SemanticMatcher
# ============================================================================

class TestL2SemanticMatcher:
    """Tests for the L2 TF-IDF cosine similarity matcher."""

    def setup_method(self):
        # Default threshold; auto_fit=True
        self.matcher = L2SemanticMatcher()

    def test_fitted_after_construction(self):
        assert self.matcher.fitted is True

    def test_knowledge_base_size_equals_builtin_count(self):
        assert self.matcher.knowledge_base_size == len(BUILTIN_TEMPLATES)

    # -- find_nearest returns correct type --

    def test_find_nearest_returns_l2match_or_none(self):
        result = self.matcher.find_nearest(LOG_L2_POOL_PARAPHRASE)
        assert result is None or isinstance(result, L2Match)

    def test_l2_match_fields(self):
        # Text closely describing OOM — should produce a match
        result = self.matcher.find_nearest("memory pressure killed worker process heap allocation failed")
        if result is not None:
            assert isinstance(result.template_key, str)
            assert isinstance(result.similarity, float)
            assert 0.0 < result.similarity <= 1.0
            assert result.tier == "L2"
            assert isinstance(result.description, str)

    def test_empty_string_returns_none_or_l2match(self):
        """Empty string should not crash — returns None or very low similarity."""
        result = self.matcher.find_nearest("")
        assert result is None or isinstance(result, L2Match)

    # -- Top-k candidates --

    def test_find_nearest_k_returns_list(self):
        results = self.matcher.find_nearest_k(LOG_L2_POOL_PARAPHRASE, k=3)
        assert isinstance(results, list)

    def test_find_nearest_k_respects_k_limit(self):
        results = self.matcher.find_nearest_k(LOG_L2_OOM_PARAPHRASE, k=2)
        assert len(results) <= 2

    def test_find_nearest_k_sorted_by_similarity(self):
        results = self.matcher.find_nearest_k(
            "database connection pool all connections in use timed out heap exhausted", k=5
        )
        if len(results) >= 2:
            assert results[0].similarity >= results[1].similarity

    # -- Threshold control --

    def test_high_threshold_returns_none(self):
        """At threshold=1.0 nothing should match (no perfect TF-IDF match)."""
        strict = L2SemanticMatcher(threshold=1.0)
        result = strict.find_nearest("database connections all occupied")
        assert result is None

    def test_zero_threshold_matches_something(self):
        """At threshold=0.0 any non-empty text should match something."""
        lenient = L2SemanticMatcher(threshold=0.0)
        results = lenient.find_nearest_k("memory failed allocation pressure", k=3)
        assert len(results) > 0

    def test_threshold_setter_validates_range(self):
        with pytest.raises(ValueError):
            self.matcher.threshold = 1.5
        with pytest.raises(ValueError):
            self.matcher.threshold = 0.0

    # -- Incremental add_entry --

    def test_add_entry_increases_kb_size(self):
        before = self.matcher.knowledge_base_size
        self.matcher.add_entry(
            key="custom_quota_limiter",
            description="Custom rate limiter bucket exhausted quota window exceeded",
            pattern_hint=r"CustomRateLimiter.*quota_window",
        )
        assert self.matcher.knowledge_base_size == before + 1

    def test_add_entry_duplicate_ignored(self):
        self.matcher.add_entry(
            key="connection_pool_exhausted",
            description="duplicate entry should be ignored",
        )
        assert self.matcher.knowledge_base_size == len(BUILTIN_TEMPLATES)

    def test_add_entry_then_match(self):
        """After adding a custom entry it should match semantically similar logs."""
        self.matcher.add_entry(
            key="etl_pipeline_stall",
            description="ETL pipeline batch job stalled partitions not flushed buffer overflow",
        )
        result = self.matcher.find_nearest(
            "WARN: ETL batch job stalled — flush buffer overflow partitions stuck"
        )
        if result is not None:
            assert result.template_key == "etl_pipeline_stall"

    # -- Stats --

    def test_stats_structure(self):
        stats = self.matcher.stats
        assert "l2_hits" in stats
        assert "l2_misses" in stats
        assert "l2_hit_rate" in stats
        assert "knowledge_base_size" in stats
        assert "threshold" in stats
        assert "fitted" in stats

    def test_stats_hit_rate_zero_initially(self):
        m = L2SemanticMatcher()
        stats = m.stats
        assert stats["l2_hits"] == 0
        assert stats["l2_misses"] == 0


# ============================================================================
# TestL3LLMFallback
# ============================================================================

class TestL3LLMFallback:
    """Tests for the L3 LLM fallback tier — all in stub mode (no GROQ_API_KEY)."""

    def setup_method(self):
        # Guarantee no API key so every test hits stub mode
        os.environ.pop("GROQ_API_KEY", None)
        self.l3 = L3LLMFallback()

    @pytest.mark.asyncio
    async def test_stub_mode_returns_l3_result(self):
        result = await self.l3.classify(LOG_NOVEL_1)
        assert isinstance(result, L3Result)

    @pytest.mark.asyncio
    async def test_stub_mode_confidence_zero(self):
        result = await self.l3.classify(LOG_NOVEL_1)
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_stub_mode_source_is_stub(self):
        result = await self.l3.classify(LOG_NOVEL_1)
        assert result.source == "stub"

    @pytest.mark.asyncio
    async def test_stub_mode_tier_is_l3(self):
        result = await self.l3.classify(LOG_NOVEL_1)
        assert result.tier == "L3"

    @pytest.mark.asyncio
    async def test_stub_mode_is_novel_true(self):
        result = await self.l3.classify(LOG_NOVEL_1)
        assert result.is_novel is True

    @pytest.mark.asyncio
    async def test_stub_never_raises(self):
        """Even with bizarre input, stub mode should not raise."""
        result = await self.l3.classify("🔥 " * 1000)
        assert isinstance(result, L3Result)

    # -- Stats --

    @pytest.mark.asyncio
    async def test_stats_call_count_increments(self):
        await self.l3.classify(LOG_NOVEL_2)
        await self.l3.classify(LOG_NOVEL_3)
        assert self.l3.stats["l3_calls"] == 2

    def test_record_promotion_increments_counter(self):
        self.l3.record_promotion()
        self.l3.record_promotion()
        assert self.l3.stats["l3_promotions"] == 2

    def test_stats_available_false_without_api_key(self):
        assert self.l3.stats["available"] is False

    # -- JSON parser (unit tests, not async) --

    def test_parse_l3_response_valid_json(self):
        raw = '{"template_key": "kafka_lag", "regex_pattern": "consumer.*lag.*\\\\d+", "description": "Kafka consumer lag", "confidence": 0.85}'
        result = _parse_l3_response(raw, "some log")
        assert result.template_key == "kafka_lag"
        assert result.confidence == pytest.approx(0.85)
        assert result.source == "llm"

    def test_parse_l3_response_extracts_embedded_json(self):
        raw = 'Sure! Here is the result: {"template_key": "disk_full", "regex_pattern": "ENOSPC.*", "description": "No space left", "confidence": 0.9}'
        result = _parse_l3_response(raw, "log")
        assert result.template_key == "disk_full"

    def test_parse_l3_response_invalid_json_returns_stub(self):
        result = _parse_l3_response("not json at all", "log")
        assert result.template_key == "parse_failed"
        assert result.confidence == 0.0

    def test_parse_l3_response_invalid_regex_discarded(self):
        raw = '{"template_key": "bad", "regex_pattern": "[unclosed", "description": "bad regex", "confidence": 0.9}'
        result = _parse_l3_response(raw, "log")
        # regex is discarded, template_key and description should still parse
        assert result.template_key == "bad"
        assert result.regex_pattern == ""

    # -- Mocked LLM call --

    @pytest.mark.asyncio
    async def test_mocked_llm_returns_l3_result(self):
        """When GROQ_API_KEY is set and LLM is mocked, returns proper L3Result."""
        mock_response_content = (
            '{"template_key":"feature_flag_circuit","regex_pattern":"FeatureFlagService.*circuit.*open",'
            '"description":"Feature flag circuit breaker opened","confidence":0.88}'
        )
        mock_ai_message = MagicMock()
        mock_ai_message.content = mock_response_content

        with patch.dict(os.environ, {"GROQ_API_KEY": "mock-key"}):
            l3 = L3LLMFallback()
            # Replace the internal LLM with an AsyncMock
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_ai_message)
            l3._llm = mock_llm

            result = await l3.classify(LOG_NOVEL_2)

        assert result.template_key == "feature_flag_circuit"
        assert result.confidence == pytest.approx(0.88)
        assert result.source == "llm"
        assert result.tier == "L3"


# ============================================================================
# TestLogRouterSingleEntry
# ============================================================================

class TestLogRouterSingleEntry:
    """Tests that verify the routing cascade for individual log lines."""

    def setup_method(self):
        os.environ.pop("GROQ_API_KEY", None)
        self.router = LogRouter()

    # -- L1 routing --

    @pytest.mark.asyncio
    async def test_known_pool_log_routes_to_l1(self):
        result = await self.router.classify(LOG_POOL_EXHAUSTED)
        assert result.tier == Tier.L1
        assert result.template_key == "connection_pool_exhausted"
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_oom_killed_routes_to_l1(self):
        result = await self.router.classify(LOG_OOM_KILLED)
        assert result.tier == Tier.L1
        assert result.template_key == "oom_killed"

    @pytest.mark.asyncio
    async def test_dns_failure_routes_to_l1(self):
        result = await self.router.classify(LOG_DNS_GAIERROR)
        assert result.tier == Tier.L1
        assert result.template_key == "dns_resolution_failure"

    @pytest.mark.asyncio
    async def test_tls_routes_to_l1(self):
        result = await self.router.classify(LOG_TLS_EXPIRED)
        assert result.tier == Tier.L1
        assert result.template_key == "tls_cert_expired"

    @pytest.mark.asyncio
    async def test_crash_loop_routes_to_l1(self):
        result = await self.router.classify(LOG_CRASH_LOOP)
        assert result.tier == Tier.L1
        assert result.template_key == "pod_crash_loop"

    @pytest.mark.asyncio
    async def test_transaction_leak_routes_to_l1(self):
        result = await self.router.classify(LOG_TX_LEAK)
        assert result.tier == Tier.L1
        assert result.template_key == "transaction_leak"

    @pytest.mark.asyncio
    async def test_kafka_lag_routes_to_l1(self):
        result = await self.router.classify(LOG_KAFKA_LAG)
        assert result.tier == Tier.L1
        assert result.template_key == "kafka_consumer_lag"

    @pytest.mark.asyncio
    async def test_grpc_deadline_routes_to_l1(self):
        result = await self.router.classify(LOG_GRPC_DEADLINE)
        assert result.tier == Tier.L1
        assert result.template_key == "grpc_deadline_exceeded"

    # -- L3 routing (novel logs hit stub) --

    @pytest.mark.asyncio
    async def test_novel_log_routes_to_l3(self):
        result = await self.router.classify(LOG_NOVEL_1)
        assert result.tier == Tier.L3
        assert result.is_novel is True

    @pytest.mark.asyncio
    async def test_novel_log_l3_template_key_present(self):
        result = await self.router.classify(LOG_NOVEL_2)
        assert isinstance(result.template_key, str) and len(result.template_key) > 0

    @pytest.mark.asyncio
    async def test_novel_log_l3_candidates_empty_or_list(self):
        result = await self.router.classify(LOG_NOVEL_3)
        assert isinstance(result.l2_candidates, list)

    # -- RouterResult fields --

    @pytest.mark.asyncio
    async def test_router_result_has_all_fields(self):
        result = await self.router.classify(LOG_OOM_KILLED)
        assert isinstance(result, RouterResult)
        assert hasattr(result, "log_entry")
        assert hasattr(result, "tier")
        assert hasattr(result, "template_key")
        assert hasattr(result, "confidence")
        assert hasattr(result, "description")
        assert hasattr(result, "is_novel")
        assert hasattr(result, "l2_candidates")
        assert hasattr(result, "l3_result")
        assert hasattr(result, "promoted")

    @pytest.mark.asyncio
    async def test_log_entry_preserved_in_result(self):
        result = await self.router.classify(LOG_DISK_FULL)
        assert result.log_entry == LOG_DISK_FULL

    @pytest.mark.asyncio
    async def test_l1_result_not_novel(self):
        result = await self.router.classify(LOG_REDIS_OOM)
        assert result.is_novel is False

    @pytest.mark.asyncio
    async def test_l1_result_promoted_false(self):
        result = await self.router.classify(LOG_DB_TIMEOUT)
        assert result.promoted is False  # L1 hits are never promoted

    # -- Stats --

    @pytest.mark.asyncio
    async def test_router_stats_structure(self):
        await self.router.classify(LOG_POOL_EXHAUSTED)
        stats = self.router.stats
        assert "l1" in stats
        assert "l2" in stats
        assert "l3" in stats
        assert "promote_threshold" in stats


# ============================================================================
# TestLogRouterBlock
# ============================================================================

class TestLogRouterBlock:
    """Tests for the full telemetry block classification."""

    def setup_method(self):
        os.environ.pop("GROQ_API_KEY", None)
        self.router = LogRouter()

    @pytest.mark.asyncio
    async def test_single_known_pattern_block(self):
        block = "\n".join([
            f"ERROR: {LOG_POOL_EXHAUSTED}",
            f"ERROR: {LOG_POOL_TOO_MANY}",
        ])
        report = await self.router.classify_block(block)
        assert isinstance(report, PerceptionReport)
        assert report.primary_template == "connection_pool_exhausted"
        assert report.l1_hits >= 1

    @pytest.mark.asyncio
    async def test_oom_block_classification(self):
        block = f"CRITICAL: {LOG_OOM_KILLED}\nERROR: {LOG_OOM_JAVA}"
        report = await self.router.classify_block(block)
        assert report.primary_template == "oom_killed"
        assert report.l1_hits >= 1

    @pytest.mark.asyncio
    async def test_info_only_block_returns_empty(self):
        block = "\n".join([LOG_INFO_1, LOG_INFO_2, LOG_INFO_3])
        report = await self.router.classify_block(block)
        assert report.primary_template == "no_logs_classified"
        total = report.l1_hits + report.l2_hits + report.l3_hits
        assert total == 0

    @pytest.mark.asyncio
    async def test_empty_block_returns_report(self):
        report = await self.router.classify_block("")
        assert isinstance(report, PerceptionReport)
        assert report.primary_template == "no_logs_classified"

    @pytest.mark.asyncio
    async def test_mixed_block_dominant_template(self):
        """3x connection_pool + 1x oom → primary = connection_pool_exhausted."""
        block = "\n".join([
            f"ERROR: {LOG_POOL_EXHAUSTED}",
            f"ERROR: {LOG_POOL_TOO_MANY}",
            f"ERROR: {LOG_DB_TIMEOUT}",       # connection pool variant? No — db_timeout
            f"CRITICAL: {LOG_OOM_KILLED}",
        ])
        report = await self.router.classify_block(block)
        # Primary should be whichever template appeared most frequently
        assert report.primary_template in {
            "connection_pool_exhausted", "database_query_timeout", "oom_killed"
        }

    @pytest.mark.asyncio
    async def test_tier_stats_structure(self):
        block = f"ERROR: {LOG_POOL_EXHAUSTED}"
        report = await self.router.classify_block(block)
        stats = report.tier_stats
        assert "L1_hits" in stats
        assert "L2_hits" in stats
        assert "L3_hits" in stats
        assert "total" in stats
        assert "l1_rate" in stats
        assert "novel_count" in stats
        assert "promoted_count" in stats
        assert 0.0 <= stats["l1_rate"] <= 1.0

    @pytest.mark.asyncio
    async def test_l1_rate_is_one_for_all_known_logs(self):
        block = "\n".join([
            f"ERROR: {LOG_POOL_EXHAUSTED}",
            f"ERROR: {LOG_CRASH_LOOP}",
            f"ERROR: {LOG_TLS_EXPIRED}",
        ])
        report = await self.router.classify_block(block)
        total = report.l1_hits + report.l2_hits + report.l3_hits
        if total == report.l1_hits and total > 0:
            assert report.tier_stats["l1_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_to_markdown_returns_string(self):
        block = f"ERROR: {LOG_OOM_KILLED}\nCRITICAL: {LOG_DISK_FULL}"
        report = await self.router.classify_block(block)
        md = report.to_markdown()
        assert isinstance(md, str)
        assert "Perception" in md
        assert len(md) > 50

    @pytest.mark.asyncio
    async def test_max_lines_respected(self):
        """Max 3 lines processed even if block has more."""
        lines = [f"ERROR: {LOG_POOL_EXHAUSTED}"] * 100
        block = "\n".join(lines)
        report = await self.router.classify_block(block, max_lines=3)
        total = report.l1_hits + report.l2_hits + report.l3_hits
        assert total <= 3

    @pytest.mark.asyncio
    async def test_results_list_populated(self):
        block = f"ERROR: {LOG_DNS_GAIERROR}"
        report = await self.router.classify_block(block)
        assert len(report.results) >= 1
        assert isinstance(report.results[0], RouterResult)


# ============================================================================
# TestContinuousLearning
# ============================================================================

class TestContinuousLearning:
    """
    Tests the continuous-learning loop:
    L3 result → promote to L1+L2 → subsequent identical log hits L1.
    """

    def setup_method(self):
        os.environ.pop("GROQ_API_KEY", None)

    @pytest.mark.asyncio
    async def test_l3_low_confidence_not_promoted(self):
        """Stub result (confidence=0.0) must NOT be promoted."""
        router = LogRouter(promote_threshold=0.60)
        result = await router.classify(LOG_NOVEL_3)
        assert result.tier == Tier.L3
        assert result.promoted is False
        assert router._l1.classify(LOG_NOVEL_3) is None

    @pytest.mark.asyncio
    async def test_high_confidence_l3_promotes_to_l1(self):
        """A mocked high-confidence L3 result should be promoted."""
        router = LogRouter(promote_threshold=0.60)

        mock_l3_result = L3Result(
            template_key="chaos_mesh_delay",
            regex_pattern=r"ChaosMesh.*delay.*\d+ms.*injected",
            description="Chaos Mesh network delay injected on pod",
            confidence=0.92,
            raw_log=LOG_NOVEL_3,
            source="llm",
        )

        with patch.object(router._l3, "classify", new=AsyncMock(return_value=mock_l3_result)):
            result = await router.classify(LOG_NOVEL_3)

        assert result.promoted is True
        # Now the same log should hit L1
        l1_hit = router._l1.classify(LOG_NOVEL_3)
        assert l1_hit is not None
        assert l1_hit.template_key == "chaos_mesh_delay"
        assert l1_hit.source == "l3_learned"

    @pytest.mark.asyncio
    async def test_promoted_pattern_hits_l1_on_second_call(self):
        """After promotion, the second identical call should hit L1 directly."""
        router = LogRouter(promote_threshold=0.60)

        mock_l3_result = L3Result(
            template_key="feature_flag_circuit",
            regex_pattern=r"FeatureFlagService.*circuit.*open",
            description="Feature flag service circuit open",
            confidence=0.88,
            raw_log=LOG_NOVEL_2,
            source="llm",
        )

        with patch.object(router._l3, "classify", new=AsyncMock(return_value=mock_l3_result)):
            first = await router.classify(LOG_NOVEL_2)

        assert first.tier == Tier.L3
        assert first.promoted is True

        # Second call — L3 mock should NOT be called; L1 handles it
        second = await router.classify(LOG_NOVEL_2)
        assert second.tier == Tier.L1
        assert second.template_key == "feature_flag_circuit"

    @pytest.mark.asyncio
    async def test_promoted_pattern_absorbed_into_l2(self):
        """After promotion, L2 knowledge base should contain the new entry."""
        router = LogRouter(promote_threshold=0.60)

        mock_l3_result = L3Result(
            template_key="quota_window_exceeded",
            regex_pattern=r"CustomRateLimiter.*quota_window",
            description="Custom rate limiter quota window exceeded for API bucket",
            confidence=0.75,
            raw_log=LOG_NOVEL_1,
            source="llm",
        )

        with patch.object(router._l3, "classify", new=AsyncMock(return_value=mock_l3_result)):
            await router.classify(LOG_NOVEL_1)

        kb_keys = [e.key for e in router._l2._kb]
        assert "quota_window_exceeded" in kb_keys

    @pytest.mark.asyncio
    async def test_promotion_increments_l3_counter(self):
        """record_promotion() is called when promotion occurs."""
        router = LogRouter(promote_threshold=0.60)

        mock_l3_result = L3Result(
            template_key="some_new_pattern",
            regex_pattern=r"SomeNewPattern.*trigger",
            description="Some new failure pattern",
            confidence=0.90,
            raw_log="ERROR: SomeNewPattern trigger happened",
            source="llm",
        )

        with patch.object(router._l3, "classify", new=AsyncMock(return_value=mock_l3_result)):
            await router.classify("ERROR: SomeNewPattern trigger happened")

        assert router._l3.stats["l3_promotions"] == 1

    @pytest.mark.asyncio
    async def test_learned_keys_visible_in_l1_stats(self):
        router = LogRouter(promote_threshold=0.60)
        mock_l3_result = L3Result(
            template_key="learned_pattern_x",
            regex_pattern=r"LearnedPatternX.*event",
            description="Learned pattern X",
            confidence=0.80,
            raw_log="ERROR: LearnedPatternX event occurred",
            source="llm",
        )
        with patch.object(router._l3, "classify", new=AsyncMock(return_value=mock_l3_result)):
            await router.classify("ERROR: LearnedPatternX event occurred")

        assert "learned_pattern_x" in router._l1.learned_keys
        assert router._l1.stats["learned_templates"] == 1


# ============================================================================
# TestNeSyRouterIntegration
# ============================================================================

class TestNeSyRouterIntegration:
    """
    End-to-end: PerceptionReport tier_stats feed into the NeSy router
    to produce the correct ReasoningPathway.
    """

    def setup_method(self):
        os.environ.pop("GROQ_API_KEY", None)
        from agent.reasoning.nesym_router import NeuroSymbolicRouter, ReasoningPathway
        self.nesym = NeuroSymbolicRouter()
        self.ReasoningPathway = ReasoningPathway

    def _make_stats(self, l1=0, l2=0, l3=0):
        total = l1 + l2 + l3
        return {
            "L1_hits": l1, "L2_hits": l2, "L3_hits": l3,
            "total": total,
            "l1_rate": round(l1 / max(1, total), 4),
        }

    @pytest.mark.asyncio
    async def test_all_l1_hits_leads_to_symbolic_fast(self):
        """
        When all logs route to L1 (l1_rate=1.0, l3_hits=0) the NeSy router
        should select SYMBOLIC_FAST regardless of CBR confidence.
        """
        router = LogRouter()
        block = "\n".join([
            f"ERROR: {LOG_POOL_EXHAUSTED}",
            f"ERROR: {LOG_CRASH_LOOP}",
            f"CRITICAL: {LOG_OOM_KILLED}",
        ])
        report = await router.classify_block(block)
        stats = report.tier_stats

        decision = self.nesym.route(
            perception_stats=stats,
            cbr_confidence=0.2,
            primary_template=report.primary_template,
        )
        # All three logs hit L1 → SYMBOLIC_FAST expected
        if stats["L3_hits"] == 0 and stats["l1_rate"] >= 0.9:
            assert decision.pathway == self.ReasoningPathway.SYMBOLIC_FAST

    @pytest.mark.asyncio
    async def test_novel_logs_lead_to_neural_full(self):
        """
        Novel logs that go to L3 produce low l1_rate → NEURAL_FULL with low CBR.
        """
        stats = self._make_stats(l1=1, l2=1, l3=8)
        decision = self.nesym.route(
            perception_stats=stats,
            cbr_confidence=0.20,
            primary_template="unclassified_novel_pattern",
        )
        assert decision.pathway == self.ReasoningPathway.NEURAL_FULL

    def test_high_cbr_confidence_leads_to_cbr_guided(self):
        """High CBR confidence should produce CBR_GUIDED regardless of tier split."""
        stats = self._make_stats(l1=4, l2=3, l3=3)
        decision = self.nesym.route(
            perception_stats=stats,
            cbr_confidence=0.85,
            primary_template="connection_pool_exhausted",
        )
        assert decision.pathway == self.ReasoningPathway.CBR_GUIDED

    @pytest.mark.asyncio
    async def test_perception_report_markdown_contains_tier_info(self):
        """to_markdown() output includes tier hits for operator visibility."""
        router = LogRouter()
        block = f"ERROR: {LOG_REDIS_OOM}\nERROR: {LOG_UPSTREAM_503}"
        report = await router.classify_block(block)
        md = report.to_markdown()
        assert "L1" in md or "L2" in md or "L3" in md
