-- ═══════════════════════════════════════════════════════════════
-- 07-clickhouse-schema.sql
-- Golden Blueprint — ClickHouse Tiered Log Schema
--
-- This script creates the telemetry database and all three
-- tiered log tables (hot, warm, cold) plus the K8s events table.
--
-- Run against the ClickHouse cluster:
--   cat 07-clickhouse-schema.sql | clickhouse-client --multiquery
--
-- For ReplicatedMergeTree, ensure you have a ClickHouse Keeper
-- (or ZooKeeper) cluster running. If running single-node ClickHouse
-- for development, replace ReplicatedMergeTree with MergeTree and
-- remove the path/replica arguments.
-- ═══════════════════════════════════════════════════════════════

CREATE DATABASE IF NOT EXISTS telemetry;

-- ═══════════════════════════════════════════════════════════════
-- HOT TIER: 7-day TTL, SSD storage, sub-100ms query latency
-- Contains: ERROR, FATAL, CRITICAL, PANIC, and rescued events
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS telemetry.logs_hot (
    -- Temporal
    Timestamp       DateTime64(9)            CODEC(Delta, ZSTD(1)),

    -- Kubernetes Identity
    Namespace       LowCardinality(String)   CODEC(ZSTD(1)),
    PodName         String                   CODEC(ZSTD(1)),
    PodUID          String                   CODEC(ZSTD(1)),
    ContainerName   LowCardinality(String)   CODEC(ZSTD(1)),
    NodeName        LowCardinality(String)   CODEC(ZSTD(1)),
    ServiceName     LowCardinality(String)   CODEC(ZSTD(1)),

    -- Log Content
    Severity        LowCardinality(String)   CODEC(ZSTD(1)),
    Message         String                   CODEC(ZSTD(1)),
    LogFormat       LowCardinality(String)   CODEC(ZSTD(1)),  -- 'json' | 'text' | 'binary'

    -- Provenance & Classification
    Source          LowCardinality(String)   CODEC(ZSTD(1)),  -- 'container' | 'init_container' | 'k8s_event' | 'previous'
    SourceClass     LowCardinality(String)   CODEC(ZSTD(1)),  -- 'infrastructure' | 'application'
    K8sReason       LowCardinality(String)   CODEC(ZSTD(1)),  -- K8s event reason (FailedScheduling, OOMKilling, etc.)
    Rescued         UInt8                    DEFAULT 0,        -- 1 = matched semantic safety net
    RescueReason    LowCardinality(String)   CODEC(ZSTD(1)),

    -- Extensible metadata (arbitrary K8s labels)
    Labels          Map(String, String)      CODEC(ZSTD(1)),

    -- ── Data Skipping Indexes ────────────────────────────────

    -- Full-text inverted index: accelerates LIKE, ILIKE, hasToken()
    -- Protects ClickHouse from full-table scans when AI generates
    -- naive SQL like: WHERE Message ILIKE '%exception%'
    INDEX idx_message Message
        TYPE tokenbf_v1(32768, 3, 0) GRANULARITY 1,

    -- Bloom filter on ServiceName for fast service-scoped queries
    INDEX idx_service ServiceName
        TYPE bloom_filter(0.01) GRANULARITY 1,

    -- Set index on Severity for fast severity filtering
    INDEX idx_severity Severity
        TYPE set(10) GRANULARITY 4,

    -- Bloom filter on Source for provenance-based queries
    INDEX idx_source Source
        TYPE bloom_filter(0.01) GRANULARITY 1

) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(Timestamp)
ORDER BY (Namespace, ServiceName, PodName, Timestamp)
TTL toDateTime(Timestamp) + INTERVAL 7 DAY DELETE
SETTINGS
    index_granularity = 8192,
    merge_with_ttl_timeout = 86400,
    ttl_only_drop_parts = 1;


-- ═══════════════════════════════════════════════════════════════
-- WARM TIER: 30-day TTL, SSD/HDD hybrid
-- Contains: WARN events
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS telemetry.logs_warm (
    Timestamp       DateTime64(9)            CODEC(Delta, ZSTD(3)),
    Namespace       LowCardinality(String)   CODEC(ZSTD(3)),
    PodName         String                   CODEC(ZSTD(3)),
    PodUID          String                   CODEC(ZSTD(3)),
    ContainerName   LowCardinality(String)   CODEC(ZSTD(3)),
    NodeName        LowCardinality(String)   CODEC(ZSTD(3)),
    ServiceName     LowCardinality(String)   CODEC(ZSTD(3)),
    Severity        LowCardinality(String)   CODEC(ZSTD(3)),
    Message         String                   CODEC(ZSTD(3)),
    LogFormat       LowCardinality(String)   CODEC(ZSTD(3)),
    Source          LowCardinality(String)   CODEC(ZSTD(3)),
    SourceClass     LowCardinality(String)   CODEC(ZSTD(3)),
    K8sReason       LowCardinality(String)   CODEC(ZSTD(3)),
    Rescued         UInt8                    DEFAULT 0,
    RescueReason    LowCardinality(String)   CODEC(ZSTD(3)),
    Labels          Map(String, String)      CODEC(ZSTD(3)),

    INDEX idx_message Message TYPE tokenbf_v1(32768, 3, 0) GRANULARITY 1,
    INDEX idx_service ServiceName TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_severity Severity TYPE set(10) GRANULARITY 4

) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(Timestamp)
ORDER BY (Namespace, ServiceName, PodName, Timestamp)
TTL toDateTime(Timestamp) + INTERVAL 30 DAY DELETE
SETTINGS
    index_granularity = 8192,
    merge_with_ttl_timeout = 86400,
    ttl_only_drop_parts = 1;


-- ═══════════════════════════════════════════════════════════════
-- COLD TIER: 90-day TTL, S3/object storage via ClickHouse tiered
-- Contains: INFO/DEBUG that survived all filters
-- Higher ZSTD compression level (5) to minimize storage cost
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS telemetry.logs_cold (
    Timestamp       DateTime64(9)            CODEC(Delta, ZSTD(5)),
    Namespace       LowCardinality(String)   CODEC(ZSTD(5)),
    PodName         String                   CODEC(ZSTD(5)),
    PodUID          String                   CODEC(ZSTD(5)),
    ContainerName   LowCardinality(String)   CODEC(ZSTD(5)),
    NodeName        LowCardinality(String)   CODEC(ZSTD(5)),
    ServiceName     LowCardinality(String)   CODEC(ZSTD(5)),
    Severity        LowCardinality(String)   CODEC(ZSTD(5)),
    Message         String                   CODEC(ZSTD(5)),
    LogFormat       LowCardinality(String)   CODEC(ZSTD(5)),
    Source          LowCardinality(String)   CODEC(ZSTD(5)),
    SourceClass     LowCardinality(String)   CODEC(ZSTD(5)),
    K8sReason       LowCardinality(String)   CODEC(ZSTD(5)),
    Rescued         UInt8                    DEFAULT 0,
    RescueReason    LowCardinality(String)   CODEC(ZSTD(5)),
    Labels          Map(String, String)      CODEC(ZSTD(5)),

    -- Minimal indexing on cold tier to reduce storage overhead
    INDEX idx_message Message TYPE tokenbf_v1(32768, 3, 0) GRANULARITY 4,
    INDEX idx_service ServiceName TYPE bloom_filter(0.01) GRANULARITY 4

) ENGINE = MergeTree()
PARTITION BY toYYYYMM(Timestamp)  -- Monthly partitions for cold data
ORDER BY (Namespace, ServiceName, PodName, Timestamp)
TTL toDateTime(Timestamp) + INTERVAL 90 DAY DELETE
SETTINGS
    index_granularity = 8192,
    merge_with_ttl_timeout = 86400,
    ttl_only_drop_parts = 1;


-- ═══════════════════════════════════════════════════════════════
-- K8S EVENTS TABLE
-- Receives Kubernetes events from the OTel Collector
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS telemetry.k8s_events (
    Timestamp       DateTime64(9)            CODEC(Delta, ZSTD(1)),
    Namespace       LowCardinality(String)   CODEC(ZSTD(1)),
    ObjectName      String                   CODEC(ZSTD(1)),  -- Involved object (pod, node, etc.)
    ObjectKind      LowCardinality(String)   CODEC(ZSTD(1)),  -- Pod, Node, Deployment, etc.
    EventType       LowCardinality(String)   CODEC(ZSTD(1)),  -- Normal, Warning
    Reason          LowCardinality(String)   CODEC(ZSTD(1)),  -- FailedScheduling, OOMKilling, etc.
    Message         String                   CODEC(ZSTD(1)),
    Source          LowCardinality(String)   DEFAULT 'k8s_event' CODEC(ZSTD(1)),
    ReportingController LowCardinality(String) CODEC(ZSTD(1)),
    Count           UInt32                   DEFAULT 1,

    INDEX idx_message Message TYPE tokenbf_v1(32768, 3, 0) GRANULARITY 1,
    INDEX idx_reason Reason TYPE set(50) GRANULARITY 1,
    INDEX idx_object_kind ObjectKind TYPE set(20) GRANULARITY 1

) ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(Timestamp)
ORDER BY (Namespace, ObjectKind, ObjectName, Timestamp)
TTL toDateTime(Timestamp) + INTERVAL 30 DAY DELETE
SETTINGS
    index_granularity = 8192;


-- ═══════════════════════════════════════════════════════════════
-- UNIFIED VIEW: Allows AI agent to query across all tiers with
-- a single SQL statement. ClickHouse merges results transparently.
-- ═══════════════════════════════════════════════════════════════
CREATE VIEW IF NOT EXISTS telemetry.logs_all AS
SELECT * FROM telemetry.logs_hot
UNION ALL
SELECT * FROM telemetry.logs_warm
UNION ALL
SELECT * FROM telemetry.logs_cold;


-- ═══════════════════════════════════════════════════════════════
-- QUERY QUOTA: Protect ClickHouse from runaway AI-generated SQL
-- Max execution time: 10 seconds
-- Max rows scanned: 100 million
-- Max memory per query: 2 GB
-- ═══════════════════════════════════════════════════════════════
CREATE SETTINGS PROFILE IF NOT EXISTS 'airs_agent_profile'
SETTINGS
    max_execution_time = 10,
    max_rows_to_read = 100000000,
    max_memory_usage = 2147483648,
    max_result_rows = 10000;

-- Create the AIRS agent user (replace password in production)
CREATE USER IF NOT EXISTS airs_agent
    IDENTIFIED WITH sha256_password BY 'CHANGE_ME_IN_PRODUCTION'
    DEFAULT DATABASE telemetry
    SETTINGS PROFILE 'airs_agent_profile';

GRANT SELECT ON telemetry.* TO airs_agent;
