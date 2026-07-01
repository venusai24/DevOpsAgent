"""Persistence layer configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PostgresConfig:
    """Connection configuration for the primary PostgreSQL instance.

    Hosts: Investigation Registry, LangGraph Checkpoint Store,
           Baseline Statistics Store, Circuit Breaker State.
    """
    dsn: str = "postgresql://devops_agent:devops_agent@localhost:5432/devops_agent"
    min_connections: int = 2
    max_connections: int = 20
    statement_cache_size: int = 100
    command_timeout: float = 30.0


@dataclass
class RedisConfig:
    """Connection configuration for the Stage 0 Artifact Cache."""
    url: str = "redis://localhost:6379/0"
    max_connections: int = 20
    decode_responses: bool = True


@dataclass
class PersistenceConfig:
    """Top-level persistence configuration — injected at app startup."""
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)

    # Retention settings.
    # Days to retain checkpoints after an investigation reaches a terminal status.
    checkpoint_retention_days: int = 30

    # Hours after which a hitl_paused investigation is flagged as stale.
    hitl_watchdog_hours: int = 24

    # Maximum SERIALIZABLE dedup transaction retries.
    max_dedup_retries: int = 3

    @classmethod
    def from_env(cls) -> PersistenceConfig:
        """Build a config from environment variables (for production use)."""
        import os
        pg_dsn = os.environ.get(
            "DEVOPS_AGENT_PG_DSN",
            "postgresql://devops_agent:devops_agent@localhost:5432/devops_agent",
        )
        redis_url = os.environ.get("DEVOPS_AGENT_REDIS_URL") or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        return cls(
            postgres=PostgresConfig(
                dsn=pg_dsn,
                min_connections=int(os.environ.get("PG_MIN_CONN", "2")),
                max_connections=int(os.environ.get("PG_MAX_CONN", "20")),
            ),
            redis=RedisConfig(url=redis_url),
            checkpoint_retention_days=int(os.environ.get("CHECKPOINT_RETENTION_DAYS", "30")),
            hitl_watchdog_hours=int(os.environ.get("HITL_WATCHDOG_HOURS", "24")),
        )
