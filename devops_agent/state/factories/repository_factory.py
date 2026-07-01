"""RepositoryFactory — builds and wires all repositories and managers.

Usage (at application startup)::

    config = PersistenceConfig.from_env()
    factory = RepositoryFactory(config)
    await factory.connect()

    inv_repo = factory.investigation_repo
    cp_mgr   = factory.checkpoint_manager
    ...

    await factory.disconnect()
"""

from __future__ import annotations

from ..adapters.langgraph_checkpointer import LangGraphCheckpointerAdapter
from ..adapters.postgres_adapter import PostgresAdapter
from ..adapters.redis_adapter import RedisAdapter
from ..config.persistence_config import PersistenceConfig
from ..managers.checkpoint_manager import CheckpointManager
from ..managers.migration_manager import MigrationManager
from ..managers.resume_manager import ResumeManager
from ..repositories.postgres_baseline_repository import PostgresBaselineRepository
from ..repositories.postgres_checkpoint_repository import PostgresCheckpointRepository
from ..repositories.postgres_circuit_breaker_repository import PostgresCircuitBreakerRepository
from ..repositories.postgres_investigation_repository import PostgresInvestigationRepository
from ..repositories.redis_artifact_cache_repository import RedisArtifactCacheRepository


class RepositoryFactory:
    """Single DI root that wires all adapters, repositories, and managers.

    Call ``await factory.connect()`` before use.
    Call ``await factory.disconnect()`` on shutdown.
    """

    def __init__(self, config: PersistenceConfig):
        self._config = config

        # Adapters
        self._pg = PostgresAdapter(
            dsn=config.postgres.dsn,
            min_connections=config.postgres.min_connections,
            max_connections=config.postgres.max_connections,
            statement_cache_size=config.postgres.statement_cache_size,
            command_timeout=config.postgres.command_timeout,
        )
        self._redis = RedisAdapter(
            url=config.redis.url,
            decode_responses=config.redis.decode_responses,
            max_connections=config.redis.max_connections,
        )

        # Repositories
        self._inv_repo = PostgresInvestigationRepository(self._pg)
        self._cp_repo = PostgresCheckpointRepository(self._pg)
        self._baseline_repo = PostgresBaselineRepository(self._pg)
        self._artifact_cache = RedisArtifactCacheRepository(self._redis)
        self._cb_repo = PostgresCircuitBreakerRepository(self._pg)

        # Managers
        self._cp_mgr = CheckpointManager(self._cp_repo, self._inv_repo)
        self._resume_mgr = ResumeManager(self._cp_mgr, self._inv_repo)
        self._migration_mgr = MigrationManager(self._pg)

        # LangGraph adapter
        self._lg_adapter = LangGraphCheckpointerAdapter(self._cp_repo)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        await self._pg.connect()
        await self._redis.connect()

    async def disconnect(self) -> None:
        await self._redis.disconnect()
        await self._pg.disconnect()

    # ── Repositories ──────────────────────────────────────────────────────────

    @property
    def investigation_repo(self) -> PostgresInvestigationRepository:
        return self._inv_repo

    @property
    def checkpoint_repo(self) -> PostgresCheckpointRepository:
        return self._cp_repo

    @property
    def baseline_repo(self) -> PostgresBaselineRepository:
        return self._baseline_repo

    @property
    def artifact_cache(self) -> RedisArtifactCacheRepository:
        return self._artifact_cache

    @property
    def circuit_breaker_repo(self) -> PostgresCircuitBreakerRepository:
        return self._cb_repo

    # ── Managers ──────────────────────────────────────────────────────────────

    @property
    def checkpoint_manager(self) -> CheckpointManager:
        return self._cp_mgr

    @property
    def resume_manager(self) -> ResumeManager:
        return self._resume_mgr

    @property
    def migration_manager(self) -> MigrationManager:
        return self._migration_mgr

    # ── LangGraph ─────────────────────────────────────────────────────────────

    @property
    def langgraph_checkpointer(self) -> LangGraphCheckpointerAdapter:
        return self._lg_adapter

    # ── Adapters (for advanced usage) ─────────────────────────────────────────

    @property
    def postgres_adapter(self) -> PostgresAdapter:
        return self._pg

    @property
    def redis_adapter(self) -> RedisAdapter:
        return self._redis
