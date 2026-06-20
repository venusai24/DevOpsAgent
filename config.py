from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """
    Global settings for the AIRS Production Environment.
    Uses pydantic_settings to load from .env file or environment variables.
    """
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # LLM Settings
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "qwen/qwen3-32b"

    # Database Settings
    DATABASE_URL: Optional[str] = None
    
    # Celery / Redis Settings
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None

    # Telemetry Integrations
    DATADOG_API_KEY: Optional[str] = None
    DATADOG_APP_KEY: Optional[str] = None
    SPLUNK_URL: Optional[str] = None
    SPLUNK_TOKEN: Optional[str] = None
    
    # Slack Integration
    SLACK_BOT_TOKEN: Optional[str] = None
    SLACK_SIGNING_SECRET: Optional[str] = None
    SLACK_CHANNEL_ID: Optional[str] = None

    # Kubernetes
    KUBECONFIG_PATH: Optional[str] = None

    # Mock settings (fallback)
    MOCK_API_BASE_URL: str = "http://localhost:8000"

    # Neo4j — Enterprise Knowledge Graph (EKG)
    # Leave unset to use in-memory NetworkX graph (demo mode)
    NEO4J_URI: Optional[str] = None
    NEO4J_USER: Optional[str] = None
    NEO4J_PASSWORD: Optional[str] = None

    # Topology fixture path (relative to project root)
    TOPOLOGY_FIXTURES_PATH: str = "mock_enterprise/topology_fixtures.json"

    # CBR / IncidentStore
    # When DATABASE_URL is set, PostgreSQL is used; otherwise in-memory
    CBR_MIN_SIMILARITY: float = 0.5       # Minimum cosine similarity for CBR match
    CBR_TOP_K: int = 3                    # Number of top historical cases to retrieve

    # Canary Controller
    CANARY_DEMO_MODE: bool = True         # Use simulated signals (set False in production)

    # Checkpoint SQLite (local dev)
    CHECKPOINT_DB_PATH: str = "airs_checkpoint.db"

    # ── Phase 5: Embedding Model ──────────────────────────────────────────────
    # None = auto-select: GPU → BAAI/bge-large-en-v1.5, CPU → all-MiniLM-L6-v2
    EMBEDDING_MODEL: Optional[str] = None

    # ── Phase 5: RAG / Vector Store ───────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = ".chromadb_data"
    RAG_TOP_K: int = 5                    # Final context window size after reranking
    RAG_CANDIDATE_K: int = 50             # Broad retrieval candidates before reranking
    RAG_MIN_SIMILARITY: float = 0.3       # Minimum cosine similarity to include
    RAG_RERANKING_ENABLED: bool = True
    # Sufficiency thresholds for hierarchical retrieval (skip docs if enough incidents)
    RAG_INCIDENT_SUFFICIENCY_COUNT: int = 3
    RAG_INCIDENT_SUFFICIENCY_SIMILARITY: float = 0.6
    # Amendment #4: Context compression guard — Phase 2 will apply LLMLingua-2
    # when assembled RAG context exceeds this token estimate.
    MAX_RAG_CONTEXT_TOKENS: int = 8000

    # ── Phase 5: Reranker (Amendment #3) ─────────────────────────────────────
    # Phase 1 default: ms-marco-MiniLM-L-6-v2 (CPU-friendly, fast)
    # Phase 2 upgrade: BAAI/bge-reranker-v2-m3 (higher accuracy, GPU recommended)
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Phase 5: Knowledge Graph ──────────────────────────────────────────────
    GRAPH_BACKEND: str = "networkx"       # "neo4j" for Phase 2 production
    GRAPH_PERSIST_PATH: str = ".knowledge_graph.json"

    # ── Phase 5: HITL Confidence Gate ────────────────────────────────────────
    ACTION_CONFIDENCE_THRESHOLD: float = 0.75
    HITL_TIMEOUT_S: float = 3600.0        # 1 hour

    # ── Phase 5: Learning Loop ────────────────────────────────────────────────
    LEARNING_LOOP_ENABLED: bool = True

    # ── Phase 5: Corpus ───────────────────────────────────────────────────────
    CORPUS_DIR: str = "Corpus"            # Root of the static document corpus

    # ── Golden Blueprint: ClickHouse Log Backend ──────────────────────────────
    # When the Golden Blueprint pipeline is deployed, set these to point at the
    # ClickHouse cluster. When unset, the system falls back to K8s API polling.
    CLICKHOUSE_HOST: str = "clickhouse.observe.svc.cluster.local"
    CLICKHOUSE_PORT: int = 9000
    CLICKHOUSE_DB: str = "telemetry"
    CLICKHOUSE_USER: str = "airs_agent"
    CLICKHOUSE_PASSWORD: str = ""
    # Lookback window (minutes) for forensic log queries.
    # Increase for long-running incidents where the anomaly may have started
    # well before the Prometheus alert fired.
    CLICKHOUSE_LOOKBACK_MINUTES: int = 60
    # Maximum rows returned per pod per forensic query (across all tiers).
    CLICKHOUSE_MAX_ROWS: int = 200

settings = Settings()
