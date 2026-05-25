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

    # HuggingFace Inference API
    HF_ACCESS_KEY: Optional[str] = None
    """
    HuggingFace API key for sentence-transformer embeddings via InferenceClient.
    Maps to the HF_ACCESS_KEY variable in .env.
    """

    # Qdrant Vector Store Settings
    QDRANT_URL: str = "http://localhost:6333"
    """
    URL of the Qdrant instance.
    Use http://localhost:6333 for local Docker dev.
    Override to a Qdrant Cloud URL for production.
    """
    QDRANT_COLLECTION: str = "kb_entries"
    """Name of the Qdrant collection that holds KB entry vectors."""

    # Knowledge Base (KB) Settings
    KB_EXACT_BYPASS_THRESHOLD: float = 0.95
    """
    Score at or above which the KB result is treated as an exact hit,
    bypassing the full LLM pipeline (investigate → extract → plan → approval)
    and routing directly to execute_node with is_approved=True.
    """
    KB_RAG_THRESHOLD: float = 0.70
    """
    Score at or above which the KB entry is injected into plan_node as
    Retrieval-Augmented Generation context.  Below this threshold the LLM
    operates in read-only diagnostic mode (no kubectl commands generated).
    """
    KB_MAX_TELEMETRY_CHARS: int = 16_000
    """
    Hard character cap on the telemetry string passed to extract_node and
    plan_node.  Prevents context-window overflow during cascading failures.
    ~4000 tokens at 4 chars/token.
    """
    KB_TOP_K: int = 5
    """
    Number of vector-search candidates retrieved from Qdrant before LLM
    reranking. Increasing this improves recall at the cost of more LLM calls.
    """

settings = Settings()
