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

settings = Settings()
