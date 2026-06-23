"""
AIRS Configuration.

All settings are loaded from environment variables (or .env file via pydantic-settings).
Every config value has a typed default; override via .env for deployment-specific tuning.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ─── Root paths ───────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent.parent  # DevOpsAgent/


class AIRSConfig(BaseSettings):
    """
    Centralised configuration for the AIRS system.

    Loaded from .env at the DevOpsAgent project root. All fields are
    individually overridable via environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Silently ignore unknown env vars
    )

    # ─── LLM ──────────────────────────────────────────────────────────────────
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    primary_model: str = Field(
        default="groq/llama-3.1-70b-versatile",
        alias="AIRS_PRIMARY_MODEL",
        description="Primary reasoning model (LiteLLM format)",
    )
    compression_model: str = Field(
        default="groq/llama-3.1-8b-instant",
        alias="AIRS_COMPRESSION_MODEL",
        description="Context compression model (smaller, cheaper)",
    )

    # ─── Embeddings ───────────────────────────────────────────────────────────
    hf_key: str = Field(default="", alias="HF_KEY")
    embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        alias="AIRS_EMBEDDING_MODEL",
        description="HuggingFace embedding model name",
    )
    embedding_dim: int = Field(
        default=768,
        description="Dimensionality of bge-large-en-v1.5 embeddings",
    )

    # ─── Infrastructure ───────────────────────────────────────────────────────
    temporal_address: str = Field(default="localhost:7233", alias="TEMPORAL_ADDRESS")
    temporal_namespace: str = Field(default="default", alias="TEMPORAL_NAMESPACE")
    temporal_task_queue: str = Field(
        default="incident-workflow-tq", alias="TEMPORAL_TASK_QUEUE"
    )

    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_grpc_port: int = Field(default=6334, alias="QDRANT_GRPC_PORT")

    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # ─── Risk Thresholds ──────────────────────────────────────────────────────
    alpha_coverage: float = Field(
        default=0.10,
        alias="AIRS_ALPHA_COVERAGE",
        description="ConANN marginal coverage guarantee level (1-α)",
        ge=0.0,
        lt=1.0,
    )
    delta_alarm: float = Field(
        default=0.05,
        alias="AIRS_DELTA_ALARM",
        description="Supermartingale escalation threshold: escalate when M_t >= 1/delta",
        gt=0.0,
        lt=1.0,
    )
    lambda_step: float = Field(
        default=0.30,
        alias="AIRS_LAMBDA_STEP",
        description="Per-step risk ceiling; ABSTAIN/PRUNE if step_risk > lambda_step",
        gt=0.0,
        le=1.0,
    )
    epsilon_threshold: float = Field(
        default=0.05,
        alias="AIRS_EPSILON_THRESHOLD",
        description="Missing mass plateau threshold; delta < epsilon triggers plateau",
        gt=0.0,
        lt=1.0,
    )
    consecutive_low_delta: int = Field(
        default=3,
        alias="AIRS_CONSECUTIVE_LOW_DELTA",
        description="Plateau hop count before QUERY_PLAYBOOK fallback",
        ge=1,
    )

    # ─── Context Budget ───────────────────────────────────────────────────────
    max_context_tokens: int = Field(
        default=80_000, alias="AIRS_MAX_CONTEXT_TOKENS", ge=10_000
    )
    constitution_pct: float = Field(default=0.08, alias="AIRS_CONSTITUTION_PCT")
    playbook_pct: float = Field(default=0.10, alias="AIRS_PLAYBOOK_PCT")
    active_evidence_pct: float = Field(default=0.50, alias="AIRS_ACTIVE_EVIDENCE_PCT")
    summarized_evidence_pct: float = Field(
        default=0.12, alias="AIRS_SUMMARIZED_EVIDENCE_PCT"
    )
    tool_output_pct: float = Field(default=0.10, alias="AIRS_TOOL_OUTPUT_PCT")
    response_pct: float = Field(default=0.05, alias="AIRS_RESPONSE_PCT")
    safety_margin_pct: float = Field(default=0.05, alias="AIRS_SAFETY_MARGIN_PCT")
    causal_chain_quota: float = Field(
        default=0.60,
        alias="AIRS_CAUSAL_CHAIN_QUOTA",
        description="Fraction of active evidence budget reserved for the causal chain",
    )

    # ─── Operational Flags ────────────────────────────────────────────────────
    use_k8s: bool = Field(default=True, alias="AIRS_USE_K8S")
    max_hops: int = Field(
        default=50,
        alias="AIRS_MAX_HOPS",
        description="Max investigation hops before forced escalation",
    )
    max_hypotheses: int = Field(
        default=3,
        alias="AIRS_MAX_HYPOTHESES",
        description="Max concurrently tracked hypotheses",
    )
    log_level: str = Field(default="INFO", alias="AIRS_LOG_LEVEL")

    # ─── MCP Server URIs ──────────────────────────────────────────────────────
    mcp_prometheus_uri: Optional[str] = Field(default=None, alias="AIRS_MCP_PROMETHEUS_URI")
    mcp_loki_uri: Optional[str] = Field(default=None, alias="AIRS_MCP_LOKI_URI")
    mcp_opensearch_uri: Optional[str] = Field(default=None, alias="AIRS_MCP_OPENSEARCH_URI")
    mcp_jaeger_uri: Optional[str] = Field(default=None, alias="AIRS_MCP_JAEGER_URI")
    mcp_kubectl_uri: Optional[str] = Field(default=None, alias="AIRS_MCP_KUBECTL_URI")
    mcp_github_uri: Optional[str] = Field(default=None, alias="AIRS_MCP_GITHUB_URI")

    # ─── Derived Properties ───────────────────────────────────────────────────
    @property
    def constitution_tokens(self) -> int:
        return int(self.max_context_tokens * self.constitution_pct)

    @property
    def playbook_tokens(self) -> int:
        return int(self.max_context_tokens * self.playbook_pct)

    @property
    def active_evidence_tokens(self) -> int:
        return int(self.max_context_tokens * self.active_evidence_pct)

    @property
    def summarized_evidence_tokens(self) -> int:
        return int(self.max_context_tokens * self.summarized_evidence_pct)

    @property
    def tool_output_tokens(self) -> int:
        return int(self.max_context_tokens * self.tool_output_pct)

    @property
    def response_tokens(self) -> int:
        return int(self.max_context_tokens * self.response_pct)

    @property
    def causal_chain_tokens(self) -> int:
        """Token ceiling for the causal chain within active evidence."""
        return int(self.active_evidence_tokens * self.causal_chain_quota)

    @property
    def supermartingale_alarm_threshold(self) -> float:
        """M_t >= 1/delta triggers escalation."""
        return 1.0 / self.delta_alarm

    @property
    def knowledge_dir(self) -> Path:
        """Path to the synthesized knowledge corpus."""
        return ROOT_DIR / "knowledge"

    @property
    def corpus_diagnosis_dir(self) -> Path:
        return ROOT_DIR / "Corpus" / "Diagnosis"

    @property
    def corpus_remediation_dir(self) -> Path:
        return ROOT_DIR / "Corpus" / "Remediation"

    @property
    def calibration_dir(self) -> Path:
        return ROOT_DIR / "calibration"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}, got '{v}'")
        return upper


# ─── Singleton ────────────────────────────────────────────────────────────────
# Import and use `settings` everywhere instead of re-instantiating AIRSConfig.
settings = AIRSConfig()
