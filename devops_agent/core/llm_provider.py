"""LLM Factory Provider for Orchestration Agents.

Two-Tier Model Strategy
───────────────────────
Tasks are classified as LIGHT or HEAVY based on expected token load and
reasoning complexity:

  LIGHT  — triage, critic
           Primary: Gemma (Google AI Studio) — fast, cheap, deterministic
           Fallback chain: Gemma backup key → muse-spark-1.2-contributor
                           (OpenRouter) → OpenCode models (OpenRouter)

  HEAVY  — rca, context_assembly
           Primary: Meta Llama 4 (Meta Model API) — large-context, strong
                    multi-step tool-calling
           Fallback chain: Gemma primary → Gemma backup → muse-spark-1.2-contributor
                           (OpenRouter) → OpenCode/Llama (OpenRouter)

This ensures expensive Meta API calls are reserved for tasks that genuinely
benefit from them, while simpler tasks stay cheap and fast.
"""

import logging
import os
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Load the keys from .env
load_dotenv()

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Role classification
# ──────────────────────────────────────────────────────────────────────────────

#: Roles that handle large, multi-step, or complex-context tasks.
#: These get the Meta API as primary (highest capability, large context window).
_HEAVY_ROLES: frozenset[str] = frozenset({"rca", "context_assembly", "reasoning"})

#: Roles that handle focused, bounded tasks with small-to-medium context.
#: These get Gemma as primary (fast, cheap, deterministic).
_LIGHT_ROLES: frozenset[str] = frozenset({"triage", "critic", "report", "evidence"})

# ──────────────────────────────────────────────────────────────────────────────
# Model identifiers
# ──────────────────────────────────────────────────────────────────────────────

# Heavy — Meta Model API (OpenAI-compatible endpoint)
_META_API_BASE = "https://api.meta.ai/v1"
# The flagship Meta Llama 4 Maverick model (large context, strong tool calling)
_META_PRIMARY_MODEL = "muse-spark-1.2-contributor"

# Light primary — Gemma via Google AI Studio
_GEMMA_PRIMARY_ENV = "GOOGLE_MODEL"
_GEMMA_PRIMARY_DEFAULT = "gemma-4-31b-it"
_GEMMA_FALLBACK = "gemma-2-9b-it"

# Shared OpenRouter fallbacks (all keys tried in order)
_MUSE_SPARK_MODEL = "muse-ai/muse-spark-1.2-contributor"
_OPENCODE_MODELS = [
    # OpenCode-compatible efficient models
    "qwen/qwen3-30b-a3b",             # Qwen 3 MoE — strong coder, low token cost
    "qwen/qwen-2.5-coder-32b-instruct",  # Qwen 2.5 Coder — fast structured output
]


class FallbackModelWrapper:
    """Wrapper that seamlessly applies LangChain fallbacks to a primary and
    multiple secondary models, including structured output and tool binding.
    """

    def __init__(self, primary: BaseChatModel, fallbacks: list[BaseChatModel]):
        self.primary = primary
        self.fallbacks = fallbacks

    def with_structured_output(self, schema: Any, **kwargs: Any):
        primary_structured = self.primary.with_structured_output(schema, **kwargs)
        fallback_structured_list = [
            f.with_structured_output(schema, **kwargs) for f in self.fallbacks
        ]
        return primary_structured.with_fallbacks(
            fallback_structured_list, exceptions_to_handle=(Exception,)
        )

    def bind_tools(self, tools: list, **kwargs):
        primary_bound = self.primary.bind_tools(tools, **kwargs)
        fallback_bound_list = [f.bind_tools(tools, **kwargs) for f in self.fallbacks]
        return FallbackModelWrapper(primary_bound, fallback_bound_list)

    def invoke(self, *args, **kwargs):
        return (
            self.primary
            .with_fallbacks(self.fallbacks, exceptions_to_handle=(Exception,))
            .invoke(*args, **kwargs)
        )

    async def ainvoke(self, *args, **kwargs):
        return await (
            self.primary
            .with_fallbacks(self.fallbacks, exceptions_to_handle=(Exception,))
            .ainvoke(*args, **kwargs)
        )


class LLMFactory:
    """Factory to instantiate specialized LLMs per agent role.

    Usage
    ─────
    llm = LLMFactory.get_llm("rca")    # Heavy tier → Meta primary
    llm = LLMFactory.get_llm("triage") # Light tier → Gemma primary
    """

    @staticmethod
    def get_llm(
        role: Literal["triage", "evidence", "reasoning", "report", "rca", "context_assembly", "critic"],
    ) -> "FallbackModelWrapper":
        """Return a configured LangChain model for the given agent role.

        Heavy roles (rca, context_assembly, reasoning) → Meta API primary.
        Light roles (triage, critic, report, evidence)  → Gemma primary.

        In both cases, the full fallback chain covers: primary backup key →
        muse-spark-1.2-contributor → OpenCode models (per available OpenRouter key).
        """
        is_heavy = role in _HEAVY_ROLES
        logger.debug(
            "LLMFactory.get_llm(role=%r) → tier=%s", role, "HEAVY" if is_heavy else "LIGHT"
        )

        # ── Credentials ───────────────────────────────────────────────────────
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        gemini_api_key_backup = os.environ.get("GEMINI_API_KEY_BACKUP")
        meta_api_key = os.environ.get("META_MODEL_API_KEY") or os.environ.get("MODEL_API_KEY")
        openrouter_keys = [
            k for k in [
                os.environ.get("OPENROUTER_KEY"),
                os.environ.get("OPENROUTER_KEY_BACKUP"),
                os.environ.get("OPENROUTER_KEY_BACKUP_2"),
            ]
            if k
        ]

        if not gemini_api_key:
            logger.warning("GEMINI_API_KEY is missing from environment variables!")
        if is_heavy and not meta_api_key:
            logger.warning(
                "META_MODEL_API_KEY is missing — heavy role '%s' will fall back to Gemma.", role
            )

        google_model = os.environ.get(_GEMMA_PRIMARY_ENV, _GEMMA_PRIMARY_DEFAULT)

        # ── Build primary model ───────────────────────────────────────────────
        if is_heavy and meta_api_key:
            primary_llm = _make_meta_model(meta_api_key, max_retries=3, timeout=360)
        else:
            # Light roles, or heavy role without Meta key → Gemma primary
            primary_llm = _make_gemma_model(
                model=google_model,
                api_key=gemini_api_key,
                max_retries=3,
                timeout=240,
            )

        # ── Build fallback chain ──────────────────────────────────────────────
        fallbacks: list[BaseChatModel] = []

        if is_heavy and meta_api_key:
            # Heavy: first fallback is Gemma (primary key)
            if gemini_api_key:
                fallbacks.append(_make_gemma_model(google_model, gemini_api_key, max_retries=2))
            if gemini_api_key_backup:
                fallbacks.append(_make_gemma_model(google_model, gemini_api_key_backup, max_retries=2))
                fallbacks.append(_make_gemma_model(_GEMMA_FALLBACK, gemini_api_key_backup, max_retries=2))
        else:
            # Light: first fallback is same Gemma model with backup key
            if gemini_api_key_backup:
                fallbacks.append(_make_gemma_model(google_model, gemini_api_key_backup, max_retries=2))
            # Then smaller Gemma model
            if gemini_api_key:
                fallbacks.append(_make_gemma_model(_GEMMA_FALLBACK, gemini_api_key, max_retries=2))
            if gemini_api_key_backup:
                fallbacks.append(_make_gemma_model(_GEMMA_FALLBACK, gemini_api_key_backup, max_retries=2))

        # ── OpenRouter fallbacks (shared: muse-spark → OpenCode models) ───────
        for or_key in openrouter_keys:
            # muse-spark-1.2-contributor — added for all roles as requested
            fallbacks.append(_make_openrouter_model(_MUSE_SPARK_MODEL, or_key, timeout=180))
            # OpenCode / efficient models
            for oc_model in _OPENCODE_MODELS:
                fallbacks.append(_make_openrouter_model(oc_model, or_key, timeout=240))

        return FallbackModelWrapper(primary_llm, fallbacks)


# ──────────────────────────────────────────────────────────────────────────────
# Private model constructors
# ──────────────────────────────────────────────────────────────────────────────

def _make_gemma_model(
    model: str,
    api_key: str | None,
    max_retries: int = 2,
    timeout: int = 240,
) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0.0,
        max_retries=max_retries,
        timeout=timeout,
    )


def _make_meta_model(
    api_key: str,
    max_retries: int = 3,
    timeout: int = 360,
) -> ChatOpenAI:
    """Meta Model API — OpenAI-compatible endpoint at https://api.meta.ai/v1."""
    return ChatOpenAI(
        model=_META_PRIMARY_MODEL,
        api_key=api_key,
        base_url=_META_API_BASE,
        temperature=0.0,
        max_retries=max_retries,
        timeout=timeout,
    )


def _make_openrouter_model(
    model: str,
    api_key: str,
    max_retries: int = 2,
    timeout: int = 240,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
        max_retries=max_retries,
        timeout=timeout,
    )
