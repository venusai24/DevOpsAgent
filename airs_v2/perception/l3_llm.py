"""
airs_v2/perception/l3_llm.py
=============================

L3 — LLM Fallback for Novel Log Patterns
-----------------------------------------
The most expensive tier in the NeSy-Edge log parsing router, triggered only
when a log entry cannot be classified by L1 (regex) or L2 (TF-IDF cosine).

Responsibilities
----------------
1. Ask the configured LLM to extract a **canonical template key** and a
   **regex pattern** from the novel log text.
2. Return a ``L3Result`` with the extracted key, confidence, and the
   learned regex pattern string.
3. Optionally **promote** the learned pattern back to L1 + L2 (continuous
   learning loop) — the ``LogRouter`` orchestrates this call after receiving
   the L3 result.

Design principles
-----------------
* **Lazy initialisation** — the LLM client is created only on the first L3
  invocation, so test suites that never hit L3 pay zero cost.
* **Stub mode** — when ``GROQ_API_KEY`` is absent (CI / unit tests) the
  layer returns a deterministic ``L3Result`` with ``source="stub"`` and a
  fixed ``confidence=0.0`` so tests can assert the routing path without a
  live API call.
* **Timeout guard** — async calls are wrapped with ``asyncio.wait_for`` to
  prevent the router from blocking indefinitely on a slow LLM.
* **Structured output** — the prompt demands a strict JSON envelope;
  ``_parse_l3_response`` applies layered fallbacks so a malformed LLM reply
  never crashes the pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM import (optional — fails gracefully in stub mode)
# ---------------------------------------------------------------------------
try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage
    _HAS_GROQ = True
except ImportError:
    ChatGroq = None  # type: ignore
    HumanMessage = None  # type: ignore
    _HAS_GROQ = False

_L3_TIMEOUT_SECONDS = float(os.getenv("AIRS_L3_TIMEOUT", "15.0"))
_DEFAULT_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_L3_SYSTEM_PROMPT = """\
You are a log analysis expert for a Site Reliability Engineering team.
A log entry could not be matched to any known failure pattern.
Your task: extract a canonical template from this single log line.

Rules:
- template_key: snake_case identifier (e.g. "kafka_producer_timeout")
- regex_pattern: a simple Python regex that would match this exact pattern
  in future logs.  Use \\d+ for numbers, .* for variable text.
  The regex must compile without error. Do NOT use Python's raw string `r""` prefix in the JSON; use standard JSON string escaping (e.g., "\\\\d+").
- description: one short sentence describing the failure class
- confidence: float 0.0–1.0 reflecting how certain you are this is a
  real failure pattern (not normal INFO noise)

Respond ONLY with a JSON object — no prose, no markdown fences:
{
  "template_key": "<snake_case_key>",
  "regex_pattern": "<compilable python regex>",
  "description": "<short description>",
  "confidence": <float>
}
"""


@dataclass
class L3Result:
    """Result produced by the L3 LLM tier."""
    template_key: str
    regex_pattern: str          # Learned regex (used for L1 promotion)
    description: str            # Human-readable description (used for L2 promotion)
    confidence: float
    raw_log: str                # Original log entry passed to LLM
    source: str = "llm"        # "llm" | "stub"
    tier: str = "L3"
    is_novel: bool = True       # Always True for L3 — new pattern discovered


class L3LLMFallback:
    """
    LLM-backed fallback classifier for novel log patterns.

    Usage
    -----
    ::

        l3 = L3LLMFallback()
        result = await l3.classify("some weird log line no one has seen before")
        if result.confidence > 0.6:
            # promote to L1 and L2
            ...

    Stub mode (no API key)
    ----------------------
    When ``GROQ_API_KEY`` is not set, every call returns::

        L3Result(template_key="unclassified_novel_pattern",
                 regex_pattern="",
                 description="Could not classify — LLM unavailable",
                 confidence=0.0, source="stub")
    """

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self._model = model
        self._llm: Optional[Any] = None   # Lazy-initialised
        self._call_count = 0
        self._error_count = 0
        self._promotion_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def classify(self, log_entry: str) -> L3Result:
        """
        Classify a novel log entry via LLM.

        Never raises — all errors produce a low-confidence stub result.
        """
        self._call_count += 1

        if not self._is_available():
            return self._stub_result(log_entry, reason="no_api_key")

        try:
            raw_response = await asyncio.wait_for(
                self._invoke_llm(log_entry),
                timeout=_L3_TIMEOUT_SECONDS,
            )
            result = _parse_l3_response(raw_response, log_entry)
            logger.info(
                "[L3LLM] Classified novel pattern: key=%s conf=%.2f",
                result.template_key, result.confidence,
            )
            return result
        except asyncio.TimeoutError:
            self._error_count += 1
            logger.warning("[L3LLM] LLM call timed out after %.1fs", _L3_TIMEOUT_SECONDS)
            return self._stub_result(log_entry, reason="timeout")
        except Exception as exc:
            self._error_count += 1
            logger.warning("[L3LLM] Classification failed: %s", exc)
            return self._stub_result(log_entry, reason=str(exc)[:80])

    @property
    def stats(self) -> dict:
        return {
            "l3_calls": self._call_count,
            "l3_errors": self._error_count,
            "l3_promotions": self._promotion_count,
            "available": self._is_available(),
            "model": self._model,
        }

    def record_promotion(self) -> None:
        """Called by LogRouter when a L3 result is promoted to L1+L2."""
        self._promotion_count += 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_available(self) -> bool:
        return bool(os.getenv("GROQ_API_KEY")) and _HAS_GROQ

    def _get_llm(self) -> Any:
        """Lazy-init the Groq LLM client."""
        if self._llm is None:
            self._llm = ChatGroq(
                model=self._model,
                temperature=0,
                max_retries=2,
            ).bind(response_format={"type": "json_object"})
        return self._llm

    async def _invoke_llm(self, log_entry: str) -> str:
        llm = self._get_llm()
        msg = HumanMessage(
            content=(
                f"{_L3_SYSTEM_PROMPT}\n\n"
                f"Log entry to classify:\n{log_entry[:600]}"
            )
        )
        response = await llm.ainvoke([msg])
        return response.content  # type: ignore[union-attr]

    @staticmethod
    def _stub_result(log_entry: str, reason: str = "") -> L3Result:
        return L3Result(
            template_key="unclassified_novel_pattern",
            regex_pattern="",
            description=f"Could not classify — {reason or 'stub mode'}",
            confidence=0.0,
            raw_log=log_entry,
            source="stub",
        )


# ---------------------------------------------------------------------------
# JSON parser (layered fallback)
# ---------------------------------------------------------------------------

def _parse_l3_response(raw: str, log_entry: str) -> L3Result:
    """
    Parse the LLM response JSON into an ``L3Result``.

    Fallback layers
    ---------------
    1. Try ``json.loads`` directly.
    2. Try extracting the first ``{...}`` block from the raw text.
    3. Return a stub result with confidence=0.0.
    """
    data: Optional[dict] = None

    # Layer 1: direct JSON parse
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Layer 2: extract first JSON object
    if data is None:
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    if data is None:
        logger.warning("[L3LLM] Could not parse LLM response: %.120s", raw)
        return L3Result(
            template_key="parse_failed",
            regex_pattern="",
            description="LLM response could not be parsed",
            confidence=0.0,
            raw_log=log_entry,
            source="stub",
        )

    # Validate regex before returning
    regex_str = data.get("regex_pattern", "")
    try:
        re.compile(regex_str)
    except re.error:
        logger.warning("[L3LLM] LLM returned invalid regex '%s' — discarding", regex_str[:60])
        regex_str = ""

    return L3Result(
        template_key=data.get("template_key", "unknown_pattern"),
        regex_pattern=regex_str,
        description=data.get("description", ""),
        confidence=float(data.get("confidence", 0.5)),
        raw_log=log_entry,
        source="llm",
    )
