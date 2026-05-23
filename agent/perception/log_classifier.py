"""
agent/perception/log_classifier.py

L1 → L2 → L3 cascade orchestrator for the AIRS Perception Layer.

Processes raw log entries through three escalating tiers of classification:
  L1 (Symbolic)   — Regex match against known templates. ~0ms. No ML.
  L2 (Retrieval)  — TF-IDF cosine similarity if L1 misses. ~5-10ms.
  L3 (Neural)     — LLM semantic extraction for novel patterns. ~1-2s.

L3 results are cached back into the L1 template registry for future
fast-path classification (the continuous learning loop).

This tiered approach ensures the system focuses expensive compute strictly
on high-value, undefined anomalies — known patterns cost near-zero.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from agent.perception.template_cache import TemplateCache, TemplateMatch
from agent.perception.embedding_matcher import EmbeddingMatcher
from agent.json_parser import parse_json_robust

logger = logging.getLogger(__name__)


@dataclass
class ClassifiedLog:
    """A log entry that has been processed through the perception pipeline."""
    raw_text: str
    template_key: str          # Symbolic category assigned
    tier: str                  # "L1", "L2", or "L3"
    confidence: float          # Classification confidence (0.0-1.0)
    is_novel: bool = False     # True if this was a new pattern (L3)
    extracted_params: dict[str, Any] = field(default_factory=dict)  # Extracted values


@dataclass
class PerceptionResult:
    """Aggregated output from classifying all logs in a telemetry block."""
    classified_logs: list[ClassifiedLog]
    primary_template: str           # Most frequent template key
    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    novel_patterns: list[str] = field(default_factory=list)

    @property
    def tier_stats(self) -> dict:
        total = self.l1_hits + self.l2_hits + self.l3_hits
        return {
            "L1_hits": self.l1_hits,
            "L2_hits": self.l2_hits,
            "L3_hits": self.l3_hits,
            "total": total,
            "l1_rate": round(self.l1_hits / max(1, total), 2),
        }

    def to_markdown(self) -> str:
        stats = self.tier_stats
        lines = [
            "## Perception Layer Results",
            f"**Primary Pattern**: `{self.primary_template}`",
            f"**Tier Stats**: L1={stats['L1_hits']} L2={stats['L2_hits']} L3={stats['L3_hits']}",
        ]
        if self.novel_patterns:
            lines.append(f"**Novel Patterns Learned**: {', '.join(self.novel_patterns)}")
        lines.append("\n### Classified Log Entries")
        for cl in self.classified_logs[:10]:  # cap for brevity
            lines.append(f"- `[{cl.tier}]` **{cl.template_key}** (conf={cl.confidence:.0%}): {cl.raw_text[:80]}")
        return "\n".join(lines)


_L3_PROMPT = """\
You are a log analysis expert. A log entry could not be matched to any known pattern.
Extract the semantic meaning and categorize it.

Log entry:
{log_entry}

Respond ONLY with a JSON object:
{{
  "template_key": "<snake_case_category_name>",
  "description": "<brief description of the error pattern>",
  "regex_pattern": "<a simple regex that would match this pattern in future logs>",
  "confidence": <float 0.0-1.0>,
  "extracted_params": {{"key": "value"}}
}}
"""


class TieredLogClassifier:
    """
    Orchestrates the L1 → L2 → L3 log classification cascade.

    Singleton-friendly: create once and reuse across node invocations to
    preserve the L1 template cache state (new patterns learned by L3 persist
    for the duration of the process).
    """

    def __init__(self) -> None:
        self._l1_cache = TemplateCache()
        self._l2_matcher = EmbeddingMatcher()
        self._l2_matcher.fit()  # Pre-fit at construction time
        self._llm: Optional[ChatGroq] = None

    def _get_llm(self) -> ChatGroq:
        """Lazy-initialize the L3 LLM (only created on first L3 invocation)."""
        if self._llm is None:
            model = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
            self._llm = ChatGroq(model=model, temperature=0, max_retries=3)
        return self._llm

    async def classify(self, log_entry: str) -> ClassifiedLog:
        """
        Classify a single log entry through the L1 → L2 → L3 cascade.

        Args:
            log_entry: A raw log line or short log block.

        Returns:
            ClassifiedLog with tier, template_key, and confidence populated.
        """
        # ----- L1: Symbolic regex match -----
        l1_result = self._l1_cache.classify(log_entry)
        if l1_result:
            return ClassifiedLog(
                raw_text=log_entry,
                template_key=l1_result.template_key,
                tier="L1",
                confidence=1.0,
            )

        # ----- L2: TF-IDF similarity -----
        l2_result = self._l2_matcher.find_nearest(log_entry)
        if l2_result:
            return ClassifiedLog(
                raw_text=log_entry,
                template_key=l2_result.template_key,
                tier="L2",
                confidence=l2_result.similarity,
            )

        # ----- L3: LLM semantic extraction -----
        return await self._classify_l3(log_entry)

    async def _classify_l3(self, log_entry: str) -> ClassifiedLog:
        """Invoke the LLM for novel log pattern extraction (L3 tier)."""
        logger.info("[L3] Novel log pattern — invoking LLM: %.80s", log_entry)
        try:
            llm = self._get_llm().bind(response_format={"type": "json_object"})
            response = await llm.ainvoke([
                HumanMessage(content=_L3_PROMPT.format(log_entry=log_entry[:500]))
            ])
            data = parse_json_robust(response.content)
            key = data.get("template_key", "unknown_pattern")
            regex = data.get("regex_pattern", "")
            confidence = float(data.get("confidence", 0.7))
            params = data.get("extracted_params", {})

            # Register new template in L1 cache for future fast-path matching
            if regex:
                self._l1_cache.register_template(key, regex)
                logger.info("[L3→L1] Cached new template: %s = %r", key, regex[:60])

            return ClassifiedLog(
                raw_text=log_entry,
                template_key=key,
                tier="L3",
                confidence=confidence,
                is_novel=True,
                extracted_params=params,
            )
        except Exception as exc:
            logger.warning("[L3] LLM extraction failed: %s", exc)
            return ClassifiedLog(
                raw_text=log_entry,
                template_key="unclassified",
                tier="L3",
                confidence=0.0,
            )

    async def classify_telemetry_block(self, telemetry: str) -> PerceptionResult:
        """
        Classify all notable log lines found in a telemetry block.

        Splits the telemetry on newlines and classifies lines that appear
        to be error/warning log entries (heuristic: contains ERROR, WARN, CRITICAL,
        or known error keywords).

        Returns:
            PerceptionResult aggregating all classifications and tier statistics.
        """
        import re
        # Extract lines that look like log entries
        log_lines = [
            line.strip() for line in telemetry.splitlines()
            if re.search(r"(ERROR|WARN|CRITICAL|Exception|Error|FATAL)", line, re.IGNORECASE)
            and len(line.strip()) > 10
        ]

        classified: list[ClassifiedLog] = []
        l1_hits = l2_hits = l3_hits = 0
        novel_patterns: list[str] = []

        for line in log_lines[:20]:  # Cap at 20 lines for performance
            result = await self.classify(line)
            classified.append(result)
            if result.tier == "L1":
                l1_hits += 1
            elif result.tier == "L2":
                l2_hits += 1
            else:
                l3_hits += 1
                if result.is_novel:
                    novel_patterns.append(result.template_key)

        # Determine the dominant template
        if classified:
            from collections import Counter
            primary = Counter(c.template_key for c in classified).most_common(1)[0][0]
        else:
            primary = "no_logs_classified"

        return PerceptionResult(
            classified_logs=classified,
            primary_template=primary,
            l1_hits=l1_hits,
            l2_hits=l2_hits,
            l3_hits=l3_hits,
            novel_patterns=novel_patterns,
        )

    @property
    def l1_stats(self) -> dict:
        """Return L1 cache statistics."""
        return self._l1_cache.stats

    @property
    def l2_stats(self) -> dict:
        """Return L2 matcher statistics."""
        return self._l2_matcher.stats
