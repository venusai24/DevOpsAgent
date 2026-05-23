"""
agent/reasoning/nesym_router.py

Neuro-symbolic routing decision engine for the AIRS Perception Layer.

Determines which reasoning pathway to use based on the output of the
TieredLogClassifier, the CBR confidence score, and the primary log template.

Three pathways:
  SYMBOLIC_FAST  — All logs classified at L1, known template → deterministic
                   rule-based diagnosis. No LLM call. ~100ms total.
  CBR_GUIDED     — L2 matches + high CBR similarity → retrieve-and-adapt.
                   LLM used only for minor adaptation. ~2s total.
  NEURAL_FULL    — Novel L3 patterns → full LLM-based investigation.
                   Standard ReAct loop. ~10s total.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ReasoningPathway(str, Enum):
    """The three neuro-symbolic reasoning pathways."""
    SYMBOLIC_FAST = "symbolic_fast"   # Pure symbolic, no LLM
    CBR_GUIDED = "cbr_guided"         # CBR retrieval + light LLM adaptation
    NEURAL_FULL = "neural_full"       # Full LLM investigation (ReAct)


@dataclass
class RoutingDecision:
    """Output of the neuro-symbolic router."""
    pathway: ReasoningPathway
    rationale: str
    l1_rate: float       # Fraction of logs classified at L1
    cbr_confidence: float
    primary_template: str

    def to_markdown(self) -> str:
        pathway_icons = {
            ReasoningPathway.SYMBOLIC_FAST: "⚡",
            ReasoningPathway.CBR_GUIDED: "📚",
            ReasoningPathway.NEURAL_FULL: "🧠",
        }
        icon = pathway_icons.get(self.pathway, "▶")
        return (
            f"**{icon} Reasoning Pathway**: `{self.pathway.value.upper()}`\n"
            f"- Primary pattern: `{self.primary_template}`\n"
            f"- L1 hit rate: {self.l1_rate:.0%} | CBR confidence: {self.cbr_confidence:.0%}\n"
            f"- Rationale: {self.rationale}"
        )


class NeuroSymbolicRouter:
    """
    Routes the reasoning process to the optimal pathway based on signal quality.

    Routing Rules (evaluated in order):
      1. If L1 hit rate >= 0.9 AND template is known → SYMBOLIC_FAST
         (Virtually all logs matched known patterns — no novel signals)
      2. If CBR confidence >= 0.75 → CBR_GUIDED
         (Strong historical precedent found — use proven solution)
      3. Otherwise → NEURAL_FULL
         (Novel or ambiguous pattern — full LLM investigation needed)
    """

    # Thresholds
    SYMBOLIC_L1_THRESHOLD: float = 0.9
    CBR_GUIDED_THRESHOLD: float = 0.75

    def route(
        self,
        perception_stats: dict[str, Any],
        cbr_confidence: float,
        primary_template: str,
    ) -> RoutingDecision:
        """
        Determine the optimal reasoning pathway.

        Args:
            perception_stats: Output of TieredLogClassifier.classify_telemetry_block().
                              Expected keys: L1_hits, L2_hits, L3_hits, total.
            cbr_confidence: Cosine similarity of the best CBR case match (0.0–1.0).
            primary_template: The dominant log template category.

        Returns:
            RoutingDecision specifying which pathway to use.
        """
        l1_rate = perception_stats.get("l1_rate", 0.0)
        total = perception_stats.get("total", 0)
        l3_hits = perception_stats.get("L3_hits", 0)

        # Rule 1: Symbolic fast path — all known patterns, no novelty
        if l1_rate >= self.SYMBOLIC_L1_THRESHOLD and l3_hits == 0 and total > 0:
            logger.info(
                "[NeSyRouter] SYMBOLIC_FAST: l1_rate=%.0f%% template=%s",
                l1_rate * 100, primary_template,
            )
            return RoutingDecision(
                pathway=ReasoningPathway.SYMBOLIC_FAST,
                rationale=(
                    f"{l1_rate:.0%} of logs matched known L1 templates with zero novel patterns. "
                    f"Applying deterministic symbolic diagnosis for `{primary_template}`."
                ),
                l1_rate=l1_rate,
                cbr_confidence=cbr_confidence,
                primary_template=primary_template,
            )

        # Rule 2: CBR-guided — strong historical precedent
        if cbr_confidence >= self.CBR_GUIDED_THRESHOLD:
            logger.info(
                "[NeSyRouter] CBR_GUIDED: cbr_confidence=%.0f%% template=%s",
                cbr_confidence * 100, primary_template,
            )
            return RoutingDecision(
                pathway=ReasoningPathway.CBR_GUIDED,
                rationale=(
                    f"CBR match confidence {cbr_confidence:.0%} exceeds threshold "
                    f"{self.CBR_GUIDED_THRESHOLD:.0%}. Adapting historical solution "
                    f"for `{primary_template}` without full LLM investigation."
                ),
                l1_rate=l1_rate,
                cbr_confidence=cbr_confidence,
                primary_template=primary_template,
            )

        # Rule 3: Full neural investigation — novel or ambiguous
        logger.info(
            "[NeSyRouter] NEURAL_FULL: l1_rate=%.0f%% cbr=%.0f%% l3_hits=%d",
            l1_rate * 100, cbr_confidence * 100, l3_hits,
        )
        return RoutingDecision(
            pathway=ReasoningPathway.NEURAL_FULL,
            rationale=(
                f"Novel or ambiguous pattern (L1={l1_rate:.0%}, CBR={cbr_confidence:.0%}, "
                f"L3 novel hits={l3_hits}). Engaging full LLM-based investigation."
            ),
            l1_rate=l1_rate,
            cbr_confidence=cbr_confidence,
            primary_template=primary_template,
        )
