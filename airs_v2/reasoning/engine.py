"""
airs_v2/reasoning/engine.py
==============================

ReasoningEngine — Neuro-Symbolic Incident Reasoning Orchestrator
-----------------------------------------------------------------

Top-level entry point for Stage 3.  Bridges Stage 1 (PerceptionReport) and
Stage 2 (ContextGraph via MCP) into a ``prior-constrained sparse symbolic
causal graph``.

Two operating modes
-------------------
1. **Direct graph injection** (``ReasoningEngine(graph=g)``):
   Receives a ``ContextGraph`` object directly — used in unit tests to avoid
   spawning MCP server subprocesses.  All topology data is read from the
   injected graph.

2. **MCP transport** (``ReasoningEngine()``):
   Spawns the MCP server as a subprocess via ``ContextMCPClient`` and calls
   four tools: ``get_blast_radius``, ``get_subgraph``, ``get_failure_correlations``,
   and ``get_metrics_snapshot`` (per node in blast radius).
   Used in integration scenarios (end-to-end with subprocess).

Analysis pipeline
-----------------
::

    perception_report  focal_service
            │                │
            └────────┬───────┘
                     ▼
           ① Retrieve topology context
             (blast radius, subgraph, correlations, metrics)
                     │
                     ▼
           ② HypothesisEngine.generate()
             → list[CausalHypothesis]
                     │
                     ▼
           ③ SymbolicValidator.validate_all()
             → (valid_hyps, rejected_hyps)
                     │
                     ▼
           ④ Build CausalGraph from valid hypotheses
                     │
                     ▼
           ⑤ Rank, promote to RootCauseHypothesis[]
                     │
                     ▼
           ⑥ Compute symbolic_path and overall_confidence
                     │
                     ▼
           ⑦ Return IncidentAnalysis
"""

from __future__ import annotations

import logging
from typing import Optional

from airs_v2.context.graph import ContextGraph, NodeSnapshot
from airs_v2.context.client import ContextMCPClient
from airs_v2.perception.router import PerceptionReport, Tier
from airs_v2.reasoning.causal_types import (
    CausalEdge,
    CausalGraph,
    CausalNode,
    IncidentAnalysis,
    RejectedHypothesis,
    RootCauseHypothesis,
)
from airs_v2.reasoning.hypothesis_engine import HypothesisEngine
from airs_v2.reasoning.symbolic_validator import SymbolicValidator, TopologyContext
from airs_v2.evaluation.tracer import ObservabilityTracer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ReasoningEngine
# ---------------------------------------------------------------------------


class ReasoningEngine:
    """
    Neuro-symbolic incident reasoning engine.

    Parameters
    ----------
    graph:
        Optional pre-seeded ``ContextGraph`` for test injection.  When set,
        all topology data is read from this object directly — no MCP subprocess
        is spawned.  When ``None`` (default), the engine uses ``ContextMCPClient``
        to communicate with the running MCP server.
    """

    def __init__(self, graph: Optional[ContextGraph] = None) -> None:
        self._injected_graph = graph

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze_incident(
        self,
        perception_report: PerceptionReport,
        focal_service: str,
    ) -> IncidentAnalysis:
        """
        Perform a full neuro-symbolic causal analysis of an incident.

        Parameters
        ----------
        perception_report:
            Output of the Stage 1 ``LogRouter.classify_block()`` call
            containing all classified log events.
        focal_service:
            The infrastructure service at the centre of the incident
            (the first service reporting symptoms).

        Returns
        -------
        IncidentAnalysis
            Fully validated causal graph, ranked root-cause candidates,
            rejected hypotheses, and LLM-consumable markdown summary.
        """
        logger.info(
            "[ReasoningEngine] BEGIN focal=%s perception_events=%d",
            focal_service,
            len(perception_report.results),
        )

        tracer = ObservabilityTracer.get_instance()
        tracer.update_phase("Observation")

        # ── ① Retrieve topology context ───────────────────────────────
        if self._injected_graph is not None:
            ctx = self._build_context_from_graph(focal_service)
            correlation_partners = self._correlation_partners_from_graph(focal_service)
        else:
            ctx, correlation_partners = await self._fetch_context_via_mcp(focal_service)

        # ── ② Generate raw hypotheses ─────────────────────────────────
        tracer.update_phase("Hypothesis Generation")
        hyp_engine = HypothesisEngine(context=ctx, correlation_partners=correlation_partners)
        raw_hypotheses = hyp_engine.generate(perception_report)
        
        tracer.record_hypotheses_generated(
            [f"{h.candidate_node} ({h.source})" for h in raw_hypotheses]
        )

        logger.info(
            "[ReasoningEngine] generated %d raw hypotheses for focal=%s",
            len(raw_hypotheses),
            focal_service,
        )

        # ── ③ Symbolic validation (reject physically impossible) ──────
        validator = SymbolicValidator(ctx)
        valid_hyps, rejected_hyps = validator.validate_all(raw_hypotheses)

        logger.info(
            "[ReasoningEngine] valid=%d rejected=%d",
            len(valid_hyps),
            len(rejected_hyps),
        )

        # ── ④ Build sparse causal graph ───────────────────────────────
        causal_graph = self._build_causal_graph(
            focal_service=focal_service,
            valid_hyps=valid_hyps,
            context=ctx,
        )

        # ── ⑤ Rank and promote to RootCauseHypothesis ─────────────────
        sorted_hyps = sorted(valid_hyps, key=lambda h: h.raw_score, reverse=True)
        root_cause_candidates: list[RootCauseHypothesis] = []
        for rank, h in enumerate(sorted_hyps, start=1):
            rch = validator.build_root_cause(h, rank=rank)
            root_cause_candidates.append(rch)

        # ── ⑥ Compute meta-fields ─────────────────────────────────────
        symbolic_path = self._determine_symbolic_path(perception_report)
        overall_confidence = self._compute_overall_confidence(root_cause_candidates)

        # ── ⑦ Assemble IncidentAnalysis ───────────────────────────────
        rejected_models = rejected_hyps  # already RejectedHypothesis objects

        # Build markdown summary
        analysis_md = self._render_analysis_markdown(
            focal_service=focal_service,
            report=perception_report,
            causal_graph=causal_graph,
            root_cause_candidates=root_cause_candidates,
            rejected=rejected_models,
            symbolic_path=symbolic_path,
            confidence=overall_confidence,
        )

        analysis = IncidentAnalysis(
            focal_service=focal_service,
            causal_graph=causal_graph,
            root_cause_candidates=root_cause_candidates,
            rejected_hypotheses=rejected_models,
            symbolic_path=symbolic_path,
            overall_confidence=round(overall_confidence, 4),
            analysis_markdown=analysis_md,
        )

        final_conclusion = (
            f"{root_cause_candidates[0].candidate_node} due to {root_cause_candidates[0].template_key}"
            if root_cause_candidates
            else "Unknown Root Cause"
        )
        tracer.conclude_diagnosis(final_conclusion)

        logger.info(
            "[ReasoningEngine] DONE focal=%s candidates=%d rejected=%d "
            "confidence=%.2f path=%s",
            focal_service,
            len(root_cause_candidates),
            len(rejected_models),
            overall_confidence,
            symbolic_path,
        )
        return analysis

    # ------------------------------------------------------------------
    # Topology context builders
    # ------------------------------------------------------------------

    def _build_context_from_graph(self, focal_service: str) -> TopologyContext:
        """Build TopologyContext from the injected ContextGraph (test mode)."""
        g = self._injected_graph
        br = g.blast_radius(focal_service)

        # Build node attribute maps
        health_by_node: dict[str, str] = {}
        metrics_by_node: dict[str, float] = {}
        on_call_by_node: dict[str, list[str]] = {}

        blast_set = set(br.affected_services)

        for name in blast_set | {focal_service}:
            snap = g.get_node(name)
            if snap:
                health_by_node[name] = snap.health_status
                on_call_by_node[name] = [snap.on_call] if snap.on_call else []

            metrics = g.get_metrics_snapshot(name)
            if metrics:
                metrics_by_node[name] = metrics.connection_saturation

        return TopologyContext(
            graph=g,
            focal_service=focal_service,
            blast_radius_nodes=blast_set,
            health_by_node=health_by_node,
            metrics_by_node=metrics_by_node,
            on_call_by_node=on_call_by_node,
        )

    def _correlation_partners_from_graph(self, focal_service: str) -> set[str]:
        """Build correlation partner set from the injected graph (test mode)."""
        g = self._injected_graph
        records = g.get_failure_correlations(focal_service)
        return {r.partner_service for r in records}

    async def _fetch_context_via_mcp(
        self, focal_service: str
    ) -> tuple[TopologyContext, set[str]]:
        """Fetch topology context by calling the MCP server via subprocess."""
        async with ContextMCPClient() as client:
            # 1. Blast radius
            br_data = await client.call("get_blast_radius", service=focal_service)
            blast_set: set[str] = set(br_data.get("affected_services", []))

            # 2. Subgraph (for node metadata)
            sg_data = await client.call("get_subgraph", service=focal_service)
            nodes_raw = sg_data.get("nodes", [])

            # 3. Failure correlations
            corr_data = await client.call(
                "get_failure_correlations", service=focal_service
            )
            partners: set[str] = {r["partner_service"] for r in corr_data}

            # 4. Metrics per node in blast radius
            health_by_node: dict[str, str] = {}
            metrics_by_node: dict[str, float] = {}
            on_call_by_node: dict[str, list[str]] = {}

            for n in nodes_raw:
                name = n["name"]
                health_by_node[name] = n.get("health_status", "unknown")
                on_call_val = n.get("on_call")
                on_call_by_node[name] = [on_call_val] if on_call_val else []

                try:
                    m = await client.call("get_metrics_snapshot", service=name)
                    metrics_by_node[name] = m.get("connection_saturation", 0.0)
                except Exception:
                    metrics_by_node[name] = 0.0

        # We still need a ContextGraph for path calculations.
        # In MCP mode, build a fresh one with default fixtures.
        from airs_v2.context.graph import ContextGraph
        g = ContextGraph()

        ctx = TopologyContext(
            graph=g,
            focal_service=focal_service,
            blast_radius_nodes=blast_set,
            health_by_node=health_by_node,
            metrics_by_node=metrics_by_node,
            on_call_by_node=on_call_by_node,
        )
        
        tracer = ObservabilityTracer.get_instance()
        metrics_analyzed = []
        for n, h in health_by_node.items():
            metrics_analyzed.append(f"{n}_health_status")
        for n, m in metrics_by_node.items():
            metrics_analyzed.append(f"{n}_connection_saturation")
        tracer.record_metrics_analyzed(metrics_analyzed)
        
        return ctx, partners

    # ------------------------------------------------------------------
    # Causal graph builder
    # ------------------------------------------------------------------

    def _build_causal_graph(
        self,
        focal_service: str,
        valid_hyps: list,
        context: TopologyContext,
    ) -> CausalGraph:
        """Assemble a ``CausalGraph`` from validated hypotheses."""
        nodes_by_name: dict[str, CausalNode] = {}
        edges: list[CausalEdge] = []

        # Add focal service as symptom node
        focal_snap = context.graph.get_node(focal_service)
        nodes_by_name[focal_service] = CausalNode(
            name=focal_service,
            role="symptom",
            health_status=context.health_by_node.get(focal_service, "unknown"),
            tier=focal_snap.tier if focal_snap else 3,
            evidence_count=0,
        )

        for h in valid_hyps:
            candidate = h.candidate_node
            snap = context.graph.get_node(candidate)

            # Determine role
            if candidate == focal_service:
                role = "symptom"  # self-fault upgrades the node role
            else:
                role = "root_cause" if h is valid_hyps[0] or h.raw_score >= 0.70 else "propagation"

            # Upsert node (keep root_cause role if already set)
            existing = nodes_by_name.get(candidate)
            if existing is None or (role == "root_cause" and existing.role != "root_cause"):
                nodes_by_name[candidate] = CausalNode(
                    name=candidate,
                    role=role,
                    health_status=context.health_by_node.get(candidate, "unknown"),
                    tier=snap.tier if snap else 3,
                    evidence_count=len(h.evidence),
                )

            # Add intermediate path nodes
            path = h.causal_path
            for node_name in path[1:-1]:  # exclude start (candidate) and end (focal)
                if node_name not in nodes_by_name:
                    intermediate_snap = context.graph.get_node(node_name)
                    nodes_by_name[node_name] = CausalNode(
                        name=node_name,
                        role="propagation",
                        health_status=context.health_by_node.get(node_name, "unknown"),
                        tier=intermediate_snap.tier if intermediate_snap else 3,
                        evidence_count=0,
                    )

            # Add causal edges along the path
            path_len = len(path)
            if path_len >= 2:
                mechanism = "depends_on" if path_len == 2 else "cascade"
                for i in range(path_len - 1):
                    src, tgt = path[i], path[i + 1]
                    edge_key = (src, tgt)
                    if not any(e.source == src and e.target == tgt for e in edges):
                        edges.append(
                            CausalEdge(
                                source=src,
                                target=tgt,
                                mechanism=mechanism,
                                strength=round(min(h.raw_score, 1.0), 3),
                            )
                        )

        return CausalGraph(
            focal_service=focal_service,
            nodes=list(nodes_by_name.values()),
            edges=edges,
        )

    # ------------------------------------------------------------------
    # Meta-field computation
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_symbolic_path(report: PerceptionReport) -> str:
        """
        Classify the symbolic path based on which perception tiers fired.

        SYMBOLIC_FAST  — all L1 hits, no LLM needed
        CBR_GUIDED     — mixed L1/L2, historical correlations used
        NEURAL_FULL    — one or more L3 novel patterns processed
        """
        stats = report.tier_stats
        if stats.get("L3_hits", 0) > 0:
            return "NEURAL_FULL"
        if stats.get("L2_hits", 0) > 0:
            return "CBR_GUIDED"
        return "SYMBOLIC_FAST"

    @staticmethod
    def _compute_overall_confidence(
        candidates: list[RootCauseHypothesis],
    ) -> float:
        """Weighted average confidence of the top-3 root-cause candidates."""
        top3 = candidates[:3]
        if not top3:
            return 0.0
        weights = [1.0, 0.5, 0.25][: len(top3)]
        total = sum(c.final_score * w for c, w in zip(top3, weights))
        denom = sum(weights[: len(top3)])
        return round(total / denom, 4) if denom else 0.0

    # ------------------------------------------------------------------
    # Markdown generation
    # ------------------------------------------------------------------

    @staticmethod
    def _render_analysis_markdown(
        focal_service: str,
        report: PerceptionReport,
        causal_graph: CausalGraph,
        root_cause_candidates: list[RootCauseHypothesis],
        rejected: list[RejectedHypothesis],
        symbolic_path: str,
        confidence: float,
    ) -> str:
        path_emoji = {
            "SYMBOLIC_FAST": "🔵",
            "CBR_GUIDED": "🟡",
            "NEURAL_FULL": "🟣",
        }.get(symbolic_path, "⚪")

        lines = [
            f"# Incident Analysis — `{focal_service}`",
            "",
            f"**Symbolic Path**: {path_emoji} `{symbolic_path}`  "
            f"| **Overall Confidence**: {confidence:.0%}",
            "",
            "## Perception Summary",
            report.to_markdown(),
            "",
            "## Causal Graph",
            causal_graph.to_markdown(),
            "",
            "## Root Cause Candidates",
        ]

        if root_cause_candidates:
            for rch in root_cause_candidates:
                path_str = " → ".join(rch.causal_path) or rch.candidate_node
                contacts = ", ".join(rch.on_call_contacts) if rch.on_call_contacts else "—"
                stale_warn = " ⚠️ *stale metrics*" if rch.stale_metrics_warning else ""
                lines += [
                    f"### #{rch.rank} `{rch.candidate_node}` "
                    f"(score={rch.final_score:.2f}){stale_warn}",
                    f"- **Template**: `{rch.template_key}`",
                    f"- **Propagation**: `{path_str}` [{rch.propagation_mechanism}]",
                    f"- **On-call**: {contacts}",
                    "",
                ]
        else:
            lines.append("*No valid root-cause candidates survived symbolic validation.*")

        if rejected:
            lines += [
                "",
                "## Rejected Hypotheses",
                f"*{len(rejected)} candidates pruned by SymbolicValidator:*",
            ]
            for r in rejected:
                lines.append(
                    f"- ❌ `{r.hypothesis.candidate_node}` "
                    f"[{r.violated_rule}]: {r.reason}"
                )

        return "\n".join(lines)
