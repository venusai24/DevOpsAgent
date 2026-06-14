"""
airs_v2 — Hybrid Neuro-Symbolic Autonomous Incident Response System (v2)

This package is the target of the 5-stage incremental rewrite.
It lives alongside the existing `agent/` package during the migration.

Stage Status
------------
Stage 0 — ✅ Scaffold initialised
Stage 1 — ✅ Core contracts & GraphState v2 (ContextGraph, MCP server)
Stage 2 — ✅ Neuro-Symbolic Perception (L1/L2/L3 router) & Reasoning (causal engine)
Stage 3 — ✅ Reasoning Engine — HypothesisEngine + SymbolicValidator (5 rules)
Stage 4 — ✅ Execution Engine — PolicyEnvelope + SlackGateway + HealthMonitor + rollback
Stage 5 — 🔄 RAG Memory + HITL Feedback + Continuous Learning Loop
"""
__version__ = "0.1.0-rag-hitl"
