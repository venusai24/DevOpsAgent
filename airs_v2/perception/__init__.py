"""
airs_v2.perception — NeSy-Edge Three-Tier Log Parsing Router
============================================================

Public surface
--------------
    LogRouter          — Async orchestrator (L1 → L2 → L3 cascade)
    RouterResult       — Typed result of a single log classification
    PerceptionReport   — Aggregated report for a full telemetry block
    Tier               — Enum: L1 | L2 | L3

Internal modules
----------------
    l1_cache.py        — Symbolic regex cache (exact matching, ~0 ms)
    l2_semantic.py     — TF-IDF cosine similarity over the knowledge base (~5-10 ms)
    l3_llm.py          — LLM fallback for novel patterns (~1-2 s, lazy-init)
    router.py          — Cascade orchestrator + PerceptionReport builder
"""
from airs_v2.perception.router import LogRouter, RouterResult, PerceptionReport, Tier

__all__ = ["LogRouter", "RouterResult", "PerceptionReport", "Tier"]
