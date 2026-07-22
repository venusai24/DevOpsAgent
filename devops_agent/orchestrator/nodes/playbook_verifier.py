import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send

from devops_agent.core.llm_provider import LLMFactory
from devops_agent.playbooks.registry import PLAYBOOK_REGISTRY

from ..agents.schemas import MatchResult, PlaybookVerdict
from ..state import InvestigationState

VERIFIER_PROMPT = """You are reviewing whether the "{playbook_name}" pattern explains this incident. 
Your default assumption is that it does NOT apply. Try to REFUTE it first using the evidence below; 
only mark it confirmed if the disconfirming signals are absent and the evidence requirements are met.

Known causal patterns:
{causal_patterns}

Signals that would DISCONFIRM this scenario (check these explicitly):
{disconfirming_signals}

Minimum evidence required to confirm:
{evidence_requirements}

EVALUATED SYSTEM FACTS:
{semantic_facts}

Observed evidence for this incident (from ContextAssembler):
{evidence_slice}

Respond with confirmed | refuted | inconclusive, a short rationale, and the specific evidence you relied on. 
If a substantially better explanation exists outside this playbook, say so and mark refuted -- do not force a fit."""

def route_to_verifiers(state: InvestigationState) -> list[Send] | str:
    """Routes candidates to verifiers or falls back to natural intelligence."""
    SIGNAL_FLOOR = 0.25 # Lowered slightly for broader recall at verifier stage
    candidates = [c for c in state.get("match_results", []) if c.fired and c.signal_strength >= SIGNAL_FLOOR]
    
    if not candidates:
        return "natural_intelligence_triage"
        
    return [Send("verify_playbook", {"candidate": c, "context": state.get("discovered_kpi_map", {})}) 
            for c in candidates]

async def verify_playbook_node(state: dict[str, Any]) -> dict[str, Any]:
    """Node that actually verifies a single candidate."""
    candidate: MatchResult = state["candidate"]
    context_slice = state["context"]
    
    playbook = PLAYBOOK_REGISTRY.get(candidate.scenario_id)
    if not playbook:
        return {"playbook_verdicts": [PlaybookVerdict(
            scenario_id=candidate.scenario_id, 
            status="refuted", 
            rationale="Playbook not found in registry."
        )]}
        
    facts = state.get("semantic_facts", [])
    facts_str = "\\n".join(f"- {f.semantic_statement}" for f in facts) if facts else "No semantic facts evaluated."
        
    prompt = VERIFIER_PROMPT.format(
        playbook_name=playbook.display_name,
        causal_patterns="\\n".join(f"- {p}" for p in playbook.causal_patterns),
        disconfirming_signals="\\n".join(f"- {p}" for p in playbook.disconfirming_signals),
        evidence_requirements="\\n".join(f"- {p}" for p in playbook.evidence_requirements),
        semantic_facts=facts_str,
        evidence_slice=json.dumps(context_slice, indent=2)
    )
    
    llm = LLMFactory.get_llm("verifier").with_structured_output(PlaybookVerdict)
    try:
        verdict = await llm.ainvoke([SystemMessage(content="You are a strict falsification-focused diagnostic reviewer."), HumanMessage(content=prompt)])
    except Exception as e:
        verdict = PlaybookVerdict(scenario_id=candidate.scenario_id, status="inconclusive", rationale=f"LLM Error: {e}")
        
    # Ensure scenario_id is set correctly in case LLM missed it
    verdict.scenario_id = getattr(candidate, "scenario_id", "")
    
    return {"playbook_verdicts": [verdict]}

def resolve_verdicts(state: InvestigationState) -> str:
    """Routes after verifiers complete."""
    confirmed = [v for v in state.get("playbook_verdicts", []) if v.status == "confirmed"]
    if not confirmed:
        return "natural_intelligence_triage"
        
    # Simple coverage heuristic: if we have 1 confirmed, playbook triage. If >1, hybrid triage.
    if len(confirmed) == 1:
        return "playbook_triage"
    else:
        return "hybrid_triage"
