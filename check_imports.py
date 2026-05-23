from agent.reasoning.cbr_engine import CBREngine, ScoredCase
from agent.reasoning.incident_store import HistoricalCase, IncidentStore
from agent.action.policy_engine import PolicyEngine, PolicyCheckResult, generate_terraform_patch
from agent.action.blast_radius import BlastRadiusEstimator
from agent.action.canary_controller import CanaryController, CanaryResult
from agent.action.rollback_controller import RollbackController, GoldenSignalSnapshot
from agent.security.guardrails import is_safe_command
from agent.reasoning.knowledge_graph import KnowledgeGraph
print("All imports OK")
# Print available names for debugging
import inspect
print("CBREngine methods:", [m for m in dir(CBREngine) if not m.startswith('_')])
print("ScoredCase fields:", ScoredCase.__dataclass_fields__.keys() if hasattr(ScoredCase, '__dataclass_fields__') else dir(ScoredCase))
print("CanaryResult:", [m for m in dir(CanaryResult) if not m.startswith('_')])
print("GoldenSignalSnapshot fields:", list(GoldenSignalSnapshot.__dataclass_fields__.keys()) if hasattr(GoldenSignalSnapshot, '__dataclass_fields__') else dir(GoldenSignalSnapshot))
print("PolicyCheckResult:", [m for m in dir(PolicyCheckResult) if not m.startswith('_')])
print("is_safe_command type:", type(is_safe_command))
