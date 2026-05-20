ENGINEERING SPECIFICATION & IMPLEMENTATION RFC
Project: Autonomous Incident Response System (AIRS) MVP
Author: Principal AI Systems Architect
Target Audience: Engineering Leadership, Technical Recruiters
Objective: Maximum architectural signal and enterprise realism within a strict 24-hour implementation constraint.

SECTION 1 — FINAL MVP DEFINITION
Exact MVP Scope:
A stateful, multi-agent AI system that ingests simulated PagerDuty alerts, autonomously investigates mock telemetry (metrics/logs), synthesizes a root cause hypothesis, proposes a remediation plan, enforces a durable human-in-the-loop (HITL) approval pause, and generates a postmortem.

Exact User Workflow:

A local script pushes an alert JSON payload to the system.

The user watches a rich terminal interface stream the agents' internal thoughts and tool calls in real-time.

The system explicitly pauses execution, presenting a generated remediation plan.

The user approves or rejects the plan via the terminal.

The system resumes execution, applies the mock fix, and writes a postmortem document.

Exact Demo Workflow:
To demonstrate extreme architectural maturity, the demo will feature a "Server Crash Test". The user will trigger an incident, let the graph run to the HITL approval node, and then intentionally kill the terminal script. The user will restart the script, pass an approval command using the original thread ID, and the system will resume instantly from the exact paused state without re-running prior nodes.  

Exact Non-Goals:
No React/Next.js frontend. No production databases (PostgreSQL/Redis). No vector databases (Chroma/Pinecone). No live cloud infrastructure deployments (AWS/Kubernetes).

Exact Mocked Systems:
Datadog/Prometheus (Metrics), Splunk/CloudWatch (Logs), Kubernetes (Execution). These will be simulated via a local Python FastAPI server returning static, pre-defined JSON payloads.

Exact Real Systems:
LangGraph (Orchestration), Google Gemini (Inference), AsyncSqliteSaver (Durable State Checkpointing), Pytest (Evaluation).

Exact Architecture Constraints:

All LLM outputs must be coerced into strictly typed Pydantic models.

System prompts must never be used to ensure safety; destructive operations must be gated by deterministic Python rules.

Loops must be strictly bounded with a retry_count state variable to prevent runaway API spend.  

Exact Success Criteria:
A fully automated pytest suite running an "LLM-as-a-judge" evaluation that proves the agent accurately deduces the root cause from the mock telemetry 80% of the time.  

SECTION 2 — FINAL TECH STACK
Backend & Mock Environment: FastAPI

Why: Async-native by default. Maps perfectly to LangGraph's asynchronous execution methods (astream_events).

Tradeoff: Slightly heavier than standard http.server, but provides instant OpenAPI validation for the mocked endpoints.

Agent Framework: LangGraph

Why: Incident response is cyclical (requiring loops and self-correction), not linear. Provides low-level control, native state checkpointing, and interrupt() capabilities.  

Rejected: CrewAI, AutoGen (Too abstracted, lack programmatic checkpointing and rigorous evaluation hooks).

State Management: AsyncSqliteSaver

Why: Provides the exact same durable execution capabilities as PostgreSQL without requiring Docker or network configuration.

LLM Provider: Gemini 3 Google AI Studio LLMs

Why: Currently leads the industry in native tool-calling reliability and strict JSON adherence for complex multi-step reasoning.

Frontend Interface: Typer & Rich (Python CLI)

Why: Building a web UI wastes 8 hours. A CLI built with Rich beautifully renders the agent's reasoning traces, JSON payloads, and pause states directly in the terminal, maximizing technical signal.  

Observability & Evals: LangSmith & Pytest

Why: LangSmith provides out-of-the-box trace visualization. Native integration with pytest via the @test decorator allows for programmatic LLM-as-a-judge evaluations.  

Exact Dependency List (requirements.txt)
langgraph>=0.2.0
langchain-google-genai>=0.1.0
langgraph-checkpoint-sqlite>=1.0.0
fastapi>=0.111.0
uvicorn>=0.30.0
pydantic>=2.7.0
typer>=0.12.0
rich>=13.7.0
pytest>=8.2.0
langsmith>=0.1.0
httpx>=0.27.0

Local Setup Instructions:

Bash
python -m venv.venv
source.venv/bin/activate
pip install -r requirements.txt
cp.env.example.env
# Edit.env with GOOGLE_API_KEY and LANGSMITH_API_KEY
SECTION 3 — SYSTEM ARCHITECTURE
High-Level Architecture:
The system is divided by a network boundary. On port 8000, a FastAPI instance simulates the enterprise (Datadog/Logs/K8s). On the command line, the LangGraph process runs, making HTTP calls via Python's httpx to the FastAPI instance.

LangGraph Architecture:
A directed cyclic graph. Nodes are pure Python functions that accept the GraphState and return updates. State is persisted to a local .sqlite file using AsyncSqliteSaver.

State Management Architecture:
A single Pydantic schema representing the MessagesState plus domain-specific typed fields (incident_severity, telemetry_context, remediation_plan, retry_count, is_approved).

Error Handling Architecture:
If an LLM hallucinates a tool argument and the FastAPI mock returns 422 Unprocessable Entity, the tool function intercepts the HTTP error, formats it as a ToolMessage, and injects it back into the graph state. This forces the LLM to read the error and self-correct on the next iteration.

SECTION 4 — FOLDER STRUCTURE
airs-mvp/
├──.env.example
├── requirements.txt
├── README.md
├── mock_enterprise/
│   ├── api.py            # FastAPI application simulating Datadog/Splunk
│   └── fixtures.json     # Hardcoded incident scenarios (CPU spike, DB exhaustion)
├── agent/
│   ├── init.py
│   ├── state.py          # Pydantic schemas for GraphState and all typed outputs
│   ├── tools.py          # Python functions decorated with @tool to hit the mock API
│   ├── nodes.py          # Core logic for Triage, Investigate, RCA, and Plan agents
│   └── orchestrator.py   # StateGraph compilation, edge routing, and SqliteSaver config
├── cli/
│   └── main.py           # Typer application and Rich terminal console streaming
└── tests/
├── conftest.py       # Pytest fixtures to initialize the graph
└── test_evals.py     # LLM-as-a-judge tests asserting RCA accuracy
Why this structure: Strict separation of concerns (Clean Architecture). The mock enterprise is isolated from the agent logic, proving that the agent interacts purely via network boundaries, mimicking a real SRE environment.  

SECTION 5 — LANGGRAPH WORKFLOW DESIGN
State Schema:

Python
class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    raw_alert: dict
    severity: str
    telemetry: str
    plan: str
    retry_count: int
Graph Nodes:

triage_node: Uses structured output to classify severity.

investigate_node: Binds get_metrics and get_logs tools. Appends to telemetry.

rca_node: Synthesizes data into a root cause string.

plan_node: Drafts remediation.

approval_node: Triggers interrupt().

execute_node: Generates postmortem.

Conditional Routing & Edges:

triage_node -> router: if severity == "P0": route to escalate_node (Zero-trust rule).

investigate_node -> router: if retry_count > 3: route to rca_node (Prevents infinite loops).

plan_node -> approval_node (Deterministic edge, no AI decision).

approval_node -> execute_node (Only reachable via human Command(resume=...)).

SECTION 6 — AGENT SPECIFICATIONS
1. Triage Agent

Responsibility: Initial severity classification.

Output Schema: Pydantic TriageResult(severity: Literal["P0", "P1", "P2", "P3"], service: str)

Tools: None.

2. Investigation Agent

Responsibility: ReAct loop fetching telemetry.

Prompt Strategy: "You are an SRE investigator. Use metrics to find anomalies, then fetch logs for the exact timestamp."

Tools: get_metrics(query, time_range), get_logs(service, time_range).

3. RCA & Recommendation Agent

Responsibility: Synthesize data and draft fix.

Output Schema: Pydantic RemediationPlan(root_cause: str, steps: list[str], rollback_command: str)

Guardrails: If rollback_command is empty, code explicitly flags plan as high-risk.

4. Human Approval Agent

Responsibility: Suspend graph execution.

Implementation: Pure python: interrupt({"plan": state["plan"], "message": "Approve execution?"}).

SECTION 7 — TOOLING + MOCK SYSTEMS
Mock Monitoring APIs (FastAPI):

GET /api/v1/metrics?query={promql}: Returns simulated CPU and DB connection metrics.

GET /api/v1/logs?service={service}: Returns a seeded stack trace (e.g., TimeoutError: Connection pool exhausted).

Tool Interfaces (agent/tools.py):
Use Python's httpx library inside @tool decorated functions to call the local FastAPI server. The LLM only sees the tool docstrings, forcing it to deduce the correct parameters.

SECTION 8 — API DESIGN
FastAPI Routes (mock_enterprise/api.py):

Python
@app.get("/metrics")
def get_metrics(query: str):
    # Returns 400 if query is malformed, forcing LLM self-correction
    # Returns JSON payload of mock metrics if valid
Python
@app.get("/logs")
def get_logs(service: str):
    # Returns JSON array of log strings
SECTION 9 — FRONTEND DESIGN (TERMINAL UI)
Focus: Rich, streaming terminal interface using Python's Rich library.
Flow:  

User runs python cli/main.py --incident bad-db-connection.json.

Terminal displays a live-updating tree.

[cyan] Tool Call: get_metrics -> {"query": "db_connections"}

[green] LLM Thought: "The connection pool is exhausted. I will draft a restart command."

Terminal clears and renders a Markdown panel showing the RemediationPlan.

Terminal pauses with a standard input prompt: Review plan. Type 'approve' to execute.

SECTION 10 — OBSERVABILITY
Agent Execution Tracing:
LangSmith is initialized via environment variables (LANGSMITH_TRACING=true). The cli/main.py will use LangGraph's astream_events(version="v2") to parse chunked tokens and tool calls, streaming them to the Rich console for local visibility, while LangSmith records the full distributed trace in the cloud.




Note: I am using a conda environment genai to run this project. Install everything in genai environment only and build in that. 