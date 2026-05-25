# AIRS: Autonomous Incident Response System

## Project Overview
AIRS is a production-ready, agentic DevOps platform that automatically triages, investigates, and remediates incoming system alerts (e.g., from PagerDuty). It utilizes a 14-node LangGraph hybrid-memory architecture to combine deterministic logic with LLM-based reasoning, ensuring zero-trust safe execution.

## Key Features
- **Neuro-Symbolic Routing:** Dynamically routes incidents to fast symbolic paths (Case-Based Reasoning/EKG) or deep ReAct neural loops based on log classification.
- **Enterprise Knowledge Graph (EKG):** Evaluates dependency chains to preemptively calculate the blast radius of remediation plans.
- **Safe Execution Guardrails:** Includes Policy-as-Code validation, automated canary deployments, rollback controllers, and Human-in-the-Loop (HITL) approvals.
- **Continuous Learning:** Retains successful remediation outcomes into the CBR database.
- **Crash-Safe Checkpointing:** Resumes seamlessly from interruptions via PostgreSQL or SQLite state persistence.

## Setup & Run Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Configure `.env` with required credentials (e.g., `GROQ_API_KEY`).
3. Run the mock enterprise server (for local testing):
   `uvicorn mock_enterprise.api:app --port 8000 --reload`
4. In another terminal, trigger the CLI interface:
   `python cli/main.py run` (Use `--incident <file>` for custom alerts)
5. Alternatively, run the FastAPI backend for webhook ingestion:
   `uvicorn api.main:app --port 8080`
   
## High-Level Architecture Summary
AIRS ingests alerts via FastAPI or CLI and hands them to a Celery background worker. The core workflow runs as a LangGraph state machine. It begins with triage and tiered log classification, branching into EKG topology traversal and Case-Based Reasoning to hypothesize root causes. A deterministic Logic Agent prunes impossible hypotheses before an LLM formulates a remediation plan. The plan is subjected to blast-radius analysis and Policy-as-Code checks. Finally, execution pauses for an operator's HITL approval before executing via a canary or direct deployment.
