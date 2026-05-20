<div align="center">
  <h1> Autonomous Incident Response System (AIRS)</h1>
  <p><strong>AI-driven incident triage, investigation, and safe remediation for Site Reliability Engineering (SRE).</strong></p>
</div>

---

## Overview

**AIRS** is a stateful, multi-agent AI system designed to handle the heavy lifting of incident response. It ingests alerts, investigates telemetry (metrics/logs), extracts verbatim identifiers into an XML scratchpad (mitigating LLM hallucination), synthesizes a root cause hypothesis, proposes a remediation plan, enforces a durable human-in-the-loop (HITL) pause, and safely executes fixes upon approval.

AIRS can run in two modes:
1. **Mock CLI Mode**: A lightweight, terminal-based simulation using mocked endpoints. Perfect for testing and demonstrating agent logic.
2. **Production Mode**: A fully scalable enterprise setup using Celery workers, PostgreSQL state tracking, real Datadog/CloudWatch API integrations, and Slack ChatOps for approvals.

---

## Quick Start (CLI Demo Mode)

To explore the agent's logic without external dependencies, use the CLI demo mode.

### Prerequisites
1. Python 3.10+
2. A Groq API key (`GROQ_API_KEY`) or Google AI Studio key (if you change the LLM provider in code).

### Setup
```bash
# Clone the repository and navigate into the directory
git clone https://github.com/venusai24/airs.git
cd airs

# Create and activate a virtual environment (e.g., using conda)
conda create -n genai python=3.11
conda activate genai

# Install dependencies
pip install -e .
```

### Running the Demo
1. **Start the Mock API server**:
   In your first terminal, launch the mock Datadog and Splunk endpoints:
   ```bash
   uvicorn mock_enterprise.api:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Trigger the CLI Agent**:
   In a second terminal, trigger an incident investigation:
   ```bash
   airs run
   ```
   *Note: This automatically uses a built-in demo payload for Database Connection Pool Exhaustion.*

3. **Approve the Fix**:
   Watch the agent's thoughts stream in real-time. When it pauses and presents a remediation plan, type `approve` or `reject`.

---

## Running in Production Mode

Production mode swaps mock APIs for live SDKs, SQLite for PostgreSQL, and the CLI terminal for Slack ChatOps.

### Setup
1. **Configure your `.env` file**:
   ```env
   # Database & Queues
   DATABASE_URL=postgresql://airs:airs_password@postgres:5432/airs_db
   CELERY_BROKER_URL=redis://redis:6379/0
   CELERY_RESULT_BACKEND=redis://redis:6379/0

   # LLM
   GROQ_API_KEY=your_api_key

   # External Tools (Optional: enables live data)
   DATADOG_API_KEY=your_datadog_key
   SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
   SLACK_CHANNEL_ID=C01XXXXXX
   KUBECONFIG_PATH=/path/to/.kube/config
   ```

2. **Spin up the Cluster**:
   ```bash
   docker-compose up -d
   ```

3. **Trigger via Webhook**:
   Send a PagerDuty payload to the ingestion endpoint:
   ```bash
   curl -X POST http://localhost:8080/webhooks/pagerduty \
        -H "Content-Type: application/json" \
        -d @mock_enterprise/fixtures.json
   ```

You will receive the generated remediation plan in your configured Slack channel, where you can securely approve or reject the execution.

---

## Documentation

For an in-depth understanding of AIRS configurations, architectures, and guardrails, please see the extended documentation:

- [User Manual](docs/USER_MANUAL.md) — Comprehensive guide to operating and configuring AIRS.
- [Architecture Details](docs/ARCHITECTURE.md) — Information on state tracking, agent nodes, and the integration adapter pattern.
