# AIRS User Manual

Welcome to the detailed User Manual for the Autonomous Incident Response System (AIRS). This guide explains how SRE and DevOps teams can operate, customize, and extend AIRS in a production environment.

## 1. How AIRS Works (The Operator's Perspective)

AIRS acts as a highly skilled Level-1 / Level-2 virtual SRE. It does not replace humans; rather, it drastically reduces Mean Time to Detect (MTTD) and Mean Time to Investigate (MTTI).

When an alert triggers (e.g. from PagerDuty), AIRS:
1. **Triages**: Extracts service names and assigns an internal Severity (P0-P3).
2. **Investigates**: Fetches live telemetry (Datadog metrics, AWS CloudWatch logs) recursively until it isolates the fault.
3. **Drafts a Plan**: Synthesizes a root cause and proposes actionable `kubectl` or shell steps, along with a `rollback_command`.
4. **Pauses for Human Approval**: Blocks execution and pings the configured Slack channel. **This is where you come in.**

### Your Role
As an operator, your primary interaction with AIRS is in the **Approval Node**. When you see a Slack Block Kit message from AIRS, review the Root Cause and the Proposed Remediation Steps. 
- If the plan is safe, click **Approve**. AIRS will execute the Kubernetes commands and post a final postmortem.
- If the plan is flawed or high-risk, click **Reject**. AIRS will safely abort, but the state is preserved for forensic analysis.

## 2. Configuration Settings

AIRS uses `pydantic-settings` to manage environment configurations securely. Configure your `.env` file to customize integrations.

| Variable Name | Description | Default / Example |
|---|---|---|
| `GROQ_API_KEY` | Your Groq API key for the underlying `qwen3-32b` inference engine. | *Required* |
| `DATABASE_URL` | PostgreSQL connection string. If provided, AIRS scales horizontally across Celery workers. | `postgresql://airs:pass@localhost/db` |
| `DATADOG_API_KEY` | Enables live metric queries instead of mock endpoints. | *Optional* |
| `SLACK_BOT_TOKEN` | Enables Slack ChatOps notifications and interactive approvals. | *Optional* |
| `KUBECONFIG_PATH` | Path to `kubeconfig` to allow AIRS to execute `kubectl` SDK commands. | `/root/.kube/config` |

## 3. Crash-Safe Resumption (CLI Mode)

If you are running the CLI and your terminal crashes while waiting for an approval, you **do not need to restart the investigation**. AIRS leverages durable state.

1. Find the `thread_id` that was printed to the console (e.g., `4bf92f3577b34da6`).
2. Resume the exact state machine:
   ```bash
   airs resume 4bf92f3577b34da6 --approve
   ```

## 4. Security & Guardrails

We know running AI against production clusters is scary. AIRS implements **Zero-Trust Guardrails** exclusively via deterministic Python (not LLM system prompts):
- **Missing Rollbacks**: If the agent's drafted plan does not contain a rollback command, the plan is classified as `is_high_risk=True`. The system will refuse to execute it automatically and explicitly flags it in Slack.
- **Destructive Command Blocking**: All commands proposed by the agent are passed through regex-based AST filters. Commands containing `rm -r`, `drop table`, `delete namespace`, etc. are stripped and blocked before reaching the Kubernetes SDK.

## 5. Adding New Integrations

Want to add New Relic instead of Datadog?
1. Create a wrapper inside `agent/integrations/newrelic.py`.
2. Update `agent/tools.py` inside the `get_metrics` tool to branch out to your new wrapper based on `.env` configuration.
3. The LLM prompt and Graph routing will automatically adapt to the new tool output, provided it is formatted as a Markdown summary.
