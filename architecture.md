# AIRS Production Architecture

This document details the actual runtime architecture, module responsibilities, control flow, and data flow of the Autonomous Incident Response System (AIRS) based on the production codebase.

---

## 1. System Topology (Mermaid Diagram)

```mermaid
graph TD
    %% Ingestion
    PagerDuty["PagerDuty Webhook / CLI"] -->|Raw JSON Alert| API["FastAPI Endpoint (api/main.py)"]
    API -->|Broker: Redis/Celery| Worker["Celery Worker (worker/tasks.py)"]
    
    %% Graph Initialization & State
    Worker -->|Initialize State & Thread ID| Orch["Orchestrator (agent/orchestrator.py)"]
    Orch -->|Select Checkpointer| DB_Check["Checkpointer (SQLite / Postgres)"]
    
    subgraph Pipeline ["AIRS 14-Node LangGraph Pipeline"]
        %% Phase 0: Triage
        Node_Triage["triage_node"] -->|Severity P0| Node_Escalate["escalate_node (Slack Notification)"]
        Node_Triage -->|Severity P1-P3| Node_Perception["perception_node"]
        Node_Escalate --> Node_Perception
        
        %% Phase 3: Perception Layer
        Node_Perception -->|Tiered Log Classification| Router{"Neuro-Symbolic Router"}
        
        %% Phase 1: EKG & Phase 2: CBR
        Node_Perception --> Node_Topology["topology_agent_node (EKG Graph)"]
        Node_Topology --> Node_Diagnostic["diagnostic_agent_node (CBR Engine)"]
        
        %% Routing Decision
        Node_Diagnostic -->|NeSy Routing| Router
        Router -->|SYMBOLIC_FAST / CBR_GUIDED| Node_Logic["logic_agent_node (Symbolic Pruning)"]
        Router -->|NEURAL_FULL| Node_Investigate["investigate_node (ReAct LLM Loop)"]
        
        %% Investigation Loop
        Node_Investigate -->|Get Logs / Metrics| Node_Investigate
        Node_Investigate -->|Extraction| Node_Extract["extract_node (Evidence XML)"]
        Node_Extract --> Node_Logic
        
        %% Remediation Plan
        Node_Logic --> Node_Remediation["remediation_agent_node (CBR Adapt / LLM)"]
        Node_Remediation -->|High Risk Plan| Node_Reject["reject_node (Abort)"]
        Node_Remediation -->|Safe Plan| Node_Risk["risk_agent_node (Blast Radius)"]
        
        %% Policy & HITL Approval Gate
        Node_Risk --> Node_Policy["policy_check_node (Terraform & Invariants)"]
        Node_Policy -->|Policy Blocked| Node_Reject
        Node_Policy -->|Approval Required| Node_Approval["approval_node (__interrupt__ HITL)"]
        
        %% Resumed Execution
        Node_Approval -->|Approved / Canary| Node_Canary["canary_execute_node (Rollout Ctrl)"]
        Node_Approval -->|Approved / Direct| Node_Direct["direct_execute_node"]
        Node_Approval -->|Rejected| Node_Reject
        
        %% Learning & Conclusion
        Node_Canary --> Node_Retain["retain_node (Save Case to CBR)"]
        Node_Direct --> Node_Retain
    end
    
    Node_Retain -->|Persist Case| CBR_Store["Incident Case Store"]
    Node_Retain --> END["Incident Resolved (postmortem.md)"]
    Node_Reject --> END
```

---

## 2. Boot Sequence and Core Execution Control Flow

The runtime lifecycle of an incident remediation is split into three main execution phases:

1. **Ingestion & De-queuing**: The FastAPI endpoint parses the PagerDuty JSON payload, generates a unique transaction `thread_id`, and dispatches it asynchronously to Celery via a Redis broker.
2. **State Construction & Node Routing**: The Celery worker runs `asyncio.run(_run_graph(...))`. The Orchestrator compiles the state machine, restoring previous execution frames from the persistent PostgreSQL or SQLite database using the `thread_id`.
3. **Execution & Checkpointing**: Every node acts as an atomic transaction. Upon completion, the node outputs a partial dict which is merged into `GraphState` and serialized to database storage before the transition edge executes.

---

## 3. Deep Dive: Perception Layer

The Perception Layer is designed to classify high-volume incident records and log payloads before calling reasoning LLMs.

### Tiered Log Classifier (`agent/perception/log_classifier.py`)
To prevent token-limit exhaustion and control API costs, log telemetry goes through a three-tiered inspection system:
* **L1 (Regex Pattern Matcher)**: Fast classification scanning for known patterns like OOM killer logs, database timeout stack traces, and DNS server failures.
* **L2 (TF-IDF Similarity Clustering)**: If L1 fails to extract a clear signature, log structures are transformed into vector spaces using TF-IDF to calculate cosine similarity against known historical error templates.
* **L3 (Few-shot LLM Parsing)**: Highly anomalous messages are routed to a lightweight, low-temperature LLM run to deduce the failure category.

### Perception State Output
The perception node outputs `perception_stats` and `primary_log_template` into `GraphState`:
```python
perception_stats = {
    "L1_hits": int, 
    "L2_hits": int, 
    "L3_hits": int
}
primary_log_template = "connection_pool_exhausted"
```

---

## 4. Deep Dive: Reasoning Layer

The Core Reasoning Layer uses a Hybrid Memory Architecture (HMA) combining deterministic rules, graphs, and neural systems.

### Neuro-Symbolic Router (`agent/reasoning/nesym_router.py`)
This component routes the processing pathway according to the severity, log classification, and historical match confidence:
1. **`SYMBOLIC_FAST`**: Executed if log classification yields a 100% match to known, deterministic failure templates (e.g., expired TLS certificate). It bypasses all neural analysis and generates a hardcoded correction script.
2. **`CBR_GUIDED`**: Executed if the CBR engine yields a cosine similarity score of $\ge 0.8$ against past incident cases. The system adapts the historical remediation steps directly.
3. **`NEURAL_FULL`**: Active when incident logs are anomalous, or when CBR confidence is low. This triggers the agent's full investigative `ReAct` loop, calling external logs and metrics APIs to diagnose the incident.

### Enterprise Knowledge Graph (EKG) (`agent/reasoning/knowledge_graph.py`)
- **No-LLM Graph Traversal**: Queries a Neo4j database (or falls back to an in-memory NetworkX model) representing dependencies between services.
- **Bi-directional Querying**: 
  - *Upstream Query*: What services does the failing service depend on? (Helps find the root cause of cascading failures).
  - *Downstream Query*: What services depend on the failing service? (Calculates the impact scope for blast radius estimation).
  
```mermaid
graph LR
    UserSvc[User Service] -->|HTTP Call| AuthSvc[Auth Service]
    AuthSvc -->|Connection Pool| AuthDB[(Auth Database)]
    
    classDef critical fill:#f9f,stroke:#333,stroke-width:2px;
    class AuthDB critical;
```

### Case-Based Reasoning (CBR) Engine (`agent/reasoning/cbr_engine.py`)
The CBR Engine implements continuous learning using four classic steps:
1. **Retrieve**: Converts current telemetry, service tier, and alert descriptions into a unified feature string. It retrieves the top $k$ matches from the database using cosine similarity.
2. **Reuse**: Copies the successful remediation command sequence from the historical precedent case.
3. **Revise**: Using a targeted prompt, the `remediation_agent_node` substitutes runtime variables (e.g., target namespace, pod names) from the current incident into the historical template.
4. **Retain**: If the execution is successful, `retain_node` saves the resolved incident details, telemetry patterns, and outcomes as a new case in the persistent store.

### Logic Agent (Symbolic Hypothesis Pruning) (`agent/nodes.py`)
To prevent LLM hallucination and ensure safety, the Logic Agent applies strict, non-neural logic filters:
- **Nominal Pruning**: Parses metrics retrieved during investigation. If CPU or memory values are beneath nominal thresholds (e.g., CPU utilization is only $35\%$), any hypotheses suggesting CPU bottlenecks or memory leaks are pruned.
- **Topological Inconsistency**: If a dependency indicates a clean health status but a child service claims dependency timeouts, the logic agent adjusts the confidence levels of the cascading failure hypothesis accordingly.

---

## 5. Safe Action & Execution Layer

### Risk Agent & Blast Radius (`agent/action/blast_radius.py`)
Estimates the safety of the proposed rollback or restart commands using EKG node tiers. If the action targets a Tier-1 service (e.g., production database, gateway), the execution strategy is upgraded to `require_approval` or `blocked`.

### Policy-as-Code Engine (`agent/action/policy_engine.py`)
Runs Terraform dry-runs and checks constraints to ensure the generated commands do not violate security compliance (e.g., exposing public SSH ports or deleting stateful volumes).

### Canary & Rollback Controller (`agent/action/canary_controller.py`)
- **Rollout Stages**: Increments traffic routing to corrected pods in stages (e.g., $10\% \rightarrow 50\% \rightarrow 100\%$).
- **Active Rollback**: Compares golden signal snapshots (error rate, latency) against baseline pre-incident telemetry. If metrics degrade, it halts the rollout and executes the `rollback_command`.
- **HITL checkpoint**: If an interrupt is triggered, the state is persisted. A restart command will fetch the saved state and resume exactly at the user approval step.
