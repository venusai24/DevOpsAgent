CREATE EXTENSION IF NOT EXISTS vector;

-- Ephemeral Anomaly Store
CREATE TABLE anomaly_events (
    event_id VARCHAR(50) PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    service VARCHAR(100) NOT NULL,
    host VARCHAR(100) NOT NULL,
    metric VARCHAR(100) NOT NULL,
    observed_value FLOAT,
    baseline_value FLOAT,
    deviation_pct FLOAT,
    severity INTEGER,
    description TEXT,
    source_dataset VARCHAR(50),
    detection_method VARCHAR(50),
    z_score FLOAT,
    onset_timestamp TIMESTAMPTZ,
    causal_predecessor_id VARCHAR(50),
    log_type VARCHAR(50),
    log_entry_id VARCHAR(50)
);

-- DSEM RCA Canonical Store
CREATE TABLE canonical_rcas (
    incident_id VARCHAR(50) PRIMARY KEY,
    incident_date DATE NOT NULL,
    root_cause_type VARCHAR(100),
    mttr_minutes INTEGER,
    resolution_summary TEXT,
    affected_services TEXT[]
);

-- DSEM RCA Vectors (Episodic Memory)
CREATE TABLE rca_vectors (
    id SERIAL PRIMARY KEY,
    incident_id VARCHAR(50) REFERENCES canonical_rcas(incident_id) ON DELETE CASCADE,
    embedding vector(768) -- Note: using 768 to match typical embedding models like nomic-embed-text or gemini
);

-- DSEM Causal Graph
CREATE TABLE anomaly_nodes (
    node_id VARCHAR(100) PRIMARY KEY,
    incident_id VARCHAR(50) REFERENCES canonical_rcas(incident_id) ON DELETE CASCADE,
    service_name VARCHAR(100),
    metric_name VARCHAR(100)
);

CREATE TABLE causal_edges (
    source_node VARCHAR(100) REFERENCES anomaly_nodes(node_id),
    target_node VARCHAR(100) REFERENCES anomaly_nodes(node_id),
    relationship VARCHAR(50),
    PRIMARY KEY (source_node, target_node)
);

-- HITL Events
CREATE TABLE hitl_events (
    event_id VARCHAR(50) PRIMARY KEY,
    incident_id VARCHAR(50) NOT NULL,
    trigger_type VARCHAR(50),
    confidence_at_pause FLOAT,
    reason TEXT,
    generated_at TIMESTAMPTZ NOT NULL,
    thread_id VARCHAR(100),
    resumption_token VARCHAR(255),
    status VARCHAR(50) DEFAULT 'awaiting_response',
    response_payload JSONB
);

-- Note: LangGraph PostgresSaver tables (checkpoints, checkpoint_writes) 
-- will be initialized automatically via python using PostgresSaver.setup()
