import asyncio
import json
import os
import sys
from datetime import UTC, datetime

import src.diagnosis.parser
import src.hitl.graph
import src.orchestrator.assembler
import src.taip.injection
import src.taip.pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from src.models.anomaly_event import AnomalyEvent
from src.taip.pipeline import run_taip

print("Starting run_dry_test.py", file=sys.stderr)


# The output tracking file
TRACKING_FILE = "/home/VenuSai/.gemini/antigravity-ide/brain/1c311fdf-e571-4bba-a040-60697892f8fb/artifacts/dry_run_tracking.md"

class InstrumentedLLM:
    """Wraps ChatGoogleGenAI to log all inputs and outputs to a markdown tracking file."""
    def __init__(self, underlying_llm):
        self.llm = underlying_llm
        # Initialize tracking file
        os.makedirs(os.path.dirname(TRACKING_FILE), exist_ok=True)
        with open(TRACKING_FILE, "w") as f:
            f.write("# Dry Run Tracking Log\n\n")

    async def ainvoke(self, messages, tools=None, **kwargs):
        with open(TRACKING_FILE, "a") as f:
            f.write("## 📥 LLM Request\n\n")
            f.write("### Messages\n")
            for msg in messages:
                content = msg.get("content", str(msg)) if isinstance(msg, dict) else getattr(msg, "content", str(msg))
                role = msg.get("role", type(msg).__name__) if isinstance(msg, dict) else type(msg).__name__
                f.write(f"**{role}:**\n```text\n" + str(content) + "\n```\n")
            
            if tools:
                f.write("\n### Tools Provided\n")
                for t in tools:
                    f.write(f"- `{t.get('name') if isinstance(t, dict) else getattr(t, 'name', str(t))}`\n")
            f.write("\n")

        # Call real LLM
        response = await self.llm.ainvoke(messages, tools=tools, **kwargs)

        with open(TRACKING_FILE, "a") as f:
            f.write("## 📤 LLM Response\n\n")
            content_str = response.content
            if isinstance(content_str, list):
                content_str = json.dumps(content_str, indent=2)
            if hasattr(response, "tool_calls") and response.tool_calls:
                f.write("```json\n" + json.dumps(response.tool_calls, indent=2) + "\n```\n\n")
            elif content_str:
                f.write("```json\n" + str(content_str) + "\n```\n\n")
            else:
                f.write("*(Empty content)*\n\n")
            f.write("---\n\n")

        return response

def strict_parse_diagnosis_output(llm_response: str) -> dict:
    """A patched parser that fails immediately on hallucination with NO fallbacks."""
    import re
    json_match = re.search(r"```(?:json)?(.*?)```", llm_response, re.DOTALL)
    json_str = json_match.group(1).strip() if json_match else llm_response.strip()
    
    # Remove markdown prefix if exists
    if not json_str.startswith("{"):
        start_idx = json_str.find("{")
        end_idx = json_str.rfind("}")
        if start_idx != -1 and end_idx != -1:
            json_str = json_str[start_idx:end_idx+1]

    if not json_str:
        raise ValueError("HALLUCINATION DETECTED: Could not find JSON in response.")

    try:
        report = json.loads(json_str)
        return {
            "rca_report": report,
            "current_step": report.get("current_step", 6),
            "epicenter_candidates": [h.get("epicenter") for h in report.get("hypotheses", [])],
            "hypotheses": report.get("hypotheses", []),
            "confidence_scores": [h.get("confidence_score", 0.0) for h in report.get("hypotheses", [])],
            "surprise_nodes": report.get("surprise_nodes", []),
            "scenario": report.get("scenario")
        }
    except json.JSONDecodeError as e:
        raise ValueError(f"HALLUCINATION DETECTED: LLM output is not valid JSON. {e}\nRaw Output: {json_str}")

# Apply patches
src.diagnosis.parser.parse_diagnosis_output = strict_parse_diagnosis_output

# Patch count_tokens to handle datetime correctly
original_count_tokens = src.taip.injection.count_tokens

def patched_count_tokens(obj):
    if isinstance(obj, str):
        return original_count_tokens(obj)
    text = json.dumps(obj, default=str)
    return len(text) // 4  # Approximation used in count_tokens

src.taip.injection.count_tokens = patched_count_tokens

src.taip.pipeline.count_tokens = patched_count_tokens

async def run_dry_test():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        return
        
    os.environ["HF_TOKEN"] = os.environ.get("HF_KEY", "")
        
    # Use Gemma 31B
    underlying = ChatGoogleGenerativeAI(model="models/gemma-4-31b-it", google_api_key=api_key)
    llm = InstrumentedLLM(underlying)

    # Patch memory saver
    
    # 3. Read Injected CSVs and run Anomaly Detection
    import pandas as pd
    
    print("Reading App Stats CSV...")
    
    anomalies = []
    
    # Process App Stats
    df_app = pd.read_csv("incident_data/injected_cluster_app_stats.csv")
    print(f"Read {len(df_app)} rows from app stats.")
    for tc in df_app['tc'].unique():
        df_tc = df_app[df_app['tc'] == tc]
        for _, row in df_tc.iterrows():
            if row['mrt'] > 1000:  # Injected network latency
                anomalies.append(AnomalyEvent(
                    event_id=f"EVT-{len(anomalies):06d}",
                    timestamp=datetime.fromtimestamp(row['timestamp'], UTC),
                    service=tc,
                    host=f"{tc}-host",
                    metric="mrt",
                    observed_value=row['mrt'],
                    baseline_value=140.0,
                    deviation_pct=((row['mrt']/140.0)*100),
                    severity=4, # CRITICAL
                    z_score=10.0,
                    description=f"High MRT anomaly detected on {tc}",
                    source_dataset="app_stats",
                    detection_method="mad_zscore"
                ))

    print("Reading Redis CSV...")
    # Process Redis
    df_redis = pd.read_csv("incident_data/injected_redis02_metrics.csv")
    print(f"Read {len(df_redis)} rows from Redis.")
    for kpi in ['redis-Redis_6379_Redis  (evicted_keys)', 'OSLinux-MEMORY_MEMORY_MEMUsedMemPerc']:
        df_kpi = df_redis[df_redis['kpi_name'] == kpi]
        for _, row in df_kpi.iterrows():
            val = row['value']
            is_anomaly = False
            if kpi == 'redis-Redis_6379_Redis  (evicted_keys)' and val > 5000:
                is_anomaly = True
            elif 'MEMUsedMemPerc' in kpi and val > 90.0:
                is_anomaly = True
                
            if is_anomaly:
                anomalies.append(AnomalyEvent(
                    event_id=f"EVT-{len(anomalies):06d}",
                    timestamp=datetime.fromtimestamp(row['timestamp'], UTC),
                    service="Redis02",
                    host="Redis02",
                    metric=kpi,
                    observed_value=val,
                    baseline_value=0.0 if 'evicted' in kpi else 40.0,
                    deviation_pct=100.0,
                    severity=4,
                    z_score=10.0,
                    description=f"Anomaly on {kpi}",
                    source_dataset="metrics",
                    detection_method="mad_zscore"
                ))
                
    # Ensure we keep a diverse set of anomalies, especially the Redis ones which are fewer
    redis_anoms = [a for a in anomalies if "Redis" in a.service]
    app_anoms = [a for a in anomalies if "ServiceTest" in a.service]
    
    import random
    if len(app_anoms) > 40:
        app_anoms = random.sample(app_anoms, 40)
    
    anomalies = redis_anoms + app_anoms
            
    print(f"Detected {len(anomalies)} anomalies from injected datasets.")
    
    # Topology DSL that resonates with the CSV data
    topo_dsl_text = """
    group APP_CLUSTER {
        node ServiceTest11;
        node ServiceTest10;
        node ServiceTest1;
    }
    group CACHE {
        node Redis02;
    }
    relation APP_CLUSTER -> CACHE;
    """

    print("Starting Dry Run...")
    try:
        from src.hitl.graph import build_diagnosis_graph
        from src.taip.assembly import assemble_pattern_bundle
        
        context_payload = run_taip(anomalies, "INC-20260628-001", llm_client=None)
        scenario = "LATENCY_SPIKE"
        pattern_bundle = assemble_pattern_bundle(
            incident_id="INC-20260628-001",
            raw_anomalies=anomalies,
            context_payload=context_payload,
            scenario=scenario,
            collection_gaps=[],
        )
        
        from src.diagnosis.csv_tools import analyze_trace_csv, query_metric_csv, search_logs_csv
        from src.diagnosis.mcp_tools import query_notebooklm
        
        checkpointer = MemorySaver()
        graph = build_diagnosis_graph(
            llm=llm,
            tools=[query_notebooklm, query_metric_csv, search_logs_csv, analyze_trace_csv],
            db_pool=None,
            checkpointer=checkpointer,
        )
        
        initial_state = {
            "messages": [],
            "incident_id": "INC-20260628-001",
            "scenario": pattern_bundle.scenario,
            "context_payload": context_payload.model_dump(),
            "topo_dsl_text": topo_dsl_text,
            "memory_pocket": "Previous incidents showed ServiceTest11 errors when Redis02 CPU spikes due to connection pooling.",
            "collection_gaps": [],
            "current_step": 0,
            "epicenter_candidates": [],
            "propagation_paths": [],
            "hypotheses": [],
            "confidence_scores": [],
            "surprise_nodes": [],
            "paging_tool_calls": 0,
            "hitl_round": 0,
            "hitl_required": False,
            "hitl_reason": None,
            "escalation": None,
            "sre_response": None,
            "rca_report": None,
            "status": "running",
        }
        
        config = {"configurable": {"thread_id": "INC-20260628-001"}}
        result = await graph.ainvoke(initial_state, config=config)
        print("Dry Run Completed Successfully.")
        print("Status:", result.get("status"))
        
    except ValueError as e:
        if "HALLUCINATION DETECTED" in str(e):
            print("\n*** DRY RUN HALTED: LLM HALLUCINATION DETECTED ***")
            print(e)
            with open(TRACKING_FILE, "a") as f:
                f.write(f"\n## 🚨 Pipeline Halted\n\n{str(e)}\n")
        else:
            raise e

if __name__ == "__main__":
    asyncio.run(run_dry_test())
