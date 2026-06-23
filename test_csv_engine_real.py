import asyncio
import logging
from pathlib import Path
from pprint import pprint

from airs.csv_engine.engine import CsvEngine
from airs.signal_adapters.csv_adapter import CsvSignalAdapter
from airs.models.intents import ExecutionIntent, IntentAction, ToolSpec, SignalType

logging.basicConfig(level=logging.INFO)

async def test_mysql02_corpus():
    data_dir = Path("/home/VenuSai/DevOpsAgent/mysql02_corpus_case")
    
    # 1. Initialize Adapter
    adapter = CsvSignalAdapter(data_dir=data_dir)
    print("--- 1. Testing Schema Discovery ---")
    intent_schema = ExecutionIntent(
        action=IntentAction.EXECUTE_TOOL,
        tool_spec=ToolSpec(
            tool_name="csv_schema_discovery",
            mcp_server_id="csv_engine",
            tier=1,
            signal_type=SignalType.CSV
        )
    )
    raw = await adapter.execute(intent_schema)
    filtered = adapter.pre_filter(raw)
    pprint(filtered.filtered_content)
    
    print("\n--- 2. Testing Z-Score Anomaly Detection ---")
    intent_anomaly = ExecutionIntent(
        action=IntentAction.EXECUTE_TOOL,
        tool_spec=ToolSpec(
            tool_name="csv_anomaly_detection",
            mcp_server_id="csv_engine",
            tier=1,
            signal_type=SignalType.CSV,
            arguments={"kpi_name": "cpu_usage_percent"}
        )
    )
    raw = await adapter.execute(intent_anomaly)
    filtered = adapter.pre_filter(raw)
    print(f"Success: {filtered.filter_stats['success']}")
    print(f"Rows returned: {filtered.filter_stats['rows_returned']}")
    print("Sample anomaly:")
    anomalies = filtered.filtered_content.get("anomalies", [])
    if anomalies:
        pprint(anomalies[0])
    else:
        print("No anomalies detected.")

    print("\n--- 3. Testing Log Pattern Extract ---")
    intent_logs = ExecutionIntent(
        action=IntentAction.EXECUTE_TOOL,
        tool_spec=ToolSpec(
            tool_name="csv_log_pattern_extract",
            mcp_server_id="csv_engine",
            tier=1,
            signal_type=SignalType.CSV,
            arguments={"table": "loki_logs", "min_count": 5}
        )
    )
    raw = await adapter.execute(intent_logs)
    filtered = adapter.pre_filter(raw)
    print(f"Total lines: {filtered.filtered_content.get('total_log_lines')}")
    print(f"Unique patterns (>5): {filtered.filtered_content.get('unique_patterns')}")
    patterns = filtered.filtered_content.get("patterns", [])
    if patterns:
        print("Top pattern:")
        pprint(patterns[0])

    print("\n--- 4. Testing SQL Query (Budget Limiting) ---")
    intent_sql = ExecutionIntent(
        action=IntentAction.EXECUTE_TOOL,
        tool_spec=ToolSpec(
            tool_name="csv_sql_query",
            mcp_server_id="csv_engine",
            tier=1,
            signal_type=SignalType.CSV,
            arguments={"query": "SELECT * FROM jaeger_traces"}
        )
    )
    raw = await adapter.execute(intent_sql)
    filtered = adapter.pre_filter(raw)
    print(f"Truncated? {filtered.filter_stats['truncated']}")
    print(f"Rows returned: {filtered.filter_stats['rows_returned']}")
    print(f"Token estimate: {filtered.token_estimate}")

if __name__ == "__main__":
    asyncio.run(test_mysql02_corpus())
