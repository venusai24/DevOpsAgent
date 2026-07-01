"""Baseline Tool (Tool 1)."""

import hashlib
from typing import Any

from pydantic import BaseModel, Field

from devops_agent.core.db.duckdb_client import DuckDBClient

from ..base import BaseTool
from ..models import ToolContext, ToolMetadata, ToolSchema

# Global in-memory store for baseline data
_BASELINE_STORE: dict[str, dict[str, Any]] = {}

# --- Schemas ---

class ComputeBaselineInput(BaseModel):
    csv_paths: list[str] = Field(description="List of absolute paths to the CSV files containing baseline metric data.")
    time_range_start: str = Field(description="ISO 8601 UTC timestamp of the start of the time range.")
    time_range_end: str = Field(description="ISO 8601 UTC timestamp of the end of the time range.")
    force_recompute: bool = Field(default=False, description="If True, ignores cache and forces a full re-scan.")

class ComputeBaselineOutput(BaseModel):
    baseline_ref: str = Field(description="Opaque reference key to the computed baseline statistics in the data store.")
    time_range_hashed: str = Field(description="Deterministic hash of the time window and input file.")
    computation_status: str = Field(description="One of: 'cached', 'completed', 'in_progress_background'.")
    estimated_completion_seconds: int | None = Field(default=None, description="If in_progress_background, time until data is ready.")
    rows_processed: int | None = Field(default=None)

# --- Tool Implementation ---

class ComputeBaselineStatisticsTool(BaseTool[ComputeBaselineInput, ComputeBaselineOutput]):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="compute_baseline_statistics",
            description="Computes and caches z-score statistics for KPI anomalies from a time-series CSV using DuckDB.",
            version="1.1.0"
        )

    @property
    def schema(self) -> ToolSchema[ComputeBaselineInput]:
        return ToolSchema(input_type=ComputeBaselineInput, output_type=ComputeBaselineOutput)

    async def execute(self, ctx: ToolContext, inputs: ComputeBaselineInput) -> ComputeBaselineOutput:
        try:
            import pandas as pd
            t_start = pd.to_datetime(inputs.time_range_start).timestamp()
        except ValueError:
            t_start = float(inputs.time_range_start)

        hash_str = f"{inputs.time_range_start}_{inputs.time_range_end}_{','.join(inputs.csv_paths)}"
        baseline_ref = "baseline_" + hashlib.md5(hash_str.encode()).hexdigest()
        
        if baseline_ref in _BASELINE_STORE and not inputs.force_recompute:
            return ComputeBaselineOutput(
                baseline_ref=baseline_ref,
                time_range_hashed=hash_str,
                computation_status="cached",
                rows_processed=0
            )

        db = DuckDBClient.get_instance()
        combined_baseline = {
            "app_metrics": {},
            "container_metrics": {}
        }
        
        total_rows = 0
        for path in inputs.csv_paths:
            # First determine if it's app stats or container metrics
            try:
                columns = db.query(f"SELECT * FROM read_csv_auto('{path}') LIMIT 1").columns
            except Exception:
                continue

            if "tc" in columns:
                # App metrics baseline: group by 'tc'
                metrics = ["rr", "sr", "cnt", "mrt"]
                for m in metrics:
                    if m not in columns: continue
                    sql = f"SELECT tc, avg({m}) as mean, stddev_pop({m}) as std FROM read_csv_auto('{path}') WHERE timestamp < ? GROUP BY tc"
                    res = db.query_dict(sql, (t_start,))
                    for row in res:
                        tc = row['tc']
                        if tc not in combined_baseline["app_metrics"]:
                            combined_baseline["app_metrics"][tc] = {}
                        # Fill NaNs with 0 using duckdb logic or fallback in dict
                        std = row['std'] if row['std'] is not None else 0.0
                        mean = row['mean'] if row['mean'] is not None else 0.0
                        combined_baseline["app_metrics"][tc][m] = {"mean": mean, "std": std}
                
                # Estimate row count
                total_rows += db.query(f"SELECT count(*) as c FROM read_csv_auto('{path}')")['c'].iloc[0]

            elif "cmdb_id" in columns and "kpi_name" in columns:
                # Container metrics baseline
                sql = f"SELECT cmdb_id, kpi_name, avg(value) as mean, stddev_pop(value) as std FROM read_csv_auto('{path}') WHERE timestamp < ? GROUP BY cmdb_id, kpi_name"
                res = db.query_dict(sql, (t_start,))
                for row in res:
                    cmdb_id = row['cmdb_id']
                    kpi_name = row['kpi_name']
                    if cmdb_id not in combined_baseline["container_metrics"]:
                        combined_baseline["container_metrics"][cmdb_id] = {}
                    std = row['std'] if row['std'] is not None else 0.0
                    mean = row['mean'] if row['mean'] is not None else 0.0
                    combined_baseline["container_metrics"][cmdb_id][kpi_name] = {"mean": mean, "std": std}
                
                total_rows += db.query(f"SELECT count(*) as c FROM read_csv_auto('{path}')")['c'].iloc[0]

        _BASELINE_STORE[baseline_ref] = combined_baseline

        return ComputeBaselineOutput(
            baseline_ref=baseline_ref,
            time_range_hashed=hash_str,
            computation_status="completed",
            rows_processed=total_rows
        )
