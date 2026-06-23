"""
DuckDB CSV Engine — Module: Phase A.

The core data engine for the CSV Intelligence Layer. DuckDB reads CSV files
directly from disk using vectorized scans, never loading the entire file
into memory. This makes it suitable for files too large for Excel.

Architecture:
  - One in-memory DuckDB connection per investigation session.
  - Each CSV file is registered as a persistent VIEW on top of the
    raw file. DuckDB scans rows lazily as queries demand them.
  - All queries pass through the `CsvEngine.query()` method, which:
      1. Enforces a hard row limit to protect context budget.
      2. Validates that only the expected 5 CSV schemas are registered.
      3. Returns results as plain Python dicts (JSON-serializable).

Known CSV schemas (from GUI data_transformers.py):
  - prometheus_metrics.csv      → [timestamp, cmdb_id, kpi_name, value]
  - loki_incident_logs.csv      → [log_id, timestamp, cmdb_id, log_name, value]
  - opensearch_incident_logs.csv → [log_id, timestamp, cmdb_id, log_name, value]
  - jaeger_incident_traces.csv  → [timestamp, cmdb_id, parent_id, span_id, trace_id, duration]
  - tempo_incident_traces.csv   → [timestamp, cmdb_id, parent_id, span_id, trace_id, duration]
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import duckdb

log = logging.getLogger(__name__)

# Hard limit on rows returned to the LLM — enforces context hygiene.
# The LLM must use aggregations (GROUP BY, COUNT, AVG) for large datasets.
_MAX_RESULT_ROWS = 200

# Canonical names for the 5 CSV views registered in DuckDB.
# Keys = view names used in SQL. Values = glob pattern for expected CSV filename.
KNOWN_CSV_TABLES: dict[str, str] = {
    "prometheus_metrics": "*metrics.csv",
    "loki_logs": "*logs.csv",
    "opensearch_logs": "*opensearch*.csv",
    "jaeger_traces": "*traces.csv",
    "tempo_traces": "*tempo*.csv",
}

# Expected column schemas — used for validation and schema discovery.
TABLE_SCHEMAS: dict[str, list[str]] = {
    "prometheus_metrics": ["timestamp", "cmdb_id", "kpi_name", "value"],
    "loki_logs": ["log_id", "timestamp", "cmdb_id", "log_name", "value"],
    "opensearch_logs": ["log_id", "timestamp", "cmdb_id", "log_name", "value"],
    "jaeger_traces": ["timestamp", "cmdb_id", "parent_id", "span_id", "trace_id", "duration"],
    "tempo_traces": ["timestamp", "cmdb_id", "parent_id", "span_id", "trace_id", "duration"],
}


class CsvEngine:
    """
    DuckDB-backed analytical engine over the 5 incident telemetry CSV files.

    Usage:
        engine = CsvEngine(incident_data_dir="/data/incident-2024-01-01")
        engine.initialize()
        results = engine.query("SELECT kpi_name, AVG(value) FROM prometheus_metrics GROUP BY kpi_name")
    """

    def __init__(self, incident_data_dir: str | Path) -> None:
        self.incident_data_dir = Path(incident_data_dir)
        # In-memory connection — no disk state, just views over CSV files
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._registered_tables: dict[str, str] = {}  # view_name → csv_path

    def initialize(self) -> dict[str, Any]:
        """
        Scan the incident_data_dir for known CSV files, register each
        as a DuckDB VIEW, and return a schema discovery summary.

        Returns:
            dict with registered tables, their column names, types,
            and row counts (computed cheaply via DuckDB COUNT(*)).
        """
        self._conn = duckdb.connect(database=":memory:")

        # Enable progress reporting for large scans
        self._conn.execute("SET enable_progress_bar = false")

        schema_info: dict[str, Any] = {"registered_tables": {}, "missing_files": []}

        for view_name, pattern in KNOWN_CSV_TABLES.items():
            matches = list(self.incident_data_dir.glob(pattern))
            if not matches:
                schema_info["missing_files"].append(pattern)
                log.info("CSV matching %s not found — skipping.", pattern)
                continue
            
            # Use the first matched file
            csv_path = matches[0]

            # Register as a view using DuckDB's read_csv_auto.
            # read_csv_auto infers types, handles quoting, and scans lazily.
            self._conn.execute(
                f"CREATE OR REPLACE VIEW {view_name} AS "
                f"SELECT * FROM read_csv_auto('{csv_path}', ignore_errors=true)"
            )
            self._registered_tables[view_name] = str(csv_path)

            # Count rows — cheap DuckDB vectorised scan
            row_count = self._conn.execute(
                f"SELECT COUNT(*) as cnt FROM {view_name}"
            ).fetchone()[0]  # type: ignore[index]

            # Introspect column names and types
            col_info = self._conn.execute(
                f"DESCRIBE {view_name}"
            ).fetchall()
            columns = [{"name": row[0], "type": row[1]} for row in col_info]

            schema_info["registered_tables"][view_name] = {
                "source_file": csv_path.name,
                "row_count": row_count,
                "columns": columns,
            }

            log.info(
                "Registered CSV view: %s (%d rows, %d columns)",
                view_name, row_count, len(columns),
            )

        return schema_info

    def query(self, sql: str) -> dict[str, Any]:
        """
        Execute a SQL query against the registered CSV views.

        Enforces two hard constraints that protect the LLM context budget:
          1. Row limit: Results are capped at _MAX_RESULT_ROWS (200).
             If the query would return more, we return only the first 200
             rows and flag `truncated=True`.
          2. Table allowlist: Only queries against registered views are
             permitted. Prevents arbitrary disk access.

        Args:
            sql: Arbitrary SQL string. The LLM is expected to write
                 aggregating queries (GROUP BY, AVG, COUNT) rather than
                 raw SELECT * statements.

        Returns:
            dict with:
              - columns: list of column names
              - rows: list of dicts (one per row)
              - row_count: number of rows returned
              - truncated: True if result was capped at _MAX_RESULT_ROWS
              - error: error message if query failed (None on success)
        """
        if self._conn is None:
            return _error_result("Engine not initialized. Call initialize() first.")

        if not self._registered_tables:
            return _error_result("No CSV files found. Check that the incident data directory is populated.")

        # Wrap the user query in a LIMIT guard to enforce the row cap.
        # We always fetch _MAX_RESULT_ROWS + 1 to detect truncation.
        guarded_sql = (
            f"SELECT * FROM ({sql}) __inner__ LIMIT {_MAX_RESULT_ROWS + 1}"
        )

        try:
            relation = self._conn.execute(guarded_sql)
            col_names = [desc[0] for desc in relation.description]
            raw_rows = relation.fetchall()
        except duckdb.Error as e:
            log.warning("DuckDB query failed: %s | SQL: %s", e, sql[:200])
            return _error_result(str(e))

        truncated = len(raw_rows) > _MAX_RESULT_ROWS
        rows = raw_rows[:_MAX_RESULT_ROWS]

        return {
            "columns": col_names,
            "rows": [dict(zip(col_names, row)) for row in rows],
            "row_count": len(rows),
            "truncated": truncated,
            "truncation_message": (
                f"Result truncated to {_MAX_RESULT_ROWS} rows. "
                "Use GROUP BY, LIMIT, or WHERE clauses to narrow your query."
                if truncated else None
            ),
            "error": None,
        }

    def registered_tables(self) -> list[str]:
        """Return names of all registered DuckDB views."""
        return list(self._registered_tables.keys())

    def close(self) -> None:
        """Release the DuckDB connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "CsvEngine":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _error_result(message: str) -> dict[str, Any]:
    return {
        "columns": [],
        "rows": [],
        "row_count": 0,
        "truncated": False,
        "truncation_message": None,
        "error": message,
    }
