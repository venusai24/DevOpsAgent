"""DuckDB Utility Client for Zero-Ops File Queries."""

import threading
from typing import Optional

import duckdb
import pandas as pd


class DuckDBClient:
    """Singleton wrapper for DuckDB execution."""
    _instance: Optional['DuckDBClient'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        # In-memory transient database optimized for analytics
        self.conn = duckdb.connect(':memory:')
        
    @classmethod
    def get_instance(cls) -> 'DuckDBClient':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        """
        Executes a SQL query against local CSVs and returns a Pandas DataFrame.
        DuckDB will stream the file directly without pulling it fully into RAM.
        """
        return self.conn.execute(sql, params).df()

    def query_dict(self, sql: str, params: tuple = ()) -> list[dict]:
        """Executes a query and returns list of dictionaries."""
        df = self.query(sql, params)
        return df.to_dict(orient='records')
