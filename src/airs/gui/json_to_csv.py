import pandas as pd
import os
from typing import List, Dict, Any

def convert_metrics_to_csv(data: List[Dict[str, Any]], output_dir: str, filename: str = "prometheus_metrics.csv") -> str:
    """
    Format: timestamp, cmdb_id, kpi_name, value
    """
    if not data:
        return ""
    
    df = pd.DataFrame(data)
    # Ensure columns exist even if data is incomplete
    columns = ["timestamp", "cmdb_id", "kpi_name", "value"]
    for col in columns:
        if col not in df.columns:
            df[col] = None
    
    df = df[columns]
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    df.to_csv(out_path, index=False)
    return out_path

def convert_logs_to_csv(data: List[Dict[str, Any]], output_dir: str, filename: str = "cluster_incident_logs.csv") -> str:
    """
    Format: log_id, timestamp, cmdb_id, log_name, value
    """
    if not data:
        return ""
    
    df = pd.DataFrame(data)
    columns = ["log_id", "timestamp", "cmdb_id", "log_name", "value"]
    for col in columns:
        if col not in df.columns:
            df[col] = None
            
    df = df[columns]
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    
    # If the file exists, we might want to append, but for now we write/overwrite per fetch
    df.to_csv(out_path, index=False)
    return out_path

def convert_traces_to_csv(data: List[Dict[str, Any]], output_dir: str, filename: str = "incident_traces.csv") -> str:
    """
    Format: timestamp, cmdb_id, parent_id, span_id, trace_id, duration
    """
    if not data:
        return ""
    
    df = pd.DataFrame(data)
    columns = ["timestamp", "cmdb_id", "parent_id", "span_id", "trace_id", "duration"]
    for col in columns:
        if col not in df.columns:
            df[col] = None
            
    df = df[columns]
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    df.to_csv(out_path, index=False)
    return out_path
