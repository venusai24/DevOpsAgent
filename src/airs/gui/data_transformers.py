from typing import List, Dict, Any
import uuid

def transform_prometheus_json(raw_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Transforms native Prometheus API JSON response into the flat format:
    timestamp, cmdb_id, kpi_name, value
    """
    results = []
    try:
        data = raw_json.get("data", {})
        result_array = data.get("result", [])
        
        for timeseries in result_array:
            metric = timeseries.get("metric", {})
            # Infer cmdb_id from common instance or pod labels
            cmdb_id = metric.get("instance") or metric.get("pod") or metric.get("host") or "unknown_cmdb"
            kpi_name = metric.get("__name__", "unknown_kpi")
            
            # Handle both 'values' (matrix) and 'value' (vector)
            values = timeseries.get("values", [])
            if "value" in timeseries:
                values = [timeseries["value"]]
                
            for val_pair in values:
                if len(val_pair) >= 2:
                    ts = val_pair[0]
                    value_str = val_pair[1]
                    results.append({
                        "timestamp": ts,
                        "cmdb_id": cmdb_id,
                        "kpi_name": kpi_name,
                        "value": float(value_str)
                    })
    except Exception as e:
        print(f"Error transforming Prometheus JSON: {e}")
    return results

def transform_loki_json(raw_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Transforms native Loki API JSON response into the flat format:
    log_id, timestamp, cmdb_id, log_name, value
    """
    results = []
    try:
        data = raw_json.get("data", {})
        result_array = data.get("result", [])
        
        for stream_obj in result_array:
            stream = stream_obj.get("stream", {})
            cmdb_id = stream.get("instance") or stream.get("host") or stream.get("pod") or "unknown_cmdb"
            log_name = stream.get("filename") or stream.get("app") or stream.get("container") or "unknown_log"
            
            values = stream_obj.get("values", [])
            for val_pair in values:
                if len(val_pair) >= 2:
                    ts = val_pair[0] # Usually epoch string in ns
                    log_line = val_pair[1]
                    results.append({
                        "log_id": str(uuid.uuid4()),
                        "timestamp": ts,
                        "cmdb_id": cmdb_id,
                        "log_name": log_name,
                        "value": log_line
                    })
    except Exception as e:
        print(f"Error transforming Loki JSON: {e}")
    return results

def transform_opensearch_json(raw_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Transforms OpenSearch/ElasticSearch JSON response into the flat format:
    log_id, timestamp, cmdb_id, log_name, value
    """
    results = []
    try:
        hits_obj = raw_json.get("hits", {})
        hits_array = hits_obj.get("hits", [])
        
        for hit in hits_array:
            log_id = hit.get("_id", str(uuid.uuid4()))
            source = hit.get("_source", {})
            
            ts = source.get("@timestamp") or source.get("timestamp") or "unknown_time"
            cmdb_id = source.get("host", {}).get("name") or source.get("kubernetes", {}).get("pod_name") or "unknown_cmdb"
            log_name = source.get("log", {}).get("file", {}).get("path") or source.get("container", {}).get("name") or "unknown_log"
            value = source.get("message") or str(source)
            
            results.append({
                "log_id": log_id,
                "timestamp": ts,
                "cmdb_id": cmdb_id,
                "log_name": log_name,
                "value": value
            })
    except Exception as e:
        print(f"Error transforming OpenSearch JSON: {e}")
    return results

def transform_jaeger_json(raw_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Transforms Jaeger traces API JSON response into the flat format:
    timestamp, cmdb_id, parent_id, span_id, trace_id, duration
    """
    results = []
    try:
        data_array = raw_json.get("data", [])
        for trace in data_array:
            trace_id = trace.get("traceID", "unknown_trace")
            
            # Build process mapping
            processes = trace.get("processes", {})
            
            spans = trace.get("spans", [])
            for span in spans:
                span_id = span.get("spanID")
                # references usually contain parent
                parent_id = "null"
                for ref in span.get("references", []):
                    if ref.get("refType") == "CHILD_OF":
                        parent_id = ref.get("spanID")
                        break
                        
                ts = span.get("startTime") # Usually epoch microseconds
                duration = span.get("duration") # Microseconds
                
                process_id = span.get("processID")
                process = processes.get(process_id, {})
                cmdb_id = process.get("serviceName", "unknown_cmdb")
                
                results.append({
                    "timestamp": ts,
                    "cmdb_id": cmdb_id,
                    "parent_id": parent_id,
                    "span_id": span_id,
                    "trace_id": trace_id,
                    "duration": duration
                })
    except Exception as e:
        print(f"Error transforming Jaeger JSON: {e}")
    return results

def transform_tempo_json(raw_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Transforms Tempo traces API JSON response into the flat format:
    timestamp, cmdb_id, parent_id, span_id, trace_id, duration
    (Assuming Tempo returns OpenTelemetry OTLP JSON trace format or similar)
    """
    results = []
    try:
        # Tempo OTLP JSON format
        resource_spans = raw_json.get("resourceSpans", [])
        for rs in resource_spans:
            resource = rs.get("resource", {})
            attributes = resource.get("attributes", [])
            
            cmdb_id = "unknown_cmdb"
            for attr in attributes:
                if attr.get("key") == "service.name":
                    cmdb_id = attr.get("value", {}).get("stringValue", cmdb_id)
                    break
                    
            scope_spans = rs.get("scopeSpans", [])
            for ss in scope_spans:
                spans = ss.get("spans", [])
                for span in spans:
                    trace_id = span.get("traceId")
                    span_id = span.get("spanId")
                    parent_id = span.get("parentSpanId", "null")
                    
                    start_time = int(span.get("startTimeUnixNano", 0))
                    end_time = int(span.get("endTimeUnixNano", 0))
                    duration = (end_time - start_time) / 1000.0 # microseconds
                    
                    results.append({
                        "timestamp": start_time,
                        "cmdb_id": cmdb_id,
                        "parent_id": parent_id,
                        "span_id": span_id,
                        "trace_id": trace_id,
                        "duration": duration
                    })
    except Exception as e:
        print(f"Error transforming Tempo JSON: {e}")
    return results
