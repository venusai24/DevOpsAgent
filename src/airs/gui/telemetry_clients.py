import httpx
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

from src.airs.gui.data_transformers import (
    transform_prometheus_json,
    transform_loki_json,
    transform_opensearch_json,
    transform_jaeger_json,
    transform_tempo_json
)

logger = logging.getLogger(__name__)

class TelemetryClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=10.0)

    def fetch(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        raise NotImplementedError

class PrometheusClient(TelemetryClient):
    def fetch(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        # In a real environment, we'd do:
        # response = self.client.get(f"{self.base_url}/api/v1/query_range", params={...})
        # raw_json = response.json()
        
        # Simulating RAW Prometheus JSON response
        raw_json = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {"__name__": "cpu_usage_percent", "instance": "srv-app-01"},
                        "values": [[start_time.timestamp(), "45.2"], [end_time.timestamp(), "48.1"]]
                    }
                ]
            }
        }
        return transform_prometheus_json(raw_json)

class LokiClient(TelemetryClient):
    def fetch(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        # Simulating RAW Loki JSON response
        raw_json = {
            "status": "success",
            "data": {
                "resultType": "streams",
                "result": [
                    {
                        "stream": {"filename": "/var/log/app.log", "host": "srv-app-01"},
                        "values": [
                            [str(int(start_time.timestamp() * 1e9)), "INFO: Service started successfully."],
                            [str(int(end_time.timestamp() * 1e9)), "INFO: Request processed."]
                        ]
                    }
                ]
            }
        }
        return transform_loki_json(raw_json)

class OpenSearchClient(TelemetryClient):
    def fetch(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        # Simulating RAW OpenSearch JSON response
        raw_json = {
            "hits": {
                "hits": [
                    {
                        "_id": "os-log-1234",
                        "_source": {
                            "@timestamp": start_time.isoformat(),
                            "host": {"name": "srv-app-02"},
                            "log": {"file": {"path": "/var/log/access.log"}},
                            "message": "GET /api/v1/status 200 OK"
                        }
                    }
                ]
            }
        }
        return transform_opensearch_json(raw_json)

class JaegerClient(TelemetryClient):
    def fetch(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        # Simulating RAW Jaeger JSON response
        raw_json = {
            "data": [
                {
                    "traceID": "trace-9999",
                    "processes": {
                        "p1": {"serviceName": "srv-app-01"}
                    },
                    "spans": [
                        {
                            "spanID": "span-1234",
                            "startTime": int(start_time.timestamp() * 1e6),
                            "duration": 150000,
                            "processID": "p1",
                            "references": []
                        }
                    ]
                }
            ]
        }
        return transform_jaeger_json(raw_json)

class TempoClient(TelemetryClient):
    def fetch(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        # Simulating RAW Tempo (OTLP) JSON response
        raw_json = {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [{"key": "service.name", "value": {"stringValue": "srv-db-01"}}]
                    },
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "traceId": "trace-9999",
                                    "spanId": "span-5678",
                                    "parentSpanId": "span-1234",
                                    "startTimeUnixNano": str(int(start_time.timestamp() * 1e9)),
                                    "endTimeUnixNano": str(int(start_time.timestamp() * 1e9 + 45000000)),
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        return transform_tempo_json(raw_json)

def get_client(service_name: str, url: str) -> TelemetryClient:
    service_name = service_name.lower()
    if service_name == "prometheus":
        return PrometheusClient(url)
    elif service_name == "loki":
        return LokiClient(url)
    elif service_name == "opensearch":
        return OpenSearchClient(url)
    elif service_name == "jaeger":
        return JaegerClient(url)
    elif service_name == "tempo":
        return TempoClient(url)
    else:
        raise ValueError(f"Unknown service: {service_name}")
