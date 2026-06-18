import requests
import json
import urllib.parse

prometheus_url = "http://20.194.26.174:9090"
query_1m = 'rate(traces_span_metrics_calls_total{status_code="STATUS_CODE_ERROR"}[1m])'
query_2m = 'rate(traces_span_metrics_calls_total{status_code="STATUS_CODE_ERROR"}[2m])'
query_3m = 'rate(traces_span_metrics_calls_total{status_code="STATUS_CODE_ERROR"}[3m])'

for q in [query_1m, query_2m, query_3m]:
    encoded_query = urllib.parse.quote(q)
    resp = requests.get(f"{prometheus_url}/api/v1/query?query={encoded_query}").json()
    count = len(resp.get('data', {}).get('result', []))
    print(f"Query: {q[-4:-1]} -> {count} results")

