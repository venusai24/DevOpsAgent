import requests
import urllib.parse
import json

prometheus_url = "http://20.194.26.174:9090"
metric_bucket = 'traces_span_metrics_duration_milliseconds_bucket{service_name!~"load-generator|flagd|kafka|valkey-cart|postgresql|wlgen"}'

# Test 3: avg_over_time with NaN filtering
q3 = f'avg_over_time((histogram_quantile(0.95, sum by (le, service_name) (rate({metric_bucket}[3m]))) >= 0)[10m:1m])'

def run_query(q):
    resp = requests.get(f"{prometheus_url}/api/v1/query?query={urllib.parse.quote(q)}")
    return resp.json()

print("--- Q3 (Filtered) ---")
print(json.dumps(run_query(q3), indent=2)[:500])

