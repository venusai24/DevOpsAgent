import requests
import json
prometheus_url = "http://20.194.26.174:9090"
resp = requests.get(f"{prometheus_url}/api/v1/series?match[]={{error!=\"\"}}")
print(json.dumps(resp.json().get('data', [])[:5], indent=2))
