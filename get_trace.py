import os
from dotenv import load_dotenv
from langsmith import Client
import json

load_dotenv()

client = Client()
project_name = "DevOpsAgent-Incidents"

# Get the most recent run for the project
runs = list(client.list_runs(
    project_name=project_name,
    run_type="llm",
    limit=10,
    order="desc"
))

if not runs:
    print("No runs found.")
else:
    for r in runs:
        if r.name == "rca":
            print(f"Run ID: {r.id}, Name: {r.name}, Status: {r.status}")
            print("Outputs:")
            print(json.dumps(r.outputs, indent=2))
            print("-" * 40)
            break
