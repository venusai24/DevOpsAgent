import os
from dotenv import load_dotenv
from langsmith import Client
import json

load_dotenv()

client = Client()
project_name = "DevOpsAgent-Incidents"
trace_id = "06a3d9e6-d186-42cc-861f-58d6e86238d2"

print(f"Fetching trace for Run ID: {trace_id}")

all_spans = list(client.list_runs(
    project_name=project_name,
    trace_id=trace_id
))

# Filter LLM and Tool spans
interesting_nodes = [s for s in all_spans if s.run_type in ("llm", "tool")]
# Sort by start_time to see the chronological order
interesting_nodes.sort(key=lambda x: x.start_time if x.start_time else getattr(x, 'start_time', 0))

with open("trace_analysis.md", "w") as f:
    f.write(f"# LangSmith Trace Analysis for Trace: {trace_id}\n\n")
    
    if not interesting_nodes:
        f.write("No interesting spans found in this trace.\n")
    
    for s in interesting_nodes:
        f.write(f"## {s.run_type.upper()} Call: {s.id} (Name: {s.name})\n")
        
        # Try to parse the outputs
        try:
            if s.outputs and "generations" in s.outputs:
                generations = s.outputs["generations"]
                if generations and isinstance(generations, list) and len(generations) > 0:
                    gen = generations[0]
                    if isinstance(gen, list):
                        gen = gen[0]
                    msg = gen.get("message", {})
                    kwargs = msg.get("kwargs", {})
                    
                    content = kwargs.get("content", "")
                    tool_calls = kwargs.get("tool_calls", [])
                    
                    if content:
                        f.write("### AI Response\n")
                        f.write(f"```text\n{content}\n```\n\n")
                    
                    if tool_calls:
                        f.write("### Tool Calls\n")
                        for t in tool_calls:
                            f.write(f"- **{t.get('name')}**: `{json.dumps(t.get('args', {}))}`\n")
                    else:
                        f.write("### Tool Calls\n*None*\n")
            else:
                f.write("### Outputs\n")
                f.write(f"```json\n{json.dumps(s.outputs, indent=2, default=str)}\n```\n")
        except Exception as e:
            f.write(f"Error parsing outputs: {e}\n")
            f.write(f"Raw outputs:\n```json\n{json.dumps(s.outputs, indent=2, default=str)}\n```\n")
            
        f.write("\n---\n\n")
        
print("Dumped to trace_analysis.md")
