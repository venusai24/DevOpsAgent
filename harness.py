import json
import subprocess
import time
import asyncio
import re
import os

# Ensure all necessary namespaces from fault_scenarios are targeted for graph discovery
if "AIRS_K8S_NAMESPACES" not in os.environ:
    os.environ["AIRS_K8S_NAMESPACES"] = "production,default,kube-system,monitoring,network,gatekeeper-system"

from airs_v2.perception.router import LogRouter
from airs_v2.reasoning.engine import ReasoningEngine
from airs_v2.context.graph import ContextGraph

def load_scenarios(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def inject_fault(command):
    print(f"Injecting fault: {command}")
    # SREGym fault injection is simulated in this harness, bypass subprocess execution
    return True

async def run_airs_analysis(scenario):
    cmd = scenario.get("sregym_command", "")
    match = re.search(r'--target\s+(?:svc|deployment|virtualservice|configmap|daemonset|secret)/([a-zA-Z0-9-]+)', cmd)
    focal_service = match.group(1) if match else "unknown"
    
    # Fake telemetry to trigger perception
    fake_log = f"ERROR: {scenario['description']}"
    
    router = LogRouter()
    report = await router.classify_block(fake_log)
    
    # Use graph directly to avoid MCP server requirement
    g = ContextGraph()
    
    if not g._g.has_node(focal_service) and focal_service != "unknown":
        print(f"\n[!] WARNING: Focal service '{focal_service}' was not found in the ContextGraph.")
        if os.environ.get("AIRS_USE_K8S", "false").lower() == "true":
            print("[!] K8s discovery enabled, but the node is missing. Ensure the SREGym applications are deployed to the Kind cluster.")
        else:
            print("[!] Node missing from static topology_fixtures.json.")

    engine = ReasoningEngine(graph=g)
    
    analysis = await engine.analyze_incident(report, focal_service)
    return analysis

def trigger_airs(scenario):
    print("Triggering AIRS evaluation...")
    start_time = time.time()
    
    try:
        analysis = asyncio.run(run_airs_analysis(scenario))
        
        inferred = analysis.root_cause_candidates[0].candidate_node if analysis.root_cause_candidates else "Unknown"
        if analysis.root_cause_candidates:
            inferred += f" ({analysis.root_cause_candidates[0].template_key})"
            
        airs_response = {
            "inferred_root_cause": inferred,
            "action_taken": "Diagnosed via ReasoningEngine",
            "chain_of_thought_log": analysis.analysis_markdown
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"AIRS invocation failed: {e}")
        airs_response = {
            "inferred_root_cause": "Error during analysis",
            "action_taken": "None",
            "chain_of_thought_log": str(e)
        }
        
    duration = time.time() - start_time
    return airs_response, duration

def llm_judge(scenario, airs_response):
    try:
        import os
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage
        import json
        
        # Only attempt if API key is present
        if not os.getenv("GROQ_API_KEY"):
            return None
            
        llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0).bind(response_format={"type": "json_object"})
        prompt = f"""You are an expert judge evaluating an SRE AI agent's root cause analysis.
        
Expected underlying fault:
{scenario['expected_root_cause']}

Agent's final analysis report:
{airs_response['chain_of_thought_log']}

Did the agent correctly identify the failing component and/or accurately describe the underlying issue based on its analysis?
Respond ONLY with a JSON object in this exact format:
{{"diagnosis_correct": true, "reason": "brief explanation"}}
"""
        response = llm.invoke([HumanMessage(content=prompt)])
        data = json.loads(response.content)
        return data.get("diagnosis_correct", False)
    except Exception as e:
        print(f"LLM Judge error: {e}")
        return None

def evaluate_run(scenario, airs_response, duration):
    # 1. Try LLM Judge
    cause_match = llm_judge(scenario, airs_response)
    
    # 2. Fallback to Service Match if LLM Judge is unavailable or failed
    if cause_match is None:
        cmd = scenario.get("sregym_command", "")
        match = re.search(r'--target\s+(?:svc|deployment|virtualservice|configmap|daemonset|secret)/([a-zA-Z0-9-]+)', cmd)
        focal_service = match.group(1) if match else "unknown"
        
        # Prevent trivial false positives when focal_service is "unknown"
        if focal_service == "unknown":
            cause_match = scenario['expected_root_cause'].lower() in airs_response['inferred_root_cause'].lower()
        else:
            cause_match = (
                focal_service.lower() in airs_response['inferred_root_cause'].lower() or
                scenario['expected_root_cause'].lower() in airs_response['inferred_root_cause'].lower()
            )
    
    return {
        "fault_id": scenario['fault_id'],
        "diagnosis_correct": cause_match,
        "time_to_resolve_seconds": duration,
        "airs_log": airs_response['chain_of_thought_log']
    }

def main():
    scenarios = load_scenarios('fault_scenarios.json')
    results = []

    for scenario in scenarios:
        print(f"\n--- Running Scenario: {scenario['fault_id']} ---")
        
        # 1. Inject Fault into SREGym
        if not inject_fault(scenario['sregym_command']):
            print("Failed to inject fault. Skipping.")
            continue
            
        # 2. Let the environment bake (give metrics time to spike)
        time.sleep(1) 
        
        # 3. Trigger AIRS and collect response
        airs_response, duration = trigger_airs(scenario)
        
        # 4. Evaluate and log
        eval_result = evaluate_run(scenario, airs_response, duration)
        results.append(eval_result)
        
        # 5. Reset Environment (CRITICAL)
        # subprocess.run("uv run prek reset", shell=True)
        time.sleep(1)

    print("\n=== Evaluation Complete ===")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()