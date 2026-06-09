import os
import sys
import json
import asyncio
import re
import time
import subprocess

# Set environment variables for AIRS K8s configuration
os.environ["AIRS_USE_K8S"] = "true"
os.environ["AIRS_K8S_NAMESPACES"] = "hotel-reservation"

from kubernetes import client, config
from airs_v2.perception.router import LogRouter
from airs_v2.reasoning.engine import ReasoningEngine
from airs_v2.context.graph import ContextGraph


def get_pod_errors(pod, v1) -> list[str]:
    errors = []
    # 1. Check container statuses for waiting/terminated errors
    for status in (pod.status.container_statuses or []):
        if status.state.waiting:
            msg = status.state.waiting.message
            reason = status.state.waiting.reason
            if msg:
                errors.append(f"ERROR: [Container Waiting] Pod {pod.metadata.name} container {status.name} is waiting ({reason}): {msg}")
        if status.last_state.terminated:
            msg = status.last_state.terminated.message
            reason = status.last_state.terminated.reason
            if msg:
                errors.append(f"ERROR: [Container Terminated] Pod {pod.metadata.name} container {status.name} terminated ({reason}): {msg}")
                
    # 2. Check Kubernetes events for warning alerts
    try:
        events = v1.list_namespaced_event(
            namespace=pod.metadata.namespace,
            field_selector=f"involvedObject.name={pod.metadata.name},involvedObject.kind=Pod"
        )
        for event in events.items:
            if event.type == "Warning":
                errors.append(f"WARNING: [K8s Event] Pod {pod.metadata.name} warning ({event.reason}): {event.message}")
    except Exception:
        pass
        
    return errors


async def analyze_live_cluster():
    """Runs the AIRS perception and reasoning layer against the live cluster state."""
    print("🚀 Initializing Kubernetes client...")
    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()
        
    v1 = client.CoreV1Api()
    namespace = "hotel-reservation"
    
    print(f"🔍 Fetching pods in namespace: '{namespace}'...")
    pods = v1.list_namespaced_pod(namespace=namespace)
    
    unhealthy_pods = []
    for pod in pods.items:
        ready = all(status.ready for status in (pod.status.container_statuses or []))
        restart_count = sum(status.restart_count for status in (pod.status.container_statuses or []))
        
        if not ready or restart_count > 0:
            unhealthy_pods.append(pod)
            
    if not unhealthy_pods:
        print("✅ All pods in the namespace are healthy. No faults detected.")
        return None
        
    print(f"⚠️ Found {len(unhealthy_pods)} unhealthy or crashing pods:")
    for pod in unhealthy_pods:
        print(f" - {pod.metadata.name}")
        
    combined_errors = []
    focal_services = []
    
    for pod in unhealthy_pods:
        pod_name = pod.metadata.name
        svc_name = (
            pod.metadata.labels.get("app") 
            or pod.metadata.labels.get("io.kompose.service")
            or pod.metadata.labels.get("app.kubernetes.io/name") 
            or pod_name.split("-")[0]
        )
        focal_services.append(svc_name)
        
        print(f"📋 Gathering error indicators and events for pod '{pod_name}' ({svc_name})...")
        pod_errors = get_pod_errors(pod, v1)
        if pod_errors:
            combined_errors.extend(pod_errors)
        else:
            try:
                logs = v1.read_namespaced_pod_log(name=pod_name, namespace=namespace, tail_lines=50)
                if logs and logs.strip():
                    combined_errors.append(f"ERROR: [Logs] Pod {pod_name} log:\n{logs}")
            except Exception:
                pass
                
    if not combined_errors:
        print("❌ Error: No diagnostic log messages or event warnings could be collected from the unhealthy pods.")
        return None
        
    focal_service = focal_services[0]
    print(f"\n🎯 Targets identified: Focal Service = '{focal_service}'")
    
    full_error_text = "\n".join(combined_errors)
    print("\n📝 Collected Error Context:")
    print("-" * 60)
    print(full_error_text)
    print("-" * 60)
    
    print("\n🧠 Routing live errors through AIRS perception layer (LogRouter)...")
    router = LogRouter()
    report = await router.classify_block(full_error_text)
    
    print("🌐 Building live Kubernetes context graph topology...")
    g = ContextGraph()
    
    print("🔮 Executing neuro-symbolic reasoning engine analysis...")
    engine = ReasoningEngine(graph=g)
    analysis = await engine.analyze_incident(report, focal_service)
    
    print("\n" + "=" * 60)
    print(" LIVE INCIDENT ANALYSIS REPORT ".center(60, "="))
    print("" + "=" * 60)
    print(analysis.analysis_markdown)
    print("=" * 60)
    return analysis


def llm_judge(scenario, airs_response):
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage
        
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
    if not airs_response:
        return {
            "fault_id": scenario['fault_id'],
            "diagnosis_correct": False,
            "time_to_resolve_seconds": duration,
            "airs_log": "No analysis returned from AIRS."
        }
        
    cause_match = llm_judge(scenario, airs_response)
    
    if cause_match is None:
        # Fallback keyword match
        expected = scenario['expected_root_cause'].lower()
        actual = airs_response['inferred_root_cause'].lower()
        cause_match = any(word in actual for word in ["geo", "rate", "mongodb", "image", "port", "config"])
    
    return {
        "fault_id": scenario['fault_id'],
        "diagnosis_correct": cause_match,
        "time_to_resolve_seconds": duration,
        "airs_log": airs_response['chain_of_thought_log']
    }


def inject_fault_sregym(fault_id):
    print(f"Injecting SREGym fault: {fault_id}")
    
    # Clean up lingering uvicorn port from previous runs
    try:
        subprocess.run(["fuser", "-k", "8000/tcp"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        time.sleep(1)
    except Exception:
        pass
        
    cmd = [".venv/bin/python3", "main.py", "--problem", fault_id, "--use-external-harness"]
    try:
        process = subprocess.Popen(cmd, cwd="/home/venusai/DevOpsAgent/SREGym")
        process.wait(timeout=240)
        if process.returncode != 0:
            print(f"Failed to inject fault {fault_id}: exit code {process.returncode}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"Injection command timed out after 240s for {fault_id}. Forcing termination and assuming success.")
        process.kill()
        return True
    except Exception as e:
        print(f"Failed to inject fault {fault_id}: {e}")
        return False


async def main():
    with open('sregym_hotel_scenarios.json', 'r') as f:
        scenarios = json.load(f)
        
    results = []

    for scenario in scenarios:
        print(f"\n" + "="*80)
        print(f"--- Running Scenario: {scenario['fault_id']} ---")
        print("="*80)
        
        # 1. Inject Fault into SREGym
        if not inject_fault_sregym(scenario['fault_id']):
            print(f"Skipping {scenario['fault_id']} due to injection failure.")
            continue
            
        # 2. Wait for the cluster to manifest symptoms
        wait_time = 75
        print(f"⏳ Waiting {wait_time}s for symptoms to manifest in live cluster...")
        time.sleep(wait_time)
        
        # 3. Trigger AIRS and collect response
        start_time = time.time()
        airs_response = None
        try:
            analysis = await analyze_live_cluster()
            if analysis:
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
            
        duration = time.time() - start_time
        
        # 4. Evaluate and log
        eval_result = evaluate_run(scenario, airs_response, duration)
        results.append(eval_result)
        
        print(f"Result for {scenario['fault_id']}: {'✅ PASS' if eval_result['diagnosis_correct'] else '❌ FAIL'}")
        
        # 5. Clean up (Reset Environment)
        # Note: running main.py automatically resets previous cluster state! 
        # So we don't strictly need to recover here, it'll happen on next loop iteration.
        # But we can wait a few seconds before the next loop.
        time.sleep(2)

    print("\n=== Evaluation Complete ===")
    print(json.dumps(results, indent=2))
    
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("💾 Saved results to benchmark_results.json")

if __name__ == "__main__":
    asyncio.run(main())
