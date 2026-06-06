import json
import subprocess
import time

def load_scenarios(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def inject_fault(command):
    print(f"Injecting fault: {command}")
    # SREGym fault injection is simulated in this harness, bypass subprocess execution
    return True

def trigger_airs():
    print("Triggering AIRS evaluation...")
    start_time = time.time()
    # Replace this with your actual AIRS invocation (API call, CLI, or direct Python function)
    # airs_response = airs_agent.diagnose_and_resolve()
    
    # Mock response for structure
    airs_response = {
        "inferred_root_cause": "Network latency on DB port",
        "action_taken": "Restarted proxy",
        "chain_of_thought_log": "/logs/airs_run_123.log"
    }
    duration = time.time() - start_time
    return airs_response, duration

def evaluate_run(scenario, airs_response, duration):
    # Basic Evaluation Logic
    cause_match = scenario['expected_root_cause'].lower() in airs_response['inferred_root_cause'].lower()
    
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
        time.sleep(10) 
        
        # 3. Trigger AIRS and collect response
        airs_response, duration = trigger_airs()
        
        # 4. Evaluate and log
        eval_result = evaluate_run(scenario, airs_response, duration)
        results.append(eval_result)
        
        # 5. Reset Environment (CRITICAL)
        # subprocess.run("uv run prek reset", shell=True)
        time.sleep(5)

    print("\n=== Evaluation Complete ===")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()