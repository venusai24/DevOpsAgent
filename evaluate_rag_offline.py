import os
import sys
import json
import asyncio
from pathlib import Path

# Force offline mode for ContextGraph
os.environ["AIRS_USE_K8S"] = "false"

from airs_v2.perception.router import LogRouter
from airs_v2.reasoning.engine import ReasoningEngine
from airs_v2.context.graph import ContextGraph
from live_harness import evaluate_run

async def evaluate_scenario(dataset_path: Path):
    print(f"\n{'='*80}")
    print(f"--- Running Offline Evaluation: {dataset_path.name} ---")
    print(f"{'='*80}")

    with open(dataset_path, "r") as f:
        data = json.load(f)

    fault_id = data["fault_id"]
    focal_service = data["focal_service"]
    full_error_text = data["full_error_text"]
    topology_snapshot = data["topology_snapshot"]
    scenario_metadata = data["scenario_metadata"]

    if not full_error_text:
        print(f"❌ No errors collected for {fault_id}. Failing scenario.")
        return evaluate_run(scenario_metadata, None, 0)

    # 1. Rehydrate Topology Snapshot
    print(f"🌐 Injecting captured dynamic topology snapshot for {fault_id}...")
    g = ContextGraph()
    g.import_snapshot(topology_snapshot)

    # 2. Perception Layer
    print("\n🧠 Routing offline errors through AIRS perception layer (LogRouter)...")
    router = LogRouter()
    report = await router.classify_block(full_error_text)

    # 3. Reasoning Engine
    print("🔮 Executing neuro-symbolic reasoning engine analysis...")
    engine_airs = ReasoningEngine(graph=g)
    
    # We mock duration to 0 since this runs offline instantly on gathered data
    airs_response = None
    try:
        analysis = await engine_airs.analyze_incident(report, focal_service)
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

    # 4. Evaluate with LLM Judge
    eval_result = evaluate_run(scenario_metadata, airs_response, duration=0)
    print(f"\nResult for {fault_id}: {'✅ PASS' if eval_result['diagnosis_correct'] else '❌ FAIL'}")
    
    return eval_result

async def main():
    dataset_dir = Path("offline_datasets")
    if not dataset_dir.exists() or not list(dataset_dir.glob("*.json")):
        print(f"❌ No datasets found in {dataset_dir}. Run `collect_dataset_cloud.py` first!")
        sys.exit(1)

    results = []
    for dataset_file in sorted(dataset_dir.glob("*.json")):
        result = await evaluate_scenario(dataset_file)
        results.append(result)

    print("\n=== Offline Evaluation Complete ===")
    print(json.dumps(results, indent=2))

    with open('offline_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("💾 Saved results to offline_benchmark_results.json")

if __name__ == "__main__":
    asyncio.run(main())
