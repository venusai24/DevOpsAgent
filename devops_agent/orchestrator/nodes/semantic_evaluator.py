from typing import Any

from devops_agent.core.db.duckdb_client import DuckDBClient
from devops_agent.playbooks.registry import PLAYBOOK_REGISTRY

from ..agents.schemas import SemanticFact
from ..state import InvestigationState


def map_generic_kpi_to_raw(generic_name: str, kpi_map: dict) -> list[str]:
    base_category = generic_name.split('.')[0]
    if "load" in generic_name:
        base_category = "cpu"
        
    raw_kpis = []
    for cid, data in kpi_map.items():
        mapping = data.get("raw_kpi_mapping", {})
        if base_category in mapping:
            raw_kpis.extend(mapping[base_category])
            
    # Sub-filter based on generic intent
    if "load" in generic_name.lower():
        raw_kpis = [k for k in raw_kpis if "load" in k.lower()]
    elif "usage" in generic_name.lower() or "util" in generic_name.lower():
        raw_kpis = [k for k in raw_kpis if "util" in k.lower() or "usage" in k.lower() or "cpu" in k.lower()]
        
    return list(set(raw_kpis))

def evaluate_metric_threshold(rule, state: InvestigationState, db: DuckDBClient, time_filter: str) -> SemanticFact:
    kpi_name = rule.parameters.get("kpi_name")
    condition = rule.parameters.get("condition", ">")
    threshold = rule.parameters.get("threshold", 0.0)
    
    # Very basic query for evaluation. In reality, sustained requires window functions.
    # For MVP of the evaluator, we check if max value breaches threshold.
    metrics_path = state.get("metrics_path")
    if not metrics_path:
        return SemanticFact(rule_id=rule.rule_id, is_true=False, observed_value=None, semantic_statement=rule.semantic_statement_fail + " (No metrics data)")
        
    kpi_map = state.get("discovered_kpi_map", {})
    raw_kpis = map_generic_kpi_to_raw(kpi_name, kpi_map)
    
    if not raw_kpis:
        return SemanticFact(rule_id=rule.rule_id, is_true=False, observed_value=None, semantic_statement=rule.semantic_statement_fail + f" (No raw telemetry maps to {kpi_name})")
        
    kpi_list_str = ", ".join(f"'{k}'" for k in raw_kpis)
    query = f"SELECT max(value) as max_val FROM read_csv_auto('{metrics_path}') WHERE kpi_name IN ({kpi_list_str}) {time_filter}"
    
    try:
        import math
        df = db.query(query)
        max_val = df['max_val'].iloc[0] if not df.empty else None
        
        is_true = False
        if max_val is not None:
            max_val = float(max_val)
            if math.isnan(max_val):
                max_val = None
            else:
                if condition == ">":
                    is_true = max_val > threshold
                elif condition == "<":
                    is_true = max_val < threshold
                elif condition == "==":
                    is_true = max_val == threshold
                
        stmt = rule.semantic_statement_pass if is_true else rule.semantic_statement_fail
        stmt += f" (Observed: {max_val})"
        
        return SemanticFact(rule_id=rule.rule_id, is_true=is_true, observed_value=max_val, semantic_statement=stmt)
    except Exception as e:
        return SemanticFact(rule_id=rule.rule_id, is_true=False, observed_value=None, semantic_statement=f"Error evaluating rule: {e}")

def evaluate_custom_query(rule, state: InvestigationState, db: DuckDBClient, time_filter: str) -> SemanticFact:
    sql = rule.parameters.get("sql", "")
    metrics_path = state.get("metrics_path", "")
    
    # Simple template rendering
    sql = sql.replace("{metrics_path}", metrics_path).replace("{time_filter}", time_filter)
    
    try:
        df = db.query(sql)
        is_true = df['is_true'].iloc[0] if not df.empty else False
        stmt = rule.semantic_statement_pass if is_true else rule.semantic_statement_fail
        return SemanticFact(rule_id=rule.rule_id, is_true=bool(is_true), observed_value=None, semantic_statement=stmt)
    except Exception as e:
        return SemanticFact(rule_id=rule.rule_id, is_true=False, observed_value=None, semantic_statement=f"Error executing custom query: {e}")

def semantic_evaluator_node(state: InvestigationState) -> dict[str, Any]:
    candidates = state.get("match_results", [])
    facts = []
    
    db = DuckDBClient.get_instance()
    
    # Calculate time filter
    time_filter = ""
    time_range = state.get("time_range")
    if time_range and len(time_range) == 2:
        start_epoch = int(time_range[0].timestamp())
        end_epoch = int(time_range[1].timestamp())
        time_filter = f"AND timestamp >= {start_epoch} AND timestamp <= {end_epoch}"
        
    evaluated_rules = set()
        
    for candidate in candidates:
        if candidate.get("fired"):
            playbook = PLAYBOOK_REGISTRY.get(candidate.get("scenario_id"))
            if not playbook:
                continue
            
            for rule in playbook.evaluation_rules:
                if rule.rule_id in evaluated_rules:
                    continue # Skip duplicates
                    
                evaluated_rules.add(rule.rule_id)
                
                if rule.rule_type == "metric_threshold":
                    fact = evaluate_metric_threshold(rule, state, db, time_filter)
                    facts.append(fact.model_dump())
                elif rule.rule_type == "custom_query":
                    fact = evaluate_custom_query(rule, state, db, time_filter)
                    facts.append(fact.model_dump())
                # Future: log_pattern, trace_duration
                else:
                    facts.append(SemanticFact(
                        rule_id=rule.rule_id, 
                        is_true=False, 
                        observed_value=None, 
                        semantic_statement=f"Unsupported rule type: {rule.rule_type}"
                    ).model_dump())
                    
    return {"semantic_facts": facts}
