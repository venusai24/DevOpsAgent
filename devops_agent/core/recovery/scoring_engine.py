import json
import logging
import math
import os
import yaml
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

class DeterministicScorer:
    def __init__(self, config_path: str = None):
        if not config_path:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "config",
                "evidence_weights.yaml"
            )
        try:
            with open(config_path, "r") as f:
                self.weights = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load evidence weights from {config_path}: {e}")
            self.weights = {}

    def _determine_expected_support_level(self, raw_ref: Dict[str, Any]) -> str:
        """Heuristic to determine if the raw data supports an anomaly."""
        if "max_z_score" in raw_ref:
            z_score = float(raw_ref.get("max_z_score", 0.0))
            if z_score >= 3.0:
                return "anomalous"
            else:
                return "normal"
        elif "match_count" in raw_ref:
            count = int(raw_ref.get("match_count", 0))
            if count > 0:
                return "anomalous"
            else:
                return "normal"
        return "unknown"

    def _is_mismatch(self, llm_support: str, raw_ref: Dict[str, Any]) -> bool:
        expected = self._determine_expected_support_level(raw_ref)
        if expected == "unknown":
            return False
            
        is_strong = llm_support in ["strongly_supports", "strongly_contradicts"]
        is_neutral = llm_support == "neutral"
        
        # If the LLM claims strong support/contradiction, there should be an anomaly
        if is_strong and expected == "normal":
            return True
        # If the LLM claims neutral, there should not be an anomaly
        if is_neutral and expected == "anomalous":
            return True
            
        return False

    def score(
        self,
        hypotheses: List[str],
        evidence_items: List[Dict[str, Any]],
        evidence_log: List[Dict[str, Any]],
        critic_verdicts: List[Dict[str, Any]] = None,
        prior_scores: Dict[str, float] = None,
        topology_graph: Dict[str, List[str]] = None,
        t0: Any = None,
        investigation_cluster: List[str] = None,
        root_cause_candidate: Dict[str, Any] = None
    ) -> Tuple[Dict[str, float], List[str]]:
        
        mismatched_items = []
        hypothesis_log_odds = {h: 0.0 for h in hypotheses}
        if prior_scores:
            for h in hypotheses:
                prob = prior_scores.get(h, 0.0)
                if prob > 0:
                    hypothesis_log_odds[h] = math.log(prob)
                else:
                    hypothesis_log_odds[h] = -20.0

        
        # Build lookup for critic verdicts
        critic_lookup = {}
        if critic_verdicts:
            for cv in critic_verdicts:
                item_id = cv.get("evidence_item_id") if isinstance(cv, dict) else getattr(cv, "evidence_item_id", None)
                if item_id:
                    critic_lookup[item_id] = cv

        # Build lookup for evidence_log (Provenance)
        evidence_lookup = {}
        if evidence_log:
            for ev in evidence_log:
                ev_id = ev.get("evidence_id")
                if ev_id:
                    evidence_lookup[ev_id] = ev

        for item in evidence_items:
            item_dict = item if isinstance(item, dict) else (item.model_dump() if hasattr(item, "model_dump") else item.dict())
            
            # Phase 2: Schema uses evidence_id directly instead of raw_reference
            evidence_id = item_dict.get("evidence_id")
            
            # Layer 5 (Provenance Verification)
            if not evidence_id or evidence_id not in evidence_lookup:
                if evidence_id:
                    mismatched_items.append(evidence_id)
                continue # Ignore hallucinated evidence entirely (weight=0)
                
            raw_ref = evidence_lookup[evidence_id]
            
            llm_support = item_dict.get("directional_support", "neutral")
            evidence_source = item_dict.get("evidence_source", "metric")
            hypothesis_id = item_dict.get("hypothesis_id")
            
            if self._is_mismatch(llm_support, raw_ref):
                if evidence_id:
                    mismatched_items.append(evidence_id)
            
            source_weights = self.weights.get(evidence_source, {})
            weight = source_weights.get(llm_support, 0.0)
            
            if hypothesis_id in hypothesis_log_odds:
                hypothesis_log_odds[hypothesis_id] += weight

        # Layer 3 (Temporal Verification) & Layer 2 (Topological Verification)
        if root_cause_candidate:
            rc_evidence_ids = []
            if "supporting_evidence_ids" in root_cause_candidate:
                rc_evidence_ids = root_cause_candidate["supporting_evidence_ids"]
            elif "evidence_id" in root_cause_candidate:
                rc_evidence_ids = [root_cause_candidate["evidence_id"]]
                
            rc_cmdb_id = root_cause_candidate.get("cmdb_id")
            
            # Layer 3 (Temporal Verification)
            if t0:
                from dateutil.parser import parse
                import datetime
                try:
                    t0_dt = parse(str(t0))
                    for ev_id in rc_evidence_ids:
                        if ev_id in evidence_lookup:
                            raw_ev = evidence_lookup[ev_id]
                            ev_ts = raw_ev.get("anomaly_timestamp") or raw_ev.get("t0_metrics")
                            if ev_ts:
                                ev_dt = parse(str(ev_ts))
                                # If the evidence anomaly happened AFTER T0 (+60s slack), it can't be the root cause.
                                if ev_dt > (t0_dt + datetime.timedelta(seconds=60)):
                                    if ev_id not in mismatched_items:
                                        mismatched_items.append(ev_id)
                except Exception as e:
                    logger.debug(f"Temporal parsing error: {e}")

            # Layer 2 (Topological Verification)
            if topology_graph and investigation_cluster and rc_cmdb_id:
                if rc_cmdb_id not in investigation_cluster:
                    reachable = set()
                    queue = [rc_cmdb_id]
                    while queue:
                        curr = queue.pop(0)
                        if curr not in reachable:
                            reachable.add(curr)
                            queue.extend(topology_graph.get(curr, []))
                    
                    if not any(node in reachable for node in investigation_cluster):
                        # Structurally impossible
                        if rc_evidence_ids:
                            for ev_id in rc_evidence_ids:
                                if ev_id not in mismatched_items:
                                    mismatched_items.append(ev_id)
                        else:
                            mismatched_items.append(f"topo_fail_{rc_cmdb_id}")

        if not hypothesis_log_odds:
            return {}, mismatched_items

        max_odds = max(hypothesis_log_odds.values())
        exp_odds = {h: math.exp(odds - max_odds) for h, odds in hypothesis_log_odds.items()}
        sum_exp = sum(exp_odds.values())
        
        updated_scores = {h: (exp / sum_exp) for h, exp in exp_odds.items()}
        
        return updated_scores, mismatched_items
