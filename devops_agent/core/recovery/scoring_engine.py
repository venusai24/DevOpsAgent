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
        prior_scores: Dict[str, float] = None
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

        for item in evidence_items:
            item_dict = item if isinstance(item, dict) else (item.model_dump() if hasattr(item, "model_dump") else item.dict())
            
            raw_ref = item_dict.get("raw_reference", {})
            evidence_id = raw_ref.get("evidence_id")
            
            llm_support = item_dict.get("directional_support", "neutral")
            evidence_source = item_dict.get("evidence_source", "metric")
            hypothesis_id = item_dict.get("hypothesis_id")
            
            if self._is_mismatch(llm_support, raw_ref):
                if evidence_id:
                    mismatched_items.append(evidence_id)
            
            if evidence_id and evidence_id in critic_lookup:
                verdict = critic_lookup[evidence_id]
                v_type = verdict.get("verdict") if isinstance(verdict, dict) else getattr(verdict, "verdict", None)
                if v_type == "override":
                    c_support = verdict.get("corrected_support") if isinstance(verdict, dict) else getattr(verdict, "corrected_support", None)
                    if c_support:
                        llm_support = c_support
            
            source_weights = self.weights.get(evidence_source, {})
            weight = source_weights.get(llm_support, 0.0)
            
            if hypothesis_id in hypothesis_log_odds:
                hypothesis_log_odds[hypothesis_id] += weight

        if not hypothesis_log_odds:
            return {}, mismatched_items

        max_odds = max(hypothesis_log_odds.values())
        exp_odds = {h: math.exp(odds - max_odds) for h, odds in hypothesis_log_odds.items()}
        sum_exp = sum(exp_odds.values())
        
        updated_scores = {h: (exp / sum_exp) for h, exp in exp_odds.items()}
        
        return updated_scores, mismatched_items
