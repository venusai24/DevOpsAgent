"""
Information Pursuit Models.

Tracks the missing mass state — the epistemic gap the agent is
actively working to fill — across the investigation lifecycle.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class InformationPursuitState(BaseModel):
    """
    Information Pursuit (IP) tracking state.

    current_missing_mass:  M_t(λ) — current estimated epistemic uncertainty.
    previous_missing_mass: M_{t-1}(λ) — previous step's mass (for delta calc).
    missing_mass_delta:    |M_t - M_{t-1}| — uncertainty reduction per hop.
    epsilon_threshold:     ε — plateau detection ceiling.
    consecutive_low_delta: Count of consecutive hops where Δ < ε.
    filled_evidence_categories: Set of evidence categories already filled.
    open_evidence_categories:   Evidence categories still needed.
    """
    current_missing_mass: float = Field(default=1.0, ge=0.0, le=1.0)
    previous_missing_mass: float = Field(default=1.0, ge=0.0, le=1.0)
    missing_mass_delta: float = Field(default=0.0, ge=0.0)
    epsilon_threshold: float = Field(default=0.05, gt=0.0, lt=1.0)
    consecutive_low_delta: int = Field(default=0, ge=0)
    filled_evidence_categories: list[str] = Field(default_factory=list)
    open_evidence_categories: list[str] = Field(default_factory=list)

    @property
    def is_plateau(self) -> bool:
        """True when missing mass reduction has stalled (≥ 3 consecutive low delta)."""
        return self.consecutive_low_delta >= 3

    @property
    def is_complete(self) -> bool:
        """True when missing mass is below the epsilon threshold."""
        return self.current_missing_mass <= self.epsilon_threshold
