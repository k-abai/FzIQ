"""
TD-Style Value Function for FzIQ Scenario Prioritization

Tracks which scenario types are most valuable for training signal.
Uses TD(0) learning to assign credit based on future improvements,
not just immediate score changes.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class ResearchValueFunction:
    """
    Tracks the training value of different scenario configurations.
    
    Uses temporal difference learning to prioritize scenarios that
    lead to durable model improvements over time.
    
    State space: discretized (num_blocks, difficulty_decile)
    Reward: change in metamodel benchmark score after a round
    
    Example:
        vf = ResearchValueFunction()
        # After each training round with benchmark delta:
        vf.update(prev_scenario, reward=+0.03, next_scenario)
        # When generating next batch:
        priority = vf.should_prioritize(candidate_scenario)
    """

    def __init__(
        self,
        alpha: float = 0.1,    # TD learning rate
        gamma: float = 0.95,   # discount factor
        persist_path: Optional[str] = None,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.V: Dict[str, float] = {}
        self.visit_count: Dict[str, int] = {}
        self._persist_path = persist_path

        if persist_path and Path(persist_path).exists():
            self._load(persist_path)

    def state_from_scenario(self, scenario) -> str:
        """
        Discretize scenario into a state string.
        State = (num_blocks, difficulty_decile)
        """
        decile = int(scenario.difficulty_score() * 10)
        return f"blocks={scenario.num_blocks}_diff={decile}"

    def update(self, scenario, reward: float, next_scenario=None):
        """
        TD(0) update: V(s) ← V(s) + α * [r + γ*V(s') - V(s)]
        
        Args:
            scenario: current BlockStackScenario
            reward: scalar feedback (e.g. Δbenchmark score after round)
            next_scenario: next scenario if available, else bootstraps with V(s)=0
        """
        s = self.state_from_scenario(scenario)
        V_s = self.V.get(s, 0.0)

        if next_scenario is not None:
            s_next = self.state_from_scenario(next_scenario)
            V_s_next = self.V.get(s_next, 0.0)
        else:
            V_s_next = 0.0

        td_error = reward + self.gamma * V_s_next - V_s
        self.V[s] = V_s + self.alpha * td_error
        self.visit_count[s] = self.visit_count.get(s, 0) + 1

        logger.debug(f"TD update: state={s}, V={V_s:.4f} → {self.V[s]:.4f}, td_error={td_error:.4f}")

        if self._persist_path:
            self._save(self._persist_path)

    def should_prioritize(self, scenario) -> float:
        """
        Returns priority score 0-1 for a candidate scenario.
        Higher = more valuable to train on.
        Unseen states get 0.5 (exploration bonus).
        """
        s = self.state_from_scenario(scenario)
        return self.V.get(s, 0.5)

    def top_states(self, n: int = 10) -> list:
        """Return the N highest-value scenario states."""
        sorted_states = sorted(self.V.items(), key=lambda x: x[1], reverse=True)
        return sorted_states[:n]

    def _save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"V": self.V, "visit_count": self.visit_count}, f, indent=2)

    def _load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.V = data.get("V", {})
        self.visit_count = data.get("visit_count", {})
        logger.info(f"Loaded value function: {len(self.V)} states")
