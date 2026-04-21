"""
FzIQ Metamodel Aggregator

Collects failure gradients from all agents over a 24-hour window.
Applies weighted federated averaging to update the shared metamodel.
Writes the update record to opBNB blockchain.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import torch

from .gradient_buffer import GradientBuffer
from .model_hash import compute_model_hash
from .onchain_logger import OnChainLogger

logger = logging.getLogger(__name__)


@dataclass
class ModelUpdate:
    """Record of a single metamodel update round."""
    round: int
    num_gradients: int
    agent_ids: List[str]
    scenario_hashes: List[str]
    scores: List[float]
    model_hash_before: str
    model_hash_after: str
    timestamp: float = field(default_factory=time.time)
    tx_hash: Optional[str] = None      # opBNB transaction hash

    def summary(self) -> str:
        return (
            f"Round {self.round}: {self.num_gradients} gradients from "
            f"{len(set(self.agent_ids))} agents, "
            f"hash {self.model_hash_before[:8]}→{self.model_hash_after[:8]}"
        )


class MetamodelAggregator:
    """
    FzIQ federated metamodel aggregator.

    Runs as a service. Agents call receive_gradient() as they complete
    scenario evaluations. Every 24 hours (or on demand), aggregate()
    applies weighted gradient averaging and updates the metamodel.

    Weighting scheme:
        weight = (1 - score) * reliability
        - Low score → large weight (model was very wrong → needs bigger correction)
        - High reliability agent → larger weight (trusted contributor)

    Example:
        aggregator = MetamodelAggregator(model, aggregation_interval_hours=24)
        # Agents call:
        aggregator.receive_gradient(gradient)
        # Aggregator checks periodically:
        if aggregator.should_aggregate():
            update = aggregator.aggregate()
    """

    def __init__(
        self,
        model,
        aggregation_interval_hours: int = 24,
        min_gradients: int = 5,
        learning_rate: float = 1e-4,
        on_chain: bool = True,
        dry_run_chain: bool = False,
    ):
        self.model = model
        self.aggregation_interval = aggregation_interval_hours * 3600
        self.min_gradients = min_gradients
        self.learning_rate = learning_rate
        self.dry_run_chain = dry_run_chain

        self.gradient_buffer = GradientBuffer(persist_path="data/gradient_log.jsonl")
        self.on_chain_logger = OnChainLogger() if on_chain else None

        self.current_round = 0
        self.previous_hash = compute_model_hash(model)
        self.last_aggregation = time.time()

        # Agent reliability tracking: agent_id → running average score
        self.agent_reliability: Dict[str, float] = {}
        self.agent_contribution_count: Dict[str, int] = {}

        logger.info(f"Aggregator initialized. Model hash: {self.previous_hash[:16]}...")

    def receive_gradient(self, gradient) -> int:
        """
        Accept a failure gradient from an agent.
        
        Args:
            gradient: FailureGradient instance
        Returns:
            Current buffer size
        """
        self._update_reliability(gradient.agent_id, gradient.score)
        return self.gradient_buffer.add(gradient)

    def should_aggregate(self) -> bool:
        """True if 24-hour window has passed and buffer has enough gradients."""
        time_ready = (time.time() - self.last_aggregation) >= self.aggregation_interval
        return time_ready and self.gradient_buffer.is_ready(self.min_gradients)

    def aggregate(self, force: bool = False) -> Optional[ModelUpdate]:
        """
        Run one federated averaging round.
        
        Args:
            force: if True, aggregate even if window hasn't elapsed
        Returns:
            ModelUpdate record (also written to opBNB)
        """
        if not force and not self.should_aggregate():
            logger.info("Aggregation conditions not met — skipping")
            return None

        gradients = self.gradient_buffer.get_all()
        if not gradients:
            logger.warning("No gradients in buffer — skipping aggregation")
            return None

        logger.info(f"Starting aggregation round {self.current_round}: {len(gradients)} gradients")

        # Compute per-gradient weights
        weights = []
        for g in gradients:
            reliability = self.agent_reliability.get(g.agent_id, 0.5)
            # Low score = high update weight; high reliability = trusted contributor
            w = (1.0 - g.score) * reliability
            weights.append(max(w, 1e-6))  # avoid zero weights

        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Weighted gradient average over LoRA parameters
        if gradients[0].gradient:
            param_names = list(gradients[0].gradient.keys())
            aggregated = {}
            for name in param_names:
                tensors = [g.gradient[name].to(torch.float32) for g in gradients if name in g.gradient]
                if not tensors:
                    continue
                ws = normalized_weights[:len(tensors)]
                aggregated[name] = sum(w * t for w, t in zip(ws, tensors))

            # Apply gradient update
            self.model.train()
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in aggregated:
                        param.data -= self.learning_rate * aggregated[name].to(param.device)
            self.model.eval()

        # Compute new model hash
        new_hash = compute_model_hash(self.model)

        # Collect metadata for on-chain log
        agent_ids = [g.agent_id for g in gradients]
        scenario_hashes = [g.scenario_hash for g in gradients]
        scores = [g.score for g in gradients]

        # Write to opBNB
        tx_hash = None
        if self.on_chain_logger:
            try:
                tx_hash = self.on_chain_logger.log_round(
                    model_hash_before=self.previous_hash,
                    model_hash_after=new_hash,
                    scenario_hashes=scenario_hashes,
                    agent_ids=agent_ids,
                    scores=scores,
                    dry_run=self.dry_run_chain,
                )
            except Exception as e:
                logger.error(f"On-chain logging failed: {e} (training continues)")

        update = ModelUpdate(
            round=self.current_round,
            num_gradients=len(gradients),
            agent_ids=agent_ids,
            scenario_hashes=scenario_hashes,
            scores=scores,
            model_hash_before=self.previous_hash,
            model_hash_after=new_hash,
            tx_hash=tx_hash,
        )

        logger.info(update.summary())

        # Reset for next round
        self.gradient_buffer.clear()
        self.previous_hash = new_hash
        self.current_round += 1
        self.last_aggregation = time.time()

        return update

    def _update_reliability(self, agent_id: str, score: float):
        """Running average of agent scores as reliability metric."""
        count = self.agent_contribution_count.get(agent_id, 0)
        prev_avg = self.agent_reliability.get(agent_id, 0.5)
        new_avg = (prev_avg * count + score) / (count + 1)
        self.agent_reliability[agent_id] = new_avg
        self.agent_contribution_count[agent_id] = count + 1

    def status(self) -> dict:
        """Current aggregator status."""
        return {
            "current_round": self.current_round,
            "buffer_size": self.gradient_buffer.size(),
            "buffer_summary": self.gradient_buffer.summary(),
            "seconds_until_next_aggregation": max(
                0, self.aggregation_interval - (time.time() - self.last_aggregation)
            ),
            "model_hash": self.previous_hash[:16] + "...",
            "num_agents": len(self.agent_reliability),
        }
