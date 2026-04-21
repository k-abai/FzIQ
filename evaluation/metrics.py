"""
FzIQ Evaluation Metrics

Stability Accuracy: binary % correct stability predictions
Consequence Accuracy: % of consequence descriptions scored ≥4/5 by human panel
Generalization Gap: delta between in-distribution and out-of-distribution performance
Sample Efficiency: accuracy as a function of training gradient count
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import statistics


@dataclass
class EvaluationResult:
    """Results from evaluating a model on FzIQ-Bench v1."""
    model_name: str
    model_hash: Optional[str]
    num_scenarios: int

    # Core metrics
    stability_accuracy: float           # 0-1
    consequence_accuracy: float         # 0-1 (proportion scored ≥4/5)
    avg_consequence_score: float        # mean human score 1-5

    # Stratified by difficulty level
    level1_stability: float = 0.0
    level2_stability: float = 0.0
    level3_stability: float = 0.0

    # Generalization
    in_distribution_stability: float = 0.0
    out_of_distribution_stability: float = 0.0
    generalization_gap: float = 0.0

    # Sample efficiency
    gradients_used: int = 0

    def summary(self) -> str:
        return (
            f"{self.model_name}: "
            f"Stability={self.stability_accuracy:.1%}, "
            f"Consequence={self.consequence_accuracy:.1%} "
            f"(n={self.num_scenarios})"
        )

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "stability_accuracy": round(self.stability_accuracy, 4),
            "consequence_accuracy": round(self.consequence_accuracy, 4),
            "avg_consequence_score": round(self.avg_consequence_score, 3),
            "level1_stability": round(self.level1_stability, 4),
            "level2_stability": round(self.level2_stability, 4),
            "level3_stability": round(self.level3_stability, 4),
            "generalization_gap": round(self.generalization_gap, 4),
            "gradients_used": self.gradients_used,
        }


class StabilityAccuracy:
    """
    Computes binary stability prediction accuracy.
    Ground truth: "stable" or any collapse variant → "unstable"
    """

    @staticmethod
    def compute(
        predictions: List[str],         # "stable" | "unstable"
        ground_truths: List[str],        # "stable" | "partial_collapse" | "full_collapse"
    ) -> float:
        """
        Args:
            predictions: list of agent stability predictions
            ground_truths: list of simulation outcomes
        Returns:
            Accuracy 0-1
        """
        assert len(predictions) == len(ground_truths), "Length mismatch"

        def binarize(outcome: str) -> str:
            return "stable" if outcome == "stable" else "unstable"

        correct = sum(
            1 for pred, gt in zip(predictions, ground_truths)
            if pred == binarize(gt)
        )
        return correct / len(predictions) if predictions else 0.0

    @staticmethod
    def by_difficulty(
        predictions: List[str],
        ground_truths: List[str],
        difficulty_levels: List[int],   # 1, 2, or 3
    ) -> Dict[int, float]:
        """Compute stability accuracy stratified by difficulty level."""
        results = {}
        for level in [1, 2, 3]:
            indices = [i for i, d in enumerate(difficulty_levels) if d == level]
            if not indices:
                continue
            level_preds = [predictions[i] for i in indices]
            level_gts = [ground_truths[i] for i in indices]
            results[level] = StabilityAccuracy.compute(level_preds, level_gts)
        return results


class ConsequenceAccuracy:
    """
    Computes consequence description accuracy.
    A prediction is "correct" if human panel scores it ≥4/5.
    """

    CORRECT_THRESHOLD = 4  # human score ≥ 4 = correct

    @staticmethod
    def compute(human_scores: List[float]) -> float:
        """
        Args:
            human_scores: list of human consequence scores 1-5
        Returns:
            Proportion scoring ≥ threshold
        """
        if not human_scores:
            return 0.0
        correct = sum(1 for s in human_scores if s >= ConsequenceAccuracy.CORRECT_THRESHOLD)
        return correct / len(human_scores)

    @staticmethod
    def mean_score(human_scores: List[float]) -> float:
        """Mean consequence score (1-5)."""
        return statistics.mean(human_scores) if human_scores else 0.0


class GeneralizationGap:
    """
    Computes generalization gap: in-distribution vs. out-of-distribution accuracy delta.
    """

    @staticmethod
    def compute(in_dist_accuracy: float, out_dist_accuracy: float) -> float:
        """
        Positive gap means model performs worse on out-of-distribution scenarios.
        """
        return in_dist_accuracy - out_dist_accuracy


def compute_sample_efficiency(
    accuracy_by_gradient_count: Dict[int, float],
) -> Dict[str, float]:
    """
    Compute sample efficiency metrics.
    
    Args:
        accuracy_by_gradient_count: {num_gradients: accuracy} dict
    Returns:
        Summary statistics
    """
    if not accuracy_by_gradient_count:
        return {}

    sorted_items = sorted(accuracy_by_gradient_count.items())
    counts = [x[0] for x in sorted_items]
    accs = [x[1] for x in sorted_items]

    return {
        "accuracy_at_100": accuracy_by_gradient_count.get(100, None),
        "accuracy_at_500": accuracy_by_gradient_count.get(500, None),
        "accuracy_at_1000": accuracy_by_gradient_count.get(1000, None),
        "peak_accuracy": max(accs),
        "peak_at_gradients": counts[accs.index(max(accs))],
    }
