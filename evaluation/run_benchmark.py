"""
FzIQ Benchmark Runner

Evaluates a model on FzIQ-Bench v1 (500 held-out scenarios).
Reports: Stability Accuracy, Consequence Accuracy, Generalization Gap, Sample Efficiency.

Usage:
    python evaluation/run_benchmark.py \
        --model_path metamodel/checkpoints/latest \
        --benchmark_path environment/benchmark/fziq_bench_v1.json \
        --output results/round_001_eval.json
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List

from agent.fziq_agent import FzIQAgent
from evaluation.metrics import (
    StabilityAccuracy, ConsequenceAccuracy, EvaluationResult, GeneralizationGap
)
from environment.scenario_generator import BlockStackScenario

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_benchmark(path: str) -> List[dict]:
    """Load FzIQ-Bench v1 scenarios from JSON."""
    with open(path) as f:
        return json.load(f)


def run_evaluation(
    agent: FzIQAgent,
    benchmark_scenarios: List[dict],
    model_name: str = "unknown",
    model_hash: str = None,
) -> EvaluationResult:
    """
    Run a model against all benchmark scenarios and compute metrics.
    
    Note: Consequence accuracy requires human grading of predictions.
    This function collects predictions and stability accuracy.
    Consequence accuracy is filled in after human grading.
    """
    predictions = []
    ground_truths = []
    difficulty_levels = []

    logger.info(f"Running benchmark: {len(benchmark_scenarios)} scenarios")
    start_time = time.time()

    for i, item in enumerate(benchmark_scenarios):
        # Reconstruct scenario object from JSON
        scenario = _dict_to_scenario(item)
        
        try:
            prediction = agent.predict(scenario)
            predictions.append(prediction.stability)
        except Exception as e:
            logger.warning(f"Prediction failed for scenario {i}: {e}")
            predictions.append("unstable")  # default fallback

        ground_truths.append(item.get("ground_truth_outcome", "stable"))
        difficulty_levels.append(item.get("difficulty_level", 2))

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            logger.info(f"  {i+1}/{len(benchmark_scenarios)} ({elapsed:.0f}s elapsed)")

    # Compute metrics
    stability_acc = StabilityAccuracy.compute(predictions, ground_truths)
    by_level = StabilityAccuracy.by_difficulty(predictions, ground_truths, difficulty_levels)

    return EvaluationResult(
        model_name=model_name,
        model_hash=model_hash,
        num_scenarios=len(benchmark_scenarios),
        stability_accuracy=stability_acc,
        consequence_accuracy=0.0,    # filled after human grading
        avg_consequence_score=0.0,   # filled after human grading
        level1_stability=by_level.get(1, 0.0),
        level2_stability=by_level.get(2, 0.0),
        level3_stability=by_level.get(3, 0.0),
    )


def _dict_to_scenario(item: dict):
    """Reconstruct a minimal scenario for agent.predict() from benchmark JSON."""
    from environment.scenario_generator import BlockStackScenario, BlockProperties
    blocks = [
        BlockProperties(
            mass=b["mass"],
            dimensions=tuple(b["dimensions"]),
            friction=b["friction"],
            material=b.get("material", "wood"),
        )
        for b in item.get("blocks", [])
    ]
    scenario = BlockStackScenario(
        scenario_id=item.get("scenario_id", "bench"),
        num_blocks=item["num_blocks"],
        blocks=blocks,
        surface_friction=item["surface_friction"],
        surface_tilt=item["surface_tilt"],
        force_magnitude=item["force_magnitude"],
        force_direction=tuple(item["force_direction"]),
        force_application_point=(0.0, 0.0, 0.0),
        scenario_text=item["scenario_text"],
    )
    return scenario


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--benchmark_path", default="environment/benchmark/fziq_bench_v1.json")
    parser.add_argument("--output", default="results/eval.json")
    parser.add_argument("--model_name", default=None)
    args = parser.parse_args()

    benchmark = load_benchmark(args.benchmark_path)

    agent = FzIQAgent(model_path=args.model_path)
    model_name = args.model_name or Path(args.model_path).name

    result = run_evaluation(
        agent=agent,
        benchmark_scenarios=benchmark,
        model_name=model_name,
    )

    logger.info(result.summary())

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
