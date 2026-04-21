"""
FzIQ Full Pipeline Runner

Runs one complete FzIQ cycle:
  1. Generate scenarios
  2. Agent produces predictions
  3. Simulate ground truth (or mock in test mode)
  4. Grade predictions (mock grades in test mode)
  5. Compute failure gradients
  6. Submit to aggregator
  7. (If --force_aggregate) aggregate and write to opBNB

Usage:
    # Test mode (no real GPU, no opBNB):
    python scripts/run_full_pipeline.py --mode test --num_scenarios 10

    # Production (real model, real opBNB):
    python scripts/run_full_pipeline.py --mode prod --num_scenarios 50
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))


def mock_ground_truth(scenario) -> str:
    """Generate deterministic mock ground truth for test mode."""
    # Harder scenarios are more likely to be unstable
    if scenario.difficulty > 0.6:
        return random.choice(["partial_collapse", "full_collapse", "partial_collapse"])
    elif scenario.difficulty > 0.3:
        return random.choice(["stable", "partial_collapse"])
    else:
        return random.choice(["stable", "stable", "partial_collapse"])


def mock_grade(prediction, ground_truth: str) -> tuple:
    """Generate mock human grades for test mode (returns stability_score, consequence_score)."""
    correct = (prediction.stability == "stable") == (ground_truth == "stable")
    stability_score = random.randint(3, 5) if correct else random.randint(1, 3)
    consequence_score = random.randint(2, 5) if correct else random.randint(1, 3)
    return stability_score, consequence_score


def main():
    parser = argparse.ArgumentParser(description="FzIQ Pipeline Runner")
    parser.add_argument("--mode", choices=["test", "prod"], default="test")
    parser.add_argument("--num_scenarios", type=int, default=10)
    parser.add_argument("--model_path", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--force_aggregate", action="store_true",
                        help="Run aggregation immediately (skip 24hr wait)")
    parser.add_argument("--dry_run_chain", action="store_true",
                        help="Skip actual opBNB transactions (log only)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    logger.info(f"FzIQ Pipeline starting — mode={args.mode}, scenarios={args.num_scenarios}")

    # ─── 1. Generate scenarios ───────────────────────────────────────────────
    from environment.scenario_generator import ScenarioGenerator
    gen = ScenarioGenerator(seed=args.seed)
    scenarios = gen.generate_batch(args.num_scenarios)
    logger.info(f"Generated {len(scenarios)} scenarios")

    if args.mode == "test":
        # ─── TEST MODE: mock model, mock grades ────────────────────────────
        from agent.fziq_agent import AgentPrediction, FailureGradient

        gradients = []
        for i, scenario in enumerate(scenarios):
            # Mock prediction
            prediction = AgentPrediction(
                stability=random.choice(["stable", "unstable"]),
                confidence=random.uniform(0.3, 0.9),
                consequence="The top block slides off and the stack partially collapses.",
                reasoning="The force exceeds the frictional resistance.",
                raw_response="{}",
            )

            # Mock ground truth
            ground_truth = mock_ground_truth(scenario)

            # Mock grading
            stab_score, cons_score = mock_grade(prediction, ground_truth)

            # Combined score
            from grading.combined_score import combined_score
            grade = combined_score(stab_score, cons_score, cnn_score=None, num_human_grades=i)

            # Mock gradient (zero tensors with correct shape signal)
            gradient = FailureGradient(
                gradient={},  # empty in test mode
                scenario_hash=scenario.hash(),
                agent_id="test_agent_001",
                score=grade.combined,
            )
            gradients.append(gradient)

            logger.info(
                f"  Scenario {i+1}: pred={prediction.stability}, gt={ground_truth}, "
                f"score={grade.combined:.3f}"
            )

    else:
        # ─── PROD MODE: real model, real grades ────────────────────────────
        from agent.fziq_agent import FzIQAgent
        from grading.combined_score import combined_score

        agent = FzIQAgent(model_path=args.model_path)
        gradients = []

        for i, scenario in enumerate(scenarios):
            logger.info(f"Processing scenario {i+1}/{len(scenarios)}")

            # Predict
            prediction = agent.predict(scenario)

            # Ground truth (Isaac Lab would go here — mocked for now)
            ground_truth = mock_ground_truth(scenario)

            # Grade (human grader / CNN — mocked for now)
            stab_score, cons_score = mock_grade(prediction, ground_truth)
            grade = combined_score(stab_score, cons_score, cnn_score=None)

            # Compute gradient
            gradient = agent.compute_failure_gradient(
                scenario=scenario,
                prediction=prediction,
                ground_truth_outcome=ground_truth,
                combined_score=grade.combined,
            )
            gradients.append(gradient)

    # ─── 2. Submit to aggregator ─────────────────────────────────────────────
    logger.info(f"\nSubmitting {len(gradients)} gradients to aggregator...")

    if args.mode == "test":
        # Skip actual model loading in test mode
        logger.info("[TEST] Skipping real aggregator — printing summary")
        scores = [g.score for g in gradients]
        logger.info(f"  Avg score: {sum(scores)/len(scores):.3f}")
        logger.info(f"  Min/Max: {min(scores):.3f} / {max(scores):.3f}")
        logger.info(f"\n✓ Test pipeline complete. {len(gradients)} gradients would be submitted.")
    else:
        from aggregator.metamodel_aggregator import MetamodelAggregator
        from agent.model_loader import load_metamodel
        import os

        model_path = os.getenv("FINETUNED_MODEL_PATH", args.model_path)
        model, tokenizer = load_metamodel(model_path)
        agg = MetamodelAggregator(
            model=model,
            on_chain=True,
            dry_run_chain=args.dry_run_chain,
        )

        for g in gradients:
            agg.receive_gradient(g)

        logger.info(f"Aggregator status: {agg.status()}")

        if args.force_aggregate:
            update = agg.aggregate(force=True)
            if update:
                logger.info(f"\n✓ {update.summary()}")
                if update.tx_hash:
                    logger.info(f"  opBNB tx: {update.tx_hash}")

    logger.info("\nFzIQ pipeline complete.")


if __name__ == "__main__":
    main()
