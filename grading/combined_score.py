"""
FzIQ Combined Scoring Formula
Fuses human grades with CNN verifier score into a single training signal.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CombinedGrade:
    human_stability: int       # 1-5
    human_consequence: int     # 1-5
    cnn_score: Optional[float] # 0-1, None if CNN not available
    combined: float            # 0-1 final combined score
    num_human_grades: int      # total grades collected so far (affects CNN weight)


def combined_score(
    human_stability: int,
    human_consequence: int,
    cnn_score: Optional[float],
    num_human_grades: int = 0,
) -> CombinedGrade:
    """
    Compute combined score from human grades and CNN verifier.

    Args:
        human_stability: int 1-5 (1=completely wrong, 5=exactly right)
        human_consequence: int 1-5 (1=no resemblance, 5=accurate and specific)
        cnn_score: float 0-1 from CNN verifier, or None if unavailable
        num_human_grades: total human grades collected (used to scale CNN weight)

    Returns:
        CombinedGrade with final combined score 0-1
    
    Score formula:
        - Normalize human average: (avg - 1) / 4  → 0-1
        - CNN weight scales from 0.1 → 0.4 as human grades accumulate (0 → 10000)
        - Combined = human_weight * human_normalized + cnn_weight * cnn_score
        - If CNN unavailable: combined = human_normalized
    """
    human_avg = (human_stability + human_consequence) / 2.0  # 1-5
    human_normalized = (human_avg - 1.0) / 4.0               # 0-1

    if cnn_score is None:
        return CombinedGrade(
            human_stability=human_stability,
            human_consequence=human_consequence,
            cnn_score=None,
            combined=human_normalized,
            num_human_grades=num_human_grades,
        )

    # CNN weight increases as training data grows
    cnn_weight = min(0.4, 0.1 + (num_human_grades / 10000.0) * 0.3)
    human_weight = 1.0 - cnn_weight

    score = human_weight * human_normalized + cnn_weight * cnn_score

    return CombinedGrade(
        human_stability=human_stability,
        human_consequence=human_consequence,
        cnn_score=cnn_score,
        combined=round(score, 4),
        num_human_grades=num_human_grades,
    )
