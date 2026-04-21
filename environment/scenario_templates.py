"""
Natural language generation templates for FzIQ scenarios.
Converts simulation parameters into human-readable scenario descriptions.
"""

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .scenario_generator import BlockStackScenario


POSITION_LABELS = ["bottom", "second from bottom", "middle", "second from top", "top"]
MASS_LABELS = [(0.5, "very light"), (1.2, "light"), (2.5, "medium"), (3.8, "heavy"), (5.0, "very heavy")]
FORCE_LABELS = [(2, "gentle"), (6, "moderate"), (12, "strong"), (18, "very strong"), (20, "forceful")]
TILT_LABELS = [(2, "almost flat"), (5, "slightly tilted"), (10, "noticeably tilted"), (15, "significantly tilted")]
SIZE_LABELS_NL = [(0.08, "small"), (0.14, "medium-sized"), (0.22, "large"), (0.30, "wide")]

DIRECTION_LABELS = {
    (1, 0): "from the left",
    (-1, 0): "from the right",
    (0, 1): "from the front",
    (0, -1): "from behind",
}


def _mass_label(mass: float) -> str:
    for threshold, label in MASS_LABELS:
        if mass <= threshold:
            return label
    return "very heavy"


def _force_label(force: float) -> str:
    for threshold, label in FORCE_LABELS:
        if force <= threshold:
            return label
    return "forceful"


def _tilt_label(tilt: float) -> str:
    for threshold, label in TILT_LABELS:
        if tilt <= threshold:
            return label
    return "significantly tilted"


def _size_label(dim: float) -> str:
    for threshold, label in SIZE_LABELS_NL:
        if dim <= threshold:
            return label
    return "wide"


def _direction_label(direction: tuple) -> str:
    """Approximate direction to nearest cardinal."""
    x, y = direction[0], direction[1]
    if abs(x) > abs(y):
        return "from the left" if x > 0 else "from the right"
    return "from the front" if y > 0 else "from behind"


class NaturalLanguageGenerator:
    """Converts BlockStackScenario parameters into natural language descriptions."""

    @staticmethod
    def generate(scenario: "BlockStackScenario") -> str:
        n = scenario.num_blocks
        blocks = scenario.blocks
        tilt = scenario.surface_tilt
        force = scenario.force_magnitude
        direction = scenario.force_direction

        # Surface description
        if tilt < 1.0:
            surface_desc = "a flat surface"
        else:
            surface_desc = f"a {_tilt_label(tilt)} surface ({tilt:.0f}° tilt)"

        # Block descriptions
        block_descs = []
        for i, block in enumerate(blocks):
            if n <= 5:
                pos = POSITION_LABELS[min(i, 4)] if n <= 5 else f"block {i+1} (from bottom)"
            else:
                if i == 0:
                    pos = "at the bottom"
                elif i == n - 1:
                    pos = "at the top"
                else:
                    pos = f"block {i+1} from the bottom"

            mass_desc = _mass_label(block.mass)
            size_desc = _size_label(block.dimensions[0])
            block_descs.append(
                f"a {mass_desc} ({block.mass:.1f} kg) {size_desc} {block.material} block {pos}"
            )

        if n == 1:
            stack_desc = block_descs[0]
        elif n <= 3:
            stack_desc = ", ".join(block_descs[:-1]) + f", and {block_descs[-1]}"
        else:
            stack_desc = (
                f"{block_descs[0]}; above it {', '.join(block_descs[1:-1])}; "
                f"and {block_descs[-1]}"
            )

        # Force description
        if force < 0.5:
            force_desc = "No external force is applied."
        else:
            dir_label = _direction_label(direction)
            force_word = _force_label(force)
            force_desc = (
                f"A {force_word} lateral force of {force:.1f} N is applied to the "
                f"top block {dir_label}."
            )

        text = (
            f"A stack of {n} block{'s' if n > 1 else ''} sits on {surface_desc}. "
            f"From bottom to top: {stack_desc}. "
            f"{force_desc} "
            f"Predict whether the stack will remain stable or collapse. "
            f"If it collapses, describe what you expect to happen — "
            f"which blocks fall, in what order, and in which direction."
        )
        return text
