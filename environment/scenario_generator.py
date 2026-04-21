"""
FzIQ Scenario Generator
Procedurally generates block stacking scenarios for FzIQ training.
"""

import random
import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class BlockProperties:
    mass: float          # kg, range: 0.1-5.0
    dimensions: Tuple    # (length, width, height) in meters
    friction: float      # friction coefficient: 0.1-0.9
    material: str        # descriptive material name

    def to_dict(self):
        return asdict(self)


@dataclass
class SimulationResult:
    outcome: str                   # "stable" | "partial_collapse" | "full_collapse"
    collapse_sequence: List[int]   # ordered list of block indices that fall (0=bottom)
    final_positions: List[dict]    # positions and orientations after settling
    simulation_time_s: float
    video_path: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class BlockStackScenario:
    """
    Represents a single block stacking scenario for FzIQ training.
    
    Procedurally generated with randomized physical parameters.
    Includes both the physical configuration and its natural language description.
    """
    scenario_id: str
    num_blocks: int
    blocks: List[BlockProperties]
    surface_friction: float
    surface_tilt: float            # degrees
    force_magnitude: float         # Newtons
    force_direction: Tuple         # unit vector (x, y, z)
    force_application_point: Tuple # on top block surface
    scenario_text: str             # natural language description
    ground_truth: Optional[SimulationResult] = None
    created_at: float = field(default_factory=time.time)
    difficulty: float = 0.0        # 0-1 normalized difficulty score

    def hash(self) -> str:
        """SHA-256 hash of scenario parameters for on-chain logging."""
        content = json.dumps({
            "num_blocks": self.num_blocks,
            "blocks": [b.to_dict() for b in self.blocks],
            "surface_friction": self.surface_friction,
            "surface_tilt": self.surface_tilt,
            "force_magnitude": self.force_magnitude,
            "force_direction": list(self.force_direction),
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def difficulty_score(self) -> float:
        """
        Computes normalized difficulty score 0-1 based on:
          - Number of blocks (more blocks = harder)
          - Mass ratio (top-heavy = harder)
          - Force relative to estimated friction resistance
          - Surface tilt
        """
        # Block count contribution (2 blocks = 0, 10 blocks = 1)
        block_score = (self.num_blocks - 2) / 8.0

        # Top-heaviness: ratio of top block mass to bottom block mass
        if len(self.blocks) >= 2:
            top_heavy = self.blocks[-1].mass / max(self.blocks[0].mass, 0.01)
            top_heavy_score = min(top_heavy / 5.0, 1.0)
        else:
            top_heavy_score = 0.0

        # Force score
        force_score = self.force_magnitude / 20.0

        # Tilt score
        tilt_score = self.surface_tilt / 15.0

        difficulty = (
            0.35 * block_score
            + 0.25 * top_heavy_score
            + 0.25 * force_score
            + 0.15 * tilt_score
        )
        return min(max(difficulty, 0.0), 1.0)

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "num_blocks": self.num_blocks,
            "blocks": [b.to_dict() for b in self.blocks],
            "surface_friction": self.surface_friction,
            "surface_tilt": self.surface_tilt,
            "force_magnitude": self.force_magnitude,
            "force_direction": list(self.force_direction),
            "scenario_text": self.scenario_text,
            "difficulty": self.difficulty,
            "hash": self.hash(),
        }


MATERIAL_FRICTION_MAP = {
    "wood": (0.3, 0.6),
    "rubber": (0.6, 0.9),
    "metal": (0.1, 0.3),
    "plastic": (0.2, 0.5),
    "foam": (0.5, 0.8),
    "glass": (0.1, 0.25),
}

SIZE_LABELS = {
    "tiny": (0.05, 0.08),
    "small": (0.08, 0.12),
    "medium": (0.12, 0.20),
    "large": (0.20, 0.28),
    "wide": (0.22, 0.30),
}


class ScenarioGenerator:
    """
    Generates BlockStackScenario instances with randomized physical parameters.
    
    Supports seeded random generation for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self._scenario_counter = 0

    def generate(self, difficulty_level: Optional[int] = None) -> BlockStackScenario:
        """
        Generate a single scenario.
        
        Args:
            difficulty_level: 1 (easy, 2-3 blocks), 2 (medium, 4-6), 3 (hard, 7-10)
                              None = random across all difficulties
        """
        from .scenario_templates import NaturalLanguageGenerator

        # Determine block count based on difficulty
        if difficulty_level == 1:
            num_blocks = self.rng.randint(2, 3)
        elif difficulty_level == 2:
            num_blocks = self.rng.randint(4, 6)
        elif difficulty_level == 3:
            num_blocks = self.rng.randint(7, 10)
        else:
            num_blocks = self.rng.randint(2, 10)

        # Generate blocks from bottom to top
        blocks = []
        for i in range(num_blocks):
            material = self.rng.choice(list(MATERIAL_FRICTION_MAP.keys()))
            friction_range = MATERIAL_FRICTION_MAP[material]
            size_label = self.rng.choice(list(SIZE_LABELS.keys()))
            dim_range = SIZE_LABELS[size_label]

            mass_bias = max(0, (num_blocks - i) / num_blocks)  # heavier at bottom
            mass = round(self.rng.uniform(0.1, 5.0) * (0.5 + mass_bias * 0.5), 2)

            dim = round(self.rng.uniform(*dim_range), 3)
            height = round(self.rng.uniform(0.05, 0.15), 3)

            blocks.append(BlockProperties(
                mass=mass,
                dimensions=(dim, dim, height),
                friction=round(self.rng.uniform(*friction_range), 2),
                material=material,
            ))

        # Surface
        surface_friction = round(self.rng.uniform(0.1, 0.9), 2)
        surface_tilt = round(self.rng.uniform(0.0, 15.0), 1)

        # Force perturbation
        force_magnitude = round(self.rng.uniform(0.0, 20.0), 1)
        angle = self.rng.uniform(0, 2 * 3.14159)
        force_direction = (round(np.cos(angle), 3), round(np.sin(angle), 3), 0.0)
        force_application_point = (
            round(self.rng.uniform(-0.3, 0.3), 3),
            round(self.rng.uniform(-0.3, 0.3), 3),
            0.0
        )

        self._scenario_counter += 1
        scenario_id = f"sc_{int(time.time())}_{self._scenario_counter:04d}"

        scenario = BlockStackScenario(
            scenario_id=scenario_id,
            num_blocks=num_blocks,
            blocks=blocks,
            surface_friction=surface_friction,
            surface_tilt=surface_tilt,
            force_magnitude=force_magnitude,
            force_direction=force_direction,
            force_application_point=force_application_point,
            scenario_text="",  # will be set below
        )
        scenario.difficulty = scenario.difficulty_score()
        scenario.scenario_text = NaturalLanguageGenerator.generate(scenario)
        return scenario

    def generate_batch(self, n: int, difficulty_level: Optional[int] = None) -> List[BlockStackScenario]:
        """Generate N scenarios."""
        return [self.generate(difficulty_level=difficulty_level) for _ in range(n)]

    def generate_benchmark(self) -> List[BlockStackScenario]:
        """
        Generate FzIQ-Bench v1: 500 stratified held-out scenarios.
        Level 1: 100 scenarios (2-3 blocks)
        Level 2: 200 scenarios (4-6 blocks)
        Level 3: 200 scenarios (7-10 blocks)
        """
        scenarios = []
        scenarios.extend(self.generate_batch(100, difficulty_level=1))
        scenarios.extend(self.generate_batch(200, difficulty_level=2))
        scenarios.extend(self.generate_batch(200, difficulty_level=3))
        return scenarios
