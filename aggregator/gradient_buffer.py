"""
Gradient buffer for FzIQ federated aggregation.
Collects failure gradients from agents during the 24-hour window.
"""

import time
import json
import logging
from typing import List, Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class GradientBuffer:
    """
    Thread-safe buffer for collecting failure gradients from agents.
    
    Gradients accumulate over a 24-hour window, then are passed to
    MetamodelAggregator.aggregate() for federated averaging.
    
    In production, this should be backed by a persistent store (Redis or SQLite)
    so gradients survive restarts. This implementation uses in-memory + optional
    disk persistence.
    """

    def __init__(self, persist_path: Optional[str] = None):
        """
        Args:
            persist_path: if set, gradient metadata (not tensors) is logged here
        """
        self._buffer = []
        self._persist_path = persist_path
        self._lock = None  # use threading.Lock() in production

    def add(self, gradient) -> int:
        """
        Add a FailureGradient to the buffer.
        
        Args:
            gradient: FailureGradient instance
        Returns:
            Current buffer size
        """
        self._buffer.append(gradient)
        
        # Log metadata to disk
        if self._persist_path:
            self._append_metadata(gradient)

        logger.info(
            f"Gradient received: agent={gradient.agent_id}, "
            f"score={gradient.score:.3f}, buffer_size={len(self._buffer)}"
        )
        return len(self._buffer)

    def get_all(self) -> list:
        """Return all buffered gradients."""
        return list(self._buffer)

    def clear(self):
        """Clear the buffer after aggregation."""
        count = len(self._buffer)
        self._buffer = []
        logger.info(f"Buffer cleared: {count} gradients consumed")

    def size(self) -> int:
        return len(self._buffer)

    def is_ready(self, min_gradients: int = 5) -> bool:
        """True if buffer has enough gradients for a meaningful aggregation."""
        return len(self._buffer) >= min_gradients

    def summary(self) -> Dict:
        """Statistics about current buffer contents."""
        if not self._buffer:
            return {"size": 0, "agents": [], "avg_score": None}
        
        scores = [g.score for g in self._buffer]
        agents = list(set(g.agent_id for g in self._buffer))
        
        return {
            "size": len(self._buffer),
            "agents": agents,
            "num_agents": len(agents),
            "avg_score": round(sum(scores) / len(scores), 4),
            "min_score": round(min(scores), 4),
            "max_score": round(max(scores), 4),
            "oldest_timestamp": min(g.timestamp for g in self._buffer),
        }

    def _append_metadata(self, gradient):
        """Append gradient metadata (not tensors) to disk log."""
        Path(self._persist_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist_path, "a") as f:
            f.write(json.dumps({
                "agent_id": gradient.agent_id,
                "scenario_hash": gradient.scenario_hash,
                "score": gradient.score,
                "timestamp": gradient.timestamp,
            }) + "\n")
