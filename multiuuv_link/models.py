"""Validated UUV and resource state structures."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import MAX_SPAWN_ATTEMPTS


@dataclass
class UUVState:
    uuv_id: int
    x: int
    y: int


@dataclass(frozen=True)
class Resource:
    resource_id: int
    x: int
    y: int
    resource_class: str


@dataclass
class UUVAgent:
    uuv_id: int
    state: UUVState
    exploration_memory: Any
    resource_memory: Any
    trajectory: list = field(default_factory=list)

    def observe(self, resources):
        """Execute one legitimate private observation cycle."""
        from .perception import discover_visible_resources

        newly_observed_cells = self.exploration_memory.update_from_fov(self.state)
        visible_resources = discover_visible_resources(self.state, resources)
        newly_discovered_resources = self.resource_memory.update(visible_resources)
        return newly_observed_cells, visible_resources, newly_discovered_resources


def generate_safe_spawn_positions(
    num_uuvs: int,
    width: int,
    height: int,
    min_separation: float,
    rng=None,
    max_attempts: int = MAX_SPAWN_ATTEMPTS,
):
    from .motion import euclidean_distance

    if rng is None:
        rng = np.random.default_rng()
    positions = []
    attempts = 0
    while len(positions) < num_uuvs:
        if attempts >= max_attempts:
            raise RuntimeError(
                "Could not generate valid UUV spawn positions within the maximum "
                "attempt limit. Reduce minimum separation or UUV count."
            )
        attempts += 1
        candidate = (
            int(rng.integers(0, width)),
            int(rng.integers(0, height)),
        )
        if all(
            euclidean_distance(candidate, existing) >= min_separation
            for existing in positions
        ):
            positions.append(candidate)
    return positions, attempts
