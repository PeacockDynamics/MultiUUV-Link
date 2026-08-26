"""Private persistent memories and communication-aware planning views."""

import numpy as np

from .config import FOV_HEIGHT, FOV_WIDTH
from .models import UUVState
from .perception import get_fov_bounds


class ResourceMemory:
    def __init__(self, uuv_id: int):
        self.uuv_id = uuv_id
        self._known_resources = {}

    def update(self, visible_resources):
        newly_discovered = []
        for resource in visible_resources:
            if resource.resource_id not in self._known_resources:
                self._known_resources[resource.resource_id] = resource
                newly_discovered.append(resource)
        return newly_discovered

    def knows(self, resource_id: int) -> bool:
        return resource_id in self._known_resources

    def get(self, resource_id: int):
        return self._known_resources.get(resource_id)

    def all_resources(self):
        return list(self._known_resources.values())

    def __len__(self):
        return len(self._known_resources)


class ExplorationMemory:
    def __init__(self, uuv_id: int, width: int, height: int):
        self.uuv_id = uuv_id
        self.width = width
        self.height = height
        self.observed = np.zeros((height, width), dtype=bool)

    def update_from_fov(
        self,
        state: UUVState,
        fov_width: int = FOV_WIDTH,
        fov_height: int = FOV_HEIGHT,
    ):
        x_min, y_min, x_max, y_max = get_fov_bounds(
            state, fov_width=fov_width, fov_height=fov_height
        )
        x0, y0 = max(0, x_min), max(0, y_min)
        x1, y1 = min(self.width, x_max), min(self.height, y_max)
        before_count = int(self.observed.sum())
        self.observed[y0:y1, x0:x1] = True
        return int(self.observed.sum()) - before_count

    def observed_count(self):
        return int(self.observed.sum())

    def coverage(self):
        return self.observed_count() / (self.width * self.height)


class PlanningMemory:
    def __init__(self, uuv_id: int, observed_mask: np.ndarray):
        self.uuv_id = uuv_id
        self.observed = observed_mask
        self.height, self.width = observed_mask.shape

    def observed_count(self):
        return int(self.observed.sum())

    def coverage(self):
        return self.observed_count() / (self.width * self.height)


def initialize_known_maps(agents):
    return {
        agent.uuv_id: agent.exploration_memory.observed.copy() for agent in agents
    }


def merge_local_into_known(agents, known_maps):
    for agent in agents:
        known_maps[agent.uuv_id] = np.logical_or(
            known_maps[agent.uuv_id], agent.exploration_memory.observed
        )
