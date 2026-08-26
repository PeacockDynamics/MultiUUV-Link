from pathlib import Path

import numpy as np
import pytest

from multiuuv_link.config import ARTIFACTS_DIR, MODEL_NAME
from multiuuv_link.learning import load_frontier_value_model
from multiuuv_link.memory import ExplorationMemory, ResourceMemory, initialize_known_maps
from multiuuv_link.models import Resource, UUVAgent, UUVState


@pytest.fixture
def resources():
    return [
        Resource(0, 128, 128, "A"),
        Resource(1, 288, 128, "B"),
        Resource(2, 128, 288, "C"),
        Resource(3, 288, 288, "A"),
    ]


@pytest.fixture
def small_swarm(resources):
    positions = [(128, 128), (288, 128), (128, 288), (288, 288)]
    agents = []
    for uuv_id, (x, y) in enumerate(positions):
        agent = UUVAgent(
            uuv_id,
            UUVState(uuv_id, x, y),
            ExplorationMemory(uuv_id, 512, 512),
            ResourceMemory(uuv_id),
            [(x, y)],
        )
        agent.observe(resources)
        agents.append(agent)
    return agents, initialize_known_maps(agents)


@pytest.fixture(scope="session")
def validated_model():
    model_path = Path(ARTIFACTS_DIR) / MODEL_NAME
    return load_frontier_value_model(model_path)
