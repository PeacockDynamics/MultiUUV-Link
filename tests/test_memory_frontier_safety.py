import numpy as np
import pytest

from multiuuv_link.config import (
    ACTION_HOLD,
    ACTION_LEFT,
    ACTION_RIGHT,
    SAFE_SEPARATION,
)
from multiuuv_link.frontier import (
    candidate_fov_statistics,
    extract_frontier_candidates,
    extract_frontier_mask,
    generate_goal_from_knowledge,
    generate_private_frontier_goal,
    sample_frontier_candidates,
    score_frontier_candidate_with_memory,
)
from multiuuv_link.memory import ExplorationMemory, PlanningMemory, ResourceMemory
from multiuuv_link.models import Resource, UUVAgent, UUVState
from multiuuv_link.safety import (
    evaluate_pairwise_safety,
    failure_aware_safety_shield_33,
    safety_shield,
)


def test_private_memory_is_persistent_monotonic_and_overlap_aware():
    memory = ExplorationMemory(0, 512, 512)
    first = memory.update_from_fov(UUVState(0, 200, 200))
    snapshot = memory.observed.copy()
    second = memory.update_from_fov(UUVState(0, 208, 200))
    assert first == 128 * 128
    assert second == 8 * 128
    assert np.all(memory.observed[snapshot])
    assert memory.observed_count() == first + second


def test_resource_memory_deduplicates_by_id_without_forgetting():
    memory = ResourceMemory(0)
    resource = Resource(7, 10, 20, "C")
    assert memory.update([resource]) == [resource]
    assert memory.update([resource]) == []
    assert memory.knows(7)
    assert memory.get(7) == resource
    assert len(memory) == 1


def test_frontier_mask_is_unknown_ring_around_observed_cells():
    memory = ExplorationMemory(0, 20, 20)
    memory.observed[8:12, 8:12] = True
    frontier = extract_frontier_mask(memory)
    assert frontier.dtype == bool
    assert not np.any(frontier & memory.observed)
    assert frontier.sum() == 20
    candidates, regions, _ = extract_frontier_candidates(frontier, min_region_size=1)
    assert regions == 1
    assert candidates[0]["region_size"] == 20


def test_farthest_point_frontier_sampling_is_deterministic_and_spaced():
    mask = np.zeros((256, 256), dtype=bool)
    mask[10, 10] = mask[10, 240] = mask[240, 10] = mask[240, 240] = True
    first = sample_frontier_candidates(mask, (128, 128), min_spacing=100, max_candidates=4)
    second = sample_frontier_candidates(mask, (128, 128), min_spacing=100, max_candidates=4)
    assert first == second
    assert len(first) == 4
    for index, a in enumerate(first):
        for b in first[index + 1 :]:
            assert np.hypot(a["x"] - b["x"], a["y"] - b["y"]) >= 100


def test_classical_frontier_scoring_uses_only_supplied_planning_memory():
    local = ExplorationMemory(0, 512, 512)
    local.update_from_fov(UUVState(0, 256, 256))
    agent = UUVAgent(0, UUVState(0, 256, 256), local, ResourceMemory(0), [(256, 256)])
    private_snapshot = local.observed.copy()
    supplied = PlanningMemory(0, np.zeros((512, 512), dtype=bool))
    candidate = {"candidate_id": 0, "x": 64, "y": 64}
    stats = candidate_fov_statistics(supplied, 64, 64)
    score = score_frontier_candidate_with_memory(agent, candidate, supplied)
    assert stats["information_ratio"] == 1.0
    assert score["info_ratio"] == 1.0
    assert np.array_equal(local.observed, private_snapshot)


def test_goal_generation_does_not_mutate_agent_or_knowledge(small_swarm):
    agents, known_maps = small_swarm
    agent = agents[0]
    local_before = agent.exploration_memory.observed.copy()
    known_before = known_maps[0].copy()
    result = generate_goal_from_knowledge(agent, known_maps[0])
    assert result["selected"] is not None
    assert np.array_equal(agent.exploration_memory.observed, local_before)
    assert np.array_equal(known_maps[0], known_before)
    private_goal = generate_private_frontier_goal(agent)
    assert private_goal is not None
    assert np.array_equal(agent.exploration_memory.observed, local_before)


def test_safety_shield_exhaustively_overrides_head_on_collision():
    states = [UUVState(0, 200, 200), UUVState(1, 360, 200)]
    result = safety_shield(states, [ACTION_RIGHT, ACTION_LEFT])
    assert result["desired_minimum_distance"] == 96.0
    assert result["minimum_distance"] >= SAFE_SEPARATION
    assert result["override_count"] == 1


def test_pairwise_threshold_is_strictly_less_than_128():
    exact = evaluate_pairwise_safety({0: (0, 0), 1: (128, 0)})[0]
    below = evaluate_pairwise_safety({0: (0, 0), 1: (127, 0)})[0]
    assert not exact["unsafe"]
    assert below["unsafe"]


def test_failure_aware_shield_forces_failed_uuv_hold():
    states = [UUVState(0, 200, 200), UUVState(1, 360, 200)]
    result = failure_aware_safety_shield_33(
        states, [ACTION_RIGHT, ACTION_LEFT], [False, True]
    )
    assert result["executed_actions"][1] == ACTION_HOLD
    assert result["minimum_distance"] >= SAFE_SEPARATION


def test_no_safe_joint_action_raises_when_initial_geometry_is_irrecoverable():
    with pytest.raises(RuntimeError):
        safety_shield([UUVState(0, 100, 100), UUVState(1, 100, 100)], [ACTION_HOLD, ACTION_HOLD])
