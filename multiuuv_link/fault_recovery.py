"""Validated failure detection, orphan-task creation, and successor selection."""

from copy import deepcopy

from .communication import packet_delivered, sample_message_delay
from .config import (
    GOAL_REACHED_RADIUS,
    MAP_DIAGONAL,
    MAX_COMM_DELAY,
    MIN_COMM_DELAY,
    PACKET_LOSS_PROBABILITY,
    REASSIGN_DISTANCE_WEIGHT,
    REASSIGN_INFO_WEIGHT,
    ORPHAN_FAILED_UUV_CLEARANCE_36,
    SAFE_SEPARATION,
    TASK_DEDUPLICATION_DISTANCE,
)
from .frontier import (
    candidate_fov_statistics,
    extract_frontier_mask,
    generate_goal_from_knowledge,
    sample_frontier_candidates,
    score_frontier_candidate_with_memory,
)
from .memory import PlanningMemory
from .motion import euclidean_distance


def queue_heartbeat(
    queue,
    sender,
    receiver,
    current_time,
    rng,
    next_message_id,
    loss_probability=PACKET_LOSS_PROBABILITY,
    min_delay=MIN_COMM_DELAY,
    max_delay=MAX_COMM_DELAY,
):
    if not packet_delivered(rng, loss_probability):
        return next_message_id, False
    delay = sample_message_delay(rng, min_delay, max_delay)
    queue.append(
        {
            "message_id": next_message_id,
            "sender": sender,
            "receiver": receiver,
            "send_time": current_time,
            "arrival_time": current_time + delay,
        }
    )
    return next_message_id + 1, True


def deliver_heartbeats(queue, current_time, failed_flags):
    due = [message for message in queue if message["arrival_time"] <= current_time]
    future = [message for message in queue if message["arrival_time"] > current_time]
    fresh_pairs = set()
    for message in due:
        if failed_flags[message["receiver"]]:
            continue
        fresh_pairs.add((message["receiver"], message["sender"]))
    return future, fresh_pairs


def update_failure_suspicion(
    healthy_ids,
    fresh_pairs,
    suspicion_count,
    declared_failed,
    detection_time,
    current_time,
    threshold,
):
    for detector in healthy_ids:
        for peer in sorted(suspicion_count[detector]):
            if peer == detector:
                continue
            if (detector, peer) in fresh_pairs:
                suspicion_count[detector][peer] = 0
            else:
                suspicion_count[detector][peer] += 1
            if suspicion_count[detector][peer] >= threshold:
                declared_failed[detector].add(peer)
                detection_time[detector].setdefault(peer, current_time)


def majority_detectors(healthy_ids, declared_failed, failed_uuv_id):
    return [
        detector
        for detector in healthy_ids
        if failed_uuv_id in declared_failed[detector]
    ]


def get_or_create_orphan_goal(
    failed_agent, existing_goal, known_maps
):
    from .allocation import goal_is_still_useful

    if existing_goal is not None and goal_is_still_useful(
        failed_agent, existing_goal, known_maps
    ):
        return deepcopy(existing_goal), "existing_assignment"
    known_goal = generate_goal_from_knowledge(
        failed_agent, known_maps[failed_agent.uuv_id]
    )["selected"]
    if known_goal is not None:
        return deepcopy(known_goal), "generated_from_known_frontier"
    local_goal = generate_goal_from_knowledge(
        failed_agent, failed_agent.exploration_memory.observed
    )["selected"]
    if local_goal is not None:
        return deepcopy(local_goal), "generated_from_local_frontier"
    raise RuntimeError("Could not create a legitimate frontier orphan task.")


def score_orphan_successor(
    agent,
    orphan_goal,
    known_maps,
    information_weight=REASSIGN_INFO_WEIGHT,
    distance_weight=REASSIGN_DISTANCE_WEIGHT,
):
    memory = PlanningMemory(agent.uuv_id, known_maps[agent.uuv_id])
    stats = candidate_fov_statistics(
        memory, int(orphan_goal["x"]), int(orphan_goal["y"])
    )
    distance_px = euclidean_distance(
        (agent.state.x, agent.state.y), (orphan_goal["x"], orphan_goal["y"])
    )
    distance_norm = distance_px / MAP_DIAGONAL
    utility = information_weight * stats["information_ratio"] - distance_weight * distance_norm
    return {
        "uuv_id": agent.uuv_id,
        "information_ratio": float(stats["information_ratio"]),
        "distance_px": float(distance_px),
        "distance_norm": float(distance_norm),
        "utility": float(utility),
    }


def choose_orphan_successor(
    agents,
    orphan_goal,
    failed_flags,
    known_maps,
    information_weight=REASSIGN_INFO_WEIGHT,
    distance_weight=REASSIGN_DISTANCE_WEIGHT,
):
    candidates = [
        score_orphan_successor(
            agent, orphan_goal, known_maps, information_weight, distance_weight
        )
        for agent in agents
        if not failed_flags[agent.uuv_id]
    ]
    if not candidates:
        raise RuntimeError("No healthy successor exists.")
    candidates.sort(
        key=lambda record: (record["utility"], -record["distance_px"]),
        reverse=True,
    )
    return candidates[0]["uuv_id"], candidates


def create_safe_orphan_task(
    failed_agent,
    active_goal_snapshot,
    known_snapshot,
    failure_position,
    minimum_clearance=SAFE_SEPARATION + GOAL_REACHED_RADIUS,
):
    if active_goal_snapshot is not None:
        distance = euclidean_distance(
            (active_goal_snapshot["x"], active_goal_snapshot["y"]),
            failure_position,
        )
        if distance >= minimum_clearance:
            return deepcopy(active_goal_snapshot), "active_goal_at_failure"
    frozen_memory = PlanningMemory(failed_agent.uuv_id, known_snapshot)
    frontier_mask = extract_frontier_mask(frozen_memory)
    candidates = sample_frontier_candidates(
        frontier_mask,
        reference_position=failure_position,
        min_spacing=TASK_DEDUPLICATION_DISTANCE,
        max_candidates=24,
    )
    feasible = []
    for candidate_index, candidate in enumerate(candidates):
        normalized = {
            "candidate_id": candidate.get("candidate_id", candidate_index),
            "x": int(candidate["x"]),
            "y": int(candidate["y"]),
        }
        if euclidean_distance(
            (normalized["x"], normalized["y"]), failure_position
        ) >= minimum_clearance:
            feasible.append(normalized)
    if not feasible:
        raise RuntimeError("No safety-feasible orphan frontier exists in failure snapshot.")
    scored = [
        score_frontier_candidate_with_memory(failed_agent, candidate, frozen_memory)
        for candidate in feasible
    ]
    return deepcopy(max(scored, key=lambda record: record["utility"])), "generated_safe_frontier"


def orphan_goal_safe_from_failed_uuv_36(
    goal,
    failure_position,
    minimum_clearance=ORPHAN_FAILED_UUV_CLEARANCE_36,
):
    """Combo 36 clearance predicate retained as a separately testable invariant."""
    return euclidean_distance(
        (int(goal["x"]), int(goal["y"])), failure_position
    ) >= minimum_clearance
