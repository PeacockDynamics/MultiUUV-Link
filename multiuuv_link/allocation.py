"""Stable component-level frontier task allocation from notebook Combos 26-27."""

from itertools import combinations

import networkx as nx

from .config import (
    FOV_HEIGHT,
    FOV_WIDTH,
    GOAL_MIN_UNKNOWN_RATIO,
    GOAL_REACHED_RADIUS,
    MIN_ASSIGNED_GOAL_SEPARATION,
    TASK_DEDUPLICATION_DISTANCE,
    TASK_OVERLAP_WEIGHT,
    WIDTH,
    HEIGHT,
)
from .frontier import (
    candidate_fov_statistics,
    generate_goal_from_knowledge,
    score_frontier_candidate_with_memory,
)
from .memory import PlanningMemory
from .models import UUVState
from .motion import euclidean_distance
from .perception import get_fov_bounds


def deduplicate_frontier_tasks(
    raw_tasks, min_distance=TASK_DEDUPLICATION_DISTANCE
):
    if not raw_tasks:
        return []
    retained = []
    for task in raw_tasks:
        candidate_position = (task["x"], task["y"])
        if all(
            euclidean_distance(
                candidate_position, (existing["x"], existing["y"])
            )
            >= min_distance
            for existing in retained
        ):
            retained.append(
                {
                    "task_id": len(retained),
                    "x": int(task["x"]),
                    "y": int(task["y"]),
                    "source_uuv": int(task["source_uuv"]),
                }
            )
    return retained


def candidate_valid_fov_rectangle(x: int, y: int, width=WIDTH, height=HEIGHT):
    state = UUVState(-1, int(x), int(y))
    x_min, y_min, x_max, y_max = get_fov_bounds(
        state, FOV_WIDTH, FOV_HEIGHT
    )
    return (
        int(max(0, x_min)),
        int(max(0, y_min)),
        int(min(width, x_max)),
        int(min(height, y_max)),
    )


def candidate_fov_iou(goal_a, goal_b, width=WIDTH, height=HEIGHT):
    ax0, ay0, ax1, ay1 = candidate_valid_fov_rectangle(
        goal_a["x"], goal_a["y"], width, height
    )
    bx0, by0, bx1, by1 = candidate_valid_fov_rectangle(
        goal_b["x"], goal_b["y"], width, height
    )
    intersection_width = max(0, min(ax1, bx1) - max(ax0, bx0))
    intersection_height = max(0, min(ay1, by1) - max(ay0, by0))
    intersection = intersection_width * intersection_height
    area_a, area_b = (ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else float(intersection / union)


def goal_information_ratio(agent, goal, known_maps):
    if goal is None:
        return 0.0
    memory = PlanningMemory(agent.uuv_id, known_maps[agent.uuv_id])
    return float(
        candidate_fov_statistics(memory, int(goal["x"]), int(goal["y"]))[
            "information_ratio"
        ]
    )


def goal_is_still_useful(agent, goal, known_maps):
    if goal is None:
        return False
    distance = euclidean_distance(
        (agent.state.x, agent.state.y), (goal["x"], goal["y"])
    )
    if distance <= GOAL_REACHED_RADIUS:
        return False
    return goal_information_ratio(agent, goal, known_maps) >= GOAL_MIN_UNKNOWN_RATIO


def build_component_task_pool(component, agent_by_id, known_maps):
    raw_tasks = []
    for uuv_id in component:
        result = generate_goal_from_knowledge(agent_by_id[uuv_id], known_maps[uuv_id])
        for candidate in result["candidates"]:
            raw_tasks.append(
                {
                    "x": int(candidate["x"]),
                    "y": int(candidate["y"]),
                    "source_uuv": int(uuv_id),
                }
            )
    return raw_tasks, deduplicate_frontier_tasks(raw_tasks)


def allocate_component_frontiers_stable(
    component, previous_goals, agent_by_id, known_maps
):
    component = sorted(component)
    assignments = {uuv_id: None for uuv_id in component}
    raw_tasks, tasks = build_component_task_pool(component, agent_by_id, known_maps)
    if len(component) == 1:
        uuv_id = component[0]
        agent = agent_by_id[uuv_id]
        old_goal = previous_goals.get(uuv_id)
        if goal_is_still_useful(agent, old_goal, known_maps):
            assignments[uuv_id] = old_goal
            return assignments, {
                "raw_tasks": len(raw_tasks),
                "dedup_tasks": len(tasks),
                "locked_goals": 1,
                "auction_assignments": 0,
                "unassigned": 0,
            }
        assignments[uuv_id] = generate_goal_from_knowledge(
            agent, known_maps[uuv_id]
        )["selected"]
        return assignments, {
            "raw_tasks": len(raw_tasks),
            "dedup_tasks": len(tasks),
            "locked_goals": 0,
            "auction_assignments": int(assignments[uuv_id] is not None),
            "unassigned": int(assignments[uuv_id] is None),
        }

    locked_goals = []
    lock_candidates = []
    for uuv_id in component:
        old_goal = previous_goals.get(uuv_id)
        agent = agent_by_id[uuv_id]
        if goal_is_still_useful(agent, old_goal, known_maps):
            lock_candidates.append(
                (goal_information_ratio(agent, old_goal, known_maps), uuv_id, old_goal)
            )
    lock_candidates.sort(key=lambda item: item[0], reverse=True)
    for _, uuv_id, old_goal in lock_candidates:
        candidate_position = (old_goal["x"], old_goal["y"])
        if all(
            euclidean_distance(
                candidate_position, (goal["x"], goal["y"])
            )
            >= MIN_ASSIGNED_GOAL_SEPARATION
            for goal in locked_goals
        ):
            assignments[uuv_id] = old_goal
            locked_goals.append(old_goal)

    available_tasks = []
    for task in tasks:
        if all(
            euclidean_distance((task["x"], task["y"]), (goal["x"], goal["y"]))
            >= MIN_ASSIGNED_GOAL_SEPARATION
            for goal in locked_goals
        ):
            available_tasks.append(task)

    unassigned_agents = {
        uuv_id for uuv_id in component if assignments[uuv_id] is None
    }
    assigned_task_ids = set()
    width = next(iter(known_maps.values())).shape[1]
    height = next(iter(known_maps.values())).shape[0]
    while unassigned_agents and len(assigned_task_ids) < len(available_tasks):
        best_bid = None
        current_goals = [goal for goal in assignments.values() if goal is not None]
        for uuv_id in sorted(unassigned_agents):
            agent = agent_by_id[uuv_id]
            memory = PlanningMemory(uuv_id, known_maps[uuv_id])
            for task in available_tasks:
                task_id = task["task_id"]
                if task_id in assigned_task_ids:
                    continue
                position = (task["x"], task["y"])
                if any(
                    euclidean_distance(position, (goal["x"], goal["y"]))
                    < MIN_ASSIGNED_GOAL_SEPARATION
                    for goal in current_goals
                ):
                    continue
                candidate = {
                    "candidate_id": task_id,
                    "x": task["x"],
                    "y": task["y"],
                }
                base = score_frontier_candidate_with_memory(agent, candidate, memory)
                overlap = (
                    max(
                        candidate_fov_iou(task, goal, width, height)
                        for goal in current_goals
                    )
                    if current_goals
                    else 0.0
                )
                bid = {
                    "uuv_id": uuv_id,
                    "task_id": task_id,
                    "x": int(task["x"]),
                    "y": int(task["y"]),
                    "base_utility": float(base["utility"]),
                    "overlap": float(overlap),
                    "utility": float(base["utility"] - TASK_OVERLAP_WEIGHT * overlap),
                    "distance_px": float(base["distance_px"]),
                    "info_ratio": float(base["info_ratio"]),
                    "redundancy_ratio": float(base["redundancy_ratio"]),
                }
                if best_bid is None or bid["utility"] > best_bid["utility"]:
                    best_bid = bid
        if best_bid is None:
            break
        uuv_id = best_bid["uuv_id"]
        assignments[uuv_id] = {
            "candidate_id": best_bid["task_id"],
            "x": best_bid["x"],
            "y": best_bid["y"],
            "base_utility": best_bid["base_utility"],
            "overlap": best_bid["overlap"],
            "utility": best_bid["utility"],
            "distance_px": best_bid["distance_px"],
            "info_ratio": best_bid["info_ratio"],
            "redundancy_ratio": best_bid["redundancy_ratio"],
        }
        assigned_task_ids.add(best_bid["task_id"])
        unassigned_agents.remove(uuv_id)
    return assignments, {
        "raw_tasks": len(raw_tasks),
        "dedup_tasks": len(tasks),
        "locked_goals": len(locked_goals),
        "auction_assignments": sum(g is not None for g in assignments.values())
        - len(locked_goals),
        "unassigned": sum(g is None for g in assignments.values()),
    }


def perform_stable_allocation_round(
    graph, previous_goals, agents, known_maps
):
    agent_by_id = {agent.uuv_id: agent for agent in agents}
    assignments, round_stats, spacing_samples, overlap_samples = {}, [], [], []
    width = next(iter(known_maps.values())).shape[1]
    height = next(iter(known_maps.values())).shape[0]
    for component_index, component in enumerate(
        sorted((sorted(c) for c in nx.connected_components(graph)), key=lambda c: c[0])
    ):
        component_assignments, stats = allocate_component_frontiers_stable(
            component, previous_goals, agent_by_id, known_maps
        )
        assignments.update(component_assignments)
        stats.update({"component": component_index, "agents": str(component)})
        round_stats.append(stats)
        assigned = [
            (uuv_id, component_assignments[uuv_id])
            for uuv_id in component
            if component_assignments[uuv_id] is not None
        ]
        for (_, goal_a), (_, goal_b) in combinations(assigned, 2):
            distance = euclidean_distance(
                (goal_a["x"], goal_a["y"]), (goal_b["x"], goal_b["y"])
            )
            if distance < MIN_ASSIGNED_GOAL_SEPARATION:
                raise RuntimeError(
                    "Final component assignments violate minimum goal separation."
                )
            spacing_samples.append(distance)
            overlap_samples.append(candidate_fov_iou(goal_a, goal_b, width, height))
    changed_count = 0
    for agent in agents:
        previous = previous_goals.get(agent.uuv_id)
        current = assignments.get(agent.uuv_id)
        if previous is None and current is None:
            continue
        if previous is None or current is None:
            changed_count += 1
        elif previous["x"] != current["x"] or previous["y"] != current["y"]:
            changed_count += 1
    return assignments, changed_count, spacing_samples, overlap_samples, round_stats
