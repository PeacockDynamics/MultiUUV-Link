"""Classical, learned, and confidence-gated hybrid frontier controllers."""

from copy import deepcopy

import networkx as nx
import numpy as np
import torch

from .config import (
    HYBRID_CLASSICAL_REGRET_MAX_43,
    HYBRID_LEARNED_MARGIN_MIN_43,
    MAP_DIAGONAL,
    MIN_ASSIGNED_GOAL_SEPARATION,
    TASK_DEDUPLICATION_DISTANCE,
)
from .frontier import (
    extract_frontier_mask,
    sample_frontier_candidates,
    score_frontier_candidate_with_memory,
)
from .memory import PlanningMemory
from .motion import euclidean_distance


def score_frontier_candidate_learned_39(
    agent,
    candidate,
    planning_memory,
    communication_degree,
    model,
    feature_mean,
    feature_std,
    num_agents,
    device="cpu",
):
    classical_score = score_frontier_candidate_with_memory(agent, candidate, planning_memory)
    feature_vector = np.asarray(
        [
            classical_score["info_ratio"],
            classical_score["distance_px"] / MAP_DIAGONAL,
            classical_score["redundancy_ratio"],
            agent.exploration_memory.coverage(),
            planning_memory.coverage(),
            communication_degree / max(1, num_agents - 1),
        ],
        dtype=np.float32,
    )
    normalized = (feature_vector - np.asarray(feature_mean).reshape(-1)) / np.asarray(feature_std).reshape(-1)
    tensor = torch.tensor(normalized, dtype=torch.float32, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        predicted_utility = float(model(tensor).cpu().item())
    return {
        "candidate_id": candidate.get("candidate_id"),
        "x": int(candidate["x"]),
        "y": int(candidate["y"]),
        "learned_utility": predicted_utility,
        "classical_utility": float(classical_score["utility"]),
        "info_ratio": float(classical_score["info_ratio"]),
        "distance_px": float(classical_score["distance_px"]),
        "redundancy_ratio": float(classical_score["redundancy_ratio"]),
        "feature_vector": feature_vector,
    }


def generate_goal_from_knowledge_learned_39(
    agent,
    knowledge_mask,
    communication_degree,
    model,
    feature_mean,
    feature_std,
    num_agents,
    device="cpu",
):
    memory = PlanningMemory(agent.uuv_id, knowledge_mask)
    frontier = extract_frontier_mask(memory)
    candidates = sample_frontier_candidates(
        frontier,
        (agent.state.x, agent.state.y),
        min_spacing=TASK_DEDUPLICATION_DISTANCE,
        max_candidates=12,
    )
    if not candidates:
        return {"selected": None, "candidates": [], "frontier_mask": frontier}
    scores = [
        score_frontier_candidate_learned_39(
            agent, candidate, memory, communication_degree, model,
            feature_mean, feature_std, num_agents, device
        )
        for candidate in candidates
    ]
    return {
        "selected": max(scores, key=lambda record: record["learned_utility"]),
        "candidates": scores,
        "frontier_mask": frontier,
    }


def choose_hybrid_frontier_43(classical_scores, learned_scores):
    if len(classical_scores) != len(learned_scores):
        raise RuntimeError("Classical and learned candidate lists have different lengths.")
    if not classical_scores:
        return {
            "selected": None,
            "decision_source": "NO_TASK",
            "learned_margin": np.nan,
            "classical_regret": 0.0,
            "classical_goal": None,
            "learned_goal": None,
        }
    classical_best_index = int(np.argmax([record["utility"] for record in classical_scores]))
    classical_best = classical_scores[classical_best_index]
    learned_utilities = np.asarray([record["learned_utility"] for record in learned_scores], dtype=float)
    learned_order = np.argsort(learned_utilities)[::-1]
    learned_best_index = int(learned_order[0])
    learned_best = learned_scores[learned_best_index]
    learned_margin = (
        float(learned_utilities[learned_order[0]] - learned_utilities[learned_order[1]])
        if len(learned_order) >= 2
        else np.inf
    )
    classical_goal = (int(classical_best["x"]), int(classical_best["y"]))
    learned_goal = (int(learned_best["x"]), int(learned_best["y"]))
    if classical_goal == learned_goal:
        selected_index, source = learned_best_index, "LEARNED_AGREEMENT"
    else:
        regret = float(classical_best["utility"] - classical_scores[learned_best_index]["utility"])
        if learned_margin >= HYBRID_LEARNED_MARGIN_MIN_43 and regret <= HYBRID_CLASSICAL_REGRET_MAX_43:
            selected_index, source = learned_best_index, "LEARNED_OVERRIDE"
        else:
            selected_index, source = classical_best_index, "CLASSICAL_FALLBACK"
    selected_classical = classical_scores[selected_index]
    selected_learned = learned_scores[selected_index]
    classical_regret = float(classical_best["utility"] - selected_classical["utility"])
    return {
        "selected": {
            "candidate_id": selected_classical["candidate_id"],
            "x": int(selected_classical["x"]),
            "y": int(selected_classical["y"]),
            "classical_utility": float(selected_classical["utility"]),
            "learned_utility": float(selected_learned["learned_utility"]),
        },
        "decision_source": source,
        "learned_margin": float(learned_margin),
        "classical_regret": classical_regret,
        "classical_goal": classical_goal,
        "learned_goal": learned_goal,
    }


def generate_goal_from_knowledge_hybrid_43(
    agent,
    knowledge_mask,
    communication_degree,
    model,
    feature_mean,
    feature_std,
    num_agents,
    device="cpu",
):
    memory = PlanningMemory(agent.uuv_id, knowledge_mask)
    frontier = extract_frontier_mask(memory)
    candidates = sample_frontier_candidates(
        frontier, (agent.state.x, agent.state.y),
        min_spacing=TASK_DEDUPLICATION_DISTANCE, max_candidates=12
    )
    if not candidates:
        return {"selected": None, "decision_source": "NO_TASK", "candidates": []}
    normalized = [
        {"candidate_id": candidate.get("candidate_id", index), "x": int(candidate["x"]), "y": int(candidate["y"])}
        for index, candidate in enumerate(candidates)
    ]
    classical = [score_frontier_candidate_with_memory(agent, candidate, memory) for candidate in normalized]
    learned = [
        score_frontier_candidate_learned_39(
            agent, candidate, memory, communication_degree, model,
            feature_mean, feature_std, num_agents, device
        )
        for candidate in normalized
    ]
    result = choose_hybrid_frontier_43(classical, learned)
    result["candidates"] = normalized
    return result


def _shared_task_pool(component_index, component, agent_by_id, known_maps):
    task_pool = []
    for uuv_id in component:
        agent = agent_by_id[uuv_id]
        memory = PlanningMemory(uuv_id, known_maps[uuv_id])
        candidates = sample_frontier_candidates(
            extract_frontier_mask(memory), (agent.state.x, agent.state.y),
            min_spacing=TASK_DEDUPLICATION_DISTANCE, max_candidates=12
        )
        for candidate in candidates:
            xy = (candidate["x"], candidate["y"])
            if any(euclidean_distance(xy, (task["x"], task["y"])) < TASK_DEDUPLICATION_DISTANCE for task in task_pool):
                continue
            task_pool.append({
                "candidate_id": component_index * 1000 + len(task_pool),
                "x": int(candidate["x"]), "y": int(candidate["y"]),
            })
    return task_pool


def perform_learned_allocation_round_40(
    operational_graph, previous_goals, agents, known_maps,
    model, feature_mean, feature_std, device="cpu"
):
    assignments = {agent.uuv_id: None for agent in agents}
    records = []
    graph = operational_graph.copy()
    graph.add_nodes_from(assignments)
    agent_by_id = {agent.uuv_id: agent for agent in agents}
    for component_index, component_set in enumerate(nx.connected_components(graph)):
        component = sorted(component_set)
        tasks = _shared_task_pool(component_index, component, agent_by_id, known_maps)
        pairs = []
        for uuv_id in component:
            agent = agent_by_id[uuv_id]
            memory = PlanningMemory(uuv_id, known_maps[uuv_id])
            for task_index, task in enumerate(tasks):
                score = score_frontier_candidate_learned_39(
                    agent, task, memory, graph.degree[uuv_id], model,
                    feature_mean, feature_std, len(agents), device
                )
                pairs.append({"uuv_id": uuv_id, "task_index": task_index, **task, "utility": float(score["learned_utility"])})
        pairs.sort(key=lambda record: record["utility"], reverse=True)
        assigned_uuvs, assigned_tasks, positions = set(), set(), []
        for record in pairs:
            if record["uuv_id"] in assigned_uuvs or record["task_index"] in assigned_tasks:
                continue
            position = (record["x"], record["y"])
            if any(euclidean_distance(position, assigned) < MIN_ASSIGNED_GOAL_SEPARATION for assigned in positions):
                continue
            assignments[record["uuv_id"]] = {
                "candidate_id": int(record["candidate_id"]), "x": int(record["x"]),
                "y": int(record["y"]), "learned_utility": float(record["utility"]),
            }
            assigned_uuvs.add(record["uuv_id"]); assigned_tasks.add(record["task_index"]); positions.append(position)
            records.append(deepcopy(record))
    changes = sum(
        (None if previous_goals.get(uuv_id) is None else (previous_goals[uuv_id]["x"], previous_goals[uuv_id]["y"]))
        != (None if goal is None else (goal["x"], goal["y"]))
        for uuv_id, goal in assignments.items()
    )
    return assignments, changes, records


def perform_hybrid_allocation_round_44(
    operational_graph, previous_goals, agents, known_maps,
    model, feature_mean, feature_std, device="cpu"
):
    assignments = {agent.uuv_id: None for agent in agents}
    decision_counts = {"LEARNED_AGREEMENT": 0, "LEARNED_OVERRIDE": 0, "CLASSICAL_FALLBACK": 0}
    graph = operational_graph.copy(); graph.add_nodes_from(assignments)
    agent_by_id = {agent.uuv_id: agent for agent in agents}
    for component_index, component_set in enumerate(nx.connected_components(graph)):
        component = sorted(component_set)
        tasks = _shared_task_pool(component_index, component, agent_by_id, known_maps)
        pair_records = []
        for uuv_id in component:
            agent = agent_by_id[uuv_id]; memory = PlanningMemory(uuv_id, known_maps[uuv_id])
            classical = [score_frontier_candidate_with_memory(agent, task, memory) for task in tasks]
            learned = [score_frontier_candidate_learned_39(
                agent, task, memory, graph.degree[uuv_id], model,
                feature_mean, feature_std, len(agents), device
            ) for task in tasks]
            result = choose_hybrid_frontier_43(classical, learned)
            if result["selected"] is None:
                continue
            source = result["decision_source"]; decision_counts[source] += 1
            preferred_id = result["selected"]["candidate_id"]
            for task_index, score in enumerate(classical):
                pair_records.append({
                    "uuv_id": uuv_id, "task_index": task_index,
                    "candidate_id": score["candidate_id"], "x": score["x"], "y": score["y"],
                    "effective_utility": float(score["utility"] + (1e-6 if score["candidate_id"] == preferred_id else 0.0)),
                    "source": source,
                })
        pair_records.sort(key=lambda record: record["effective_utility"], reverse=True)
        assigned_uuvs, assigned_tasks, positions = set(), set(), []
        for record in pair_records:
            if record["uuv_id"] in assigned_uuvs or record["task_index"] in assigned_tasks:
                continue
            position = (record["x"], record["y"])
            if any(euclidean_distance(position, assigned) < MIN_ASSIGNED_GOAL_SEPARATION for assigned in positions):
                continue
            assignments[record["uuv_id"]] = {"candidate_id": int(record["candidate_id"]), "x": int(record["x"]), "y": int(record["y"])}
            assigned_uuvs.add(record["uuv_id"]); assigned_tasks.add(record["task_index"]); positions.append(position)
    changes = sum(
        (None if previous_goals.get(uuv_id) is None else (previous_goals[uuv_id]["x"], previous_goals[uuv_id]["y"]))
        != (None if goal is None else (goal["x"], goal["y"]))
        for uuv_id, goal in assignments.items()
    )
    return assignments, changes, decision_counts
