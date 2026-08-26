"""Private/known-map frontier extraction, sampling, and classical scoring."""

import cv2
import numpy as np

from .config import (
    FOV_HEIGHT,
    FOV_WIDTH,
    FRONTIER_CANDIDATE_SPACING,
    MAP_DIAGONAL,
    MAX_FRONTIER_CANDIDATES,
    MIN_FRONTIER_REGION_SIZE,
    W_DISTANCE,
    W_INFORMATION,
    W_REDUNDANCY,
)
from .memory import PlanningMemory
from .models import UUVState
from .motion import euclidean_distance
from .perception import get_fov_bounds


def extract_frontier_mask(exploration_memory):
    observed = exploration_memory.observed.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(observed, kernel, iterations=1)
    return (dilated > 0) & (~exploration_memory.observed)


def extract_frontier_candidates(
    frontier_mask: np.ndarray,
    min_region_size: int = MIN_FRONTIER_REGION_SIZE,
):
    if frontier_mask.ndim != 2:
        raise ValueError("frontier_mask must be a 2D boolean/binary array.")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        frontier_mask.astype(np.uint8), connectivity=8
    )
    candidates = []
    for label_id in range(1, num_labels):
        region_size = int(stats[label_id, cv2.CC_STAT_AREA])
        if region_size < min_region_size:
            continue
        centroid_x = float(centroids[label_id, 0])
        centroid_y = float(centroids[label_id, 1])
        region_y, region_x = np.where(labels == label_id)
        squared_distances = (region_x - centroid_x) ** 2 + (
            region_y - centroid_y
        ) ** 2
        nearest_index = int(np.argmin(squared_distances))
        candidates.append(
            {
                "component_id": int(label_id),
                "x": int(region_x[nearest_index]),
                "y": int(region_y[nearest_index]),
                "region_size": region_size,
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
            }
        )
    return candidates, num_labels - 1, labels


def sample_frontier_candidates(
    frontier_mask: np.ndarray,
    reference_position,
    min_spacing: float = FRONTIER_CANDIDATE_SPACING,
    max_candidates: int = MAX_FRONTIER_CANDIDATES,
):
    frontier_y, frontier_x = np.where(frontier_mask)
    if len(frontier_x) == 0:
        return []
    frontier_points = np.column_stack((frontier_x, frontier_y)).astype(float)
    ref = np.asarray(reference_position, dtype=float)
    distances_from_uuv = np.linalg.norm(frontier_points - ref, axis=1)
    selected_indices = [int(np.argmax(distances_from_uuv))]
    while len(selected_indices) < max_candidates:
        selected_points = frontier_points[selected_indices]
        pairwise_distances = np.linalg.norm(
            frontier_points[:, None, :] - selected_points[None, :, :], axis=2
        )
        nearest_selected_distance = pairwise_distances.min(axis=1)
        nearest_selected_distance[selected_indices] = -np.inf
        next_index = int(np.argmax(nearest_selected_distance))
        if float(nearest_selected_distance[next_index]) < min_spacing:
            break
        selected_indices.append(next_index)
    candidates = []
    for candidate_id, point_index in enumerate(selected_indices):
        x = int(frontier_points[point_index, 0])
        y = int(frontier_points[point_index, 1])
        candidates.append(
            {
                "candidate_id": candidate_id,
                "x": x,
                "y": y,
                "distance_from_uuv": euclidean_distance(reference_position, (x, y)),
            }
        )
    return candidates


def candidate_fov_statistics(
    exploration_memory,
    candidate_x: int,
    candidate_y: int,
    fov_width: int = FOV_WIDTH,
    fov_height: int = FOV_HEIGHT,
):
    temporary_state = UUVState(
        uuv_id=exploration_memory.uuv_id, x=int(candidate_x), y=int(candidate_y)
    )
    x_min, y_min, x_max, y_max = get_fov_bounds(
        temporary_state, fov_width=fov_width, fov_height=fov_height
    )
    x0, y0 = max(0, x_min), max(0, y_min)
    x1, y1 = min(exploration_memory.width, x_max), min(
        exploration_memory.height, y_max
    )
    memory_patch = exploration_memory.observed[y0:y1, x0:x1]
    valid_cells = int(memory_patch.size)
    if valid_cells <= 0:
        raise RuntimeError("Candidate FOV contains no valid world cells.")
    known_cells = int(memory_patch.sum())
    unknown_cells = valid_cells - known_cells
    return {
        "valid_cells": valid_cells,
        "unknown_cells": unknown_cells,
        "known_cells": known_cells,
        "information_ratio": float(unknown_cells / valid_cells),
        "redundancy_ratio": float(known_cells / valid_cells),
    }


def score_frontier_candidate_with_memory(agent, candidate, planning_memory):
    x, y = int(candidate["x"]), int(candidate["y"])
    stats = candidate_fov_statistics(planning_memory, x, y)
    distance_px = euclidean_distance((agent.state.x, agent.state.y), (x, y))
    distance_norm = distance_px / MAP_DIAGONAL
    information_ratio = stats["information_ratio"]
    redundancy_ratio = stats["redundancy_ratio"]
    utility = (
        W_INFORMATION * information_ratio
        - W_DISTANCE * distance_norm
        - W_REDUNDANCY * redundancy_ratio
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "x": x,
        "y": y,
        "distance_px": distance_px,
        "info_ratio": information_ratio,
        "redundancy_ratio": redundancy_ratio,
        "utility": utility,
    }


def score_frontier_candidate(agent, candidate):
    score = score_frontier_candidate_with_memory(
        agent, candidate, agent.exploration_memory
    )
    stats = candidate_fov_statistics(
        agent.exploration_memory, score["x"], score["y"]
    )
    return {
        "candidate_id": int(score["candidate_id"]),
        "x": score["x"],
        "y": score["y"],
        "distance_px": float(score["distance_px"]),
        "distance_norm": float(score["distance_px"] / MAP_DIAGONAL),
        "valid_cells": stats["valid_cells"],
        "unknown_cells": stats["unknown_cells"],
        "known_cells": stats["known_cells"],
        "info_ratio": score["info_ratio"],
        "redundancy_ratio": score["redundancy_ratio"],
        "utility": float(score["utility"]),
    }


def generate_private_frontier_goal(agent):
    """Combo 20 private-memory planner, including its empty-frontier behavior."""
    frontier = extract_frontier_mask(agent.exploration_memory)
    if int(frontier.sum()) == 0:
        return None
    candidates = sample_frontier_candidates(
        frontier,
        (agent.state.x, agent.state.y),
        min_spacing=FRONTIER_CANDIDATE_SPACING,
        max_candidates=MAX_FRONTIER_CANDIDATES,
    )
    if not candidates:
        return None
    scored = [score_frontier_candidate(agent, candidate) for candidate in candidates]
    return max(scored, key=lambda item: item["utility"])


def generate_goal_from_knowledge(agent, knowledge_mask):
    memory = PlanningMemory(agent.uuv_id, knowledge_mask)
    frontier = extract_frontier_mask(memory)
    candidates = sample_frontier_candidates(
        frontier,
        (agent.state.x, agent.state.y),
        min_spacing=FRONTIER_CANDIDATE_SPACING,
        max_candidates=MAX_FRONTIER_CANDIDATES,
    )
    if not candidates:
        return {
            "memory": memory,
            "frontier": frontier,
            "candidates": [],
            "scores": [],
            "selected": None,
        }
    scores = [
        score_frontier_candidate_with_memory(agent, candidate, memory)
        for candidate in candidates
    ]
    return {
        "memory": memory,
        "frontier": frontier,
        "candidates": candidates,
        "scores": scores,
        "selected": max(scores, key=lambda item: item["utility"]),
    }


def generate_known_map_goal(agent, known_map):
    return generate_goal_from_knowledge(agent, known_map)["selected"]
