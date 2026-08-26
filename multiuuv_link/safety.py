"""Pairwise risk checks and exhaustive predictive joint-action safety shields."""

from itertools import combinations, product

import numpy as np

from .config import (
    ACTION_HOLD,
    ROLLOUT_STEP_SIZE,
    SAFE_SEPARATION,
    SAFETY_ACTION_SET,
)
from .models import UUVState
from .motion import euclidean_distance, move_uuv_scaled


def evaluate_pairwise_safety(positions, safe_separation=SAFE_SEPARATION):
    results = []
    for uuv_a, uuv_b in combinations(sorted(positions), 2):
        distance = euclidean_distance(positions[uuv_a], positions[uuv_b])
        results.append(
            {
                "uuv_a": uuv_a,
                "uuv_b": uuv_b,
                "distance": float(distance),
                "unsafe": distance < safe_separation,
            }
        )
    return results


def predict_safety_state(
    state: UUVState, action: int, step_size: int = ROLLOUT_STEP_SIZE
):
    if action == ACTION_HOLD:
        return UUVState(state.uuv_id, state.x, state.y)
    return move_uuv_scaled(state, action, step_size)


def predict_joint_positions(
    states, joint_actions, step_size=ROLLOUT_STEP_SIZE
):
    predicted = {}
    for state, action in zip(states, joint_actions):
        next_state = predict_safety_state(state, action, step_size)
        predicted[state.uuv_id] = (next_state.x, next_state.y)
    return predicted


def minimum_pairwise_separation(positions):
    distances = []
    for uuv_a, uuv_b in combinations(sorted(positions), 2):
        distances.append(
            (euclidean_distance(positions[uuv_a], positions[uuv_b]), uuv_a, uuv_b)
        )
    if not distances:
        return np.inf, None
    return min(distances, key=lambda item: item[0])


def safety_shield(
    states,
    desired_actions,
    safe_separation=SAFE_SEPARATION,
    step_size=ROLLOUT_STEP_SIZE,
):
    if len(states) != len(desired_actions):
        raise ValueError("states and desired_actions must have equal length.")
    desired_actions = tuple(desired_actions)
    desired_positions = predict_joint_positions(states, desired_actions, step_size)
    desired_min_distance, desired_a, desired_b = minimum_pairwise_separation(
        desired_positions
    )
    best_solution = None
    for joint_actions in product(SAFETY_ACTION_SET, repeat=len(states)):
        positions = predict_joint_positions(states, joint_actions, step_size)
        minimum_distance, closest_a, closest_b = minimum_pairwise_separation(positions)
        if minimum_distance < safe_separation:
            continue
        override_count = sum(
            executed != desired
            for executed, desired in zip(joint_actions, desired_actions)
        )
        candidate = {
            "executed_actions": joint_actions,
            "predicted_positions": positions,
            "minimum_distance": float(minimum_distance),
            "closest_pair": (closest_a, closest_b),
            "override_count": int(override_count),
        }
        if best_solution is None:
            best_solution = candidate
        elif candidate["override_count"] < best_solution["override_count"]:
            best_solution = candidate
        elif (
            candidate["override_count"] == best_solution["override_count"]
            and candidate["minimum_distance"] > best_solution["minimum_distance"]
        ):
            best_solution = candidate
    if best_solution is None:
        raise RuntimeError("No safe joint action exists for the current state.")
    best_solution.update(
        {
            "desired_actions": desired_actions,
            "desired_positions": desired_positions,
            "desired_minimum_distance": float(desired_min_distance),
            "desired_closest_pair": (desired_a, desired_b),
        }
    )
    return best_solution


def minimum_state_separation_33(states):
    if len(states) < 2:
        return np.inf
    minimum_distance = np.inf
    for index in range(len(states)):
        for other in range(index + 1, len(states)):
            minimum_distance = min(
                minimum_distance,
                euclidean_distance(
                    (states[index].x, states[index].y),
                    (states[other].x, states[other].y),
                ),
            )
    return float(minimum_distance)


def predict_joint_states_33(states, actions, step_size=ROLLOUT_STEP_SIZE):
    return [
        predict_safety_state(state, action, step_size)
        for state, action in zip(states, actions)
    ]


def failure_aware_safety_shield_33(
    states,
    desired_actions,
    failed_mask,
    safe_separation=SAFE_SEPARATION,
    step_size=ROLLOUT_STEP_SIZE,
):
    action_options = [
        [ACTION_HOLD] if failed else list(SAFETY_ACTION_SET) for failed in failed_mask
    ]
    best_feasible = None
    for candidate_actions in product(*action_options):
        predicted_states = predict_joint_states_33(states, candidate_actions, step_size)
        minimum_distance = minimum_state_separation_33(predicted_states)
        if minimum_distance < safe_separation:
            continue
        override_count = sum(
            1
            for index, (desired, candidate) in enumerate(
                zip(desired_actions, candidate_actions)
            )
            if not failed_mask[index] and desired != candidate
        )
        record = {
            "executed_actions": list(candidate_actions),
            "minimum_distance": float(minimum_distance),
            "override_count": int(override_count),
        }
        if best_feasible is None:
            best_feasible = record
        elif record["override_count"] < best_feasible["override_count"]:
            best_feasible = record
        elif (
            record["override_count"] == best_feasible["override_count"]
            and record["minimum_distance"] > best_feasible["minimum_distance"]
        ):
            best_feasible = record
    if best_feasible is None:
        fallback_actions = [ACTION_HOLD for _ in states]
        fallback_states = predict_joint_states_33(states, fallback_actions, step_size)
        return {
            "executed_actions": fallback_actions,
            "minimum_distance": minimum_state_separation_33(fallback_states),
            "override_count": sum(
                1
                for index, desired in enumerate(desired_actions)
                if not failed_mask[index] and desired != ACTION_HOLD
            ),
            "fallback": True,
        }
    best_feasible["fallback"] = False
    return best_feasible
