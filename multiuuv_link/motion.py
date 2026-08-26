"""Discrete point-UUV motion and goal pursuit primitives."""

import numpy as np

from .config import (
    ACTION_DELTAS,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    HEIGHT,
    ROLLOUT_STEP_SIZE,
    WIDTH,
)
from .models import UUVState


def euclidean_distance(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def move_uuv(
    state: UUVState,
    action: int,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> UUVState:
    if action not in ACTION_DELTAS:
        raise ValueError(
            f"Invalid action {action}. Valid actions are {tuple(ACTION_DELTAS.keys())}."
        )
    dx, dy = ACTION_DELTAS[action]
    return UUVState(
        state.uuv_id,
        int(np.clip(state.x + dx, 0, width - 1)),
        int(np.clip(state.y + dy, 0, height - 1)),
    )


def move_uuv_scaled(
    state: UUVState,
    action: int,
    step_size: int,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> UUVState:
    if step_size <= 0:
        raise ValueError("step_size must be positive.")
    action_deltas = {
        ACTION_UP: (0, -step_size),
        ACTION_RIGHT: (step_size, 0),
        ACTION_DOWN: (0, step_size),
        ACTION_LEFT: (-step_size, 0),
    }
    if action not in action_deltas:
        raise ValueError(f"Invalid action {action}.")
    dx, dy = action_deltas[action]
    return UUVState(
        state.uuv_id,
        int(np.clip(state.x + dx, 0, width - 1)),
        int(np.clip(state.y + dy, 0, height - 1)),
    )


def trajectory_path_length(trajectory):
    if len(trajectory) < 2:
        return 0.0
    return float(
        sum(
            euclidean_distance(trajectory[index - 1], trajectory[index])
            for index in range(1, len(trajectory))
        )
    )


def choose_greedy_action(state: UUVState, goal_x: int, goal_y: int):
    error_x = goal_x - state.x
    error_y = goal_y - state.y
    if abs(error_x) >= abs(error_y):
        if error_x > 0:
            return ACTION_RIGHT
        if error_x < 0:
            return ACTION_LEFT
    if error_y > 0:
        return ACTION_DOWN
    if error_y < 0:
        return ACTION_UP
    return None


def action_toward_goal(state: UUVState, goal) -> int | None:
    return choose_greedy_action(state, int(goal["x"]), int(goal["y"]))


OPERATIONAL_STEP_SIZE = ROLLOUT_STEP_SIZE
