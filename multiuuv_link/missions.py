"""Validated synchronized mission orchestration and fault-recovery showcase."""

from copy import deepcopy

import networkx as nx
import numpy as np

from .allocation import goal_is_still_useful, perform_stable_allocation_round
from .communication import (
    DegradedCommunicationChannel,
    LinkBlackoutManager,
    build_communication_graph,
    deliver_due_failure_aware_33,
)
from .config import (
    ACTION_HOLD,
    ALLOCATION_COOLDOWN,
    ALLOCATION_INTERVAL,
    AUTONOMOUS_PLANNING_CYCLES,
    BLACKOUT_MAX_DURATION,
    BLACKOUT_MIN_DURATION,
    BLACKOUT_START_PROBABILITY,
    COMMUNICATION_RANGE,
    FAILED_UUV_ID_46,
    HEARTBEAT_THRESHOLD_46,
    HEIGHT,
    MAJORITY_CONFIRMATIONS_46,
    MAX_STEPS_PER_GOAL,
    MAX_COMM_DELAY,
    MIN_COMM_DELAY,
    MIN_SPAWN_SEPARATION,
    NUM_UUVS,
    PACKET_LOSS_PROBABILITY,
    PHYSICAL_FAILURE_TIME_46,
    ROLLOUT_STEPS,
    ROLLOUT_STEP_SIZE,
    SAFE_SEPARATION,
    SAFE_AUTONOMY_STEPS,
    SHOWCASE_SEED_46,
    SHOWCASE_STEPS_46,
    WIDTH,
)
from .controllers import (
    perform_hybrid_allocation_round_44,
    perform_learned_allocation_round_40,
)
from .fault_recovery import (
    choose_orphan_successor,
    create_safe_orphan_task,
    deliver_heartbeats,
    majority_detectors,
    queue_heartbeat,
)
from .frontier import generate_private_frontier_goal
from .memory import ExplorationMemory, ResourceMemory, initialize_known_maps
from .models import UUVAgent, UUVState, generate_safe_spawn_positions
from .motion import (
    choose_greedy_action,
    euclidean_distance,
    move_uuv_scaled,
    trajectory_path_length,
)
from .perception import discover_visible_resources
from .safety import (
    failure_aware_safety_shield_33,
    predict_joint_states_33,
    predict_safety_state,
    safety_shield,
)


def create_swarm(
    resources,
    rng=None,
    num_uuvs=NUM_UUVS,
    width=WIDTH,
    height=HEIGHT,
    min_separation=MIN_SPAWN_SEPARATION,
):
    if rng is None:
        rng = np.random.default_rng()
    positions, attempts = generate_safe_spawn_positions(
        num_uuvs, width, height, min_separation, rng
    )
    agents = []
    for uuv_id, (x, y) in enumerate(positions):
        state = UUVState(uuv_id, x, y)
        agent = UUVAgent(
            uuv_id,
            state,
            ExplorationMemory(uuv_id, width, height),
            ResourceMemory(uuv_id),
            [(x, y)],
        )
        agents.append(agent)
    for agent in agents:
        agent.observe(resources)
    return agents, attempts


def get_swarm_state_signature(agents, known_maps):
    return [
        {
            "uuv_id": agent.uuv_id,
            "position": (agent.state.x, agent.state.y),
            "trajectory_length": len(agent.trajectory),
            "local_cells": agent.exploration_memory.observed_count(),
            "known_cells": int(known_maps[agent.uuv_id].sum()),
            "known_resources": len(agent.resource_memory),
        }
        for agent in sorted(agents, key=lambda item: item.uuv_id)
    ]


def create_checkpoint(agents, known_maps, resource_provenance=None):
    return {
        "agents": deepcopy(agents),
        "known_exploration_maps": deepcopy(known_maps),
        "resource_provenance": deepcopy(
            resource_provenance
            if resource_provenance is not None
            else {agent.uuv_id: {} for agent in agents}
        ),
    }


def restore_checkpoint(checkpoint):
    return (
        deepcopy(checkpoint["agents"]),
        deepcopy(checkpoint["known_exploration_maps"]),
        deepcopy(checkpoint["resource_provenance"]),
    )


def run_independent_random_rollout(
    agents,
    resources,
    steps=ROLLOUT_STEPS,
    step_size=ROLLOUT_STEP_SIZE,
    rng=None,
):
    """Extract Combo 11's sequential per-agent random rollout."""
    if rng is None:
        rng = np.random.default_rng()
    information_gain_log = {agent.uuv_id: [] for agent in agents}
    action_log = {agent.uuv_id: [] for agent in agents}
    displacement_log = {agent.uuv_id: [] for agent in agents}
    for _ in range(steps):
        for agent in agents:
            action = int(rng.integers(0, 4))
            action_log[agent.uuv_id].append(action)
            previous = (agent.state.x, agent.state.y)
            next_state = move_uuv_scaled(agent.state, action, step_size)
            next_position = (next_state.x, next_state.y)
            displacement_log[agent.uuv_id].append(
                euclidean_distance(previous, next_position)
            )
            agent.state = next_state
            agent.trajectory.append(next_position)
            information_gain_log[agent.uuv_id].append(
                agent.exploration_memory.update_from_fov(agent.state)
            )
            agent.resource_memory.update(
                discover_visible_resources(agent.state, resources)
            )
    expected_length = steps + 1
    reports = []
    for agent in agents:
        if len(agent.trajectory) != expected_length:
            raise RuntimeError(
                f"UUV-{agent.uuv_id} trajectory contains {len(agent.trajectory)} "
                f"points; expected {expected_length}."
            )
        if any(not (0 <= x < WIDTH and 0 <= y < HEIGHT) for x, y in agent.trajectory):
            raise RuntimeError(f"UUV-{agent.uuv_id} contains an out-of-bounds point.")
        gains = np.asarray(information_gain_log[agent.uuv_id], dtype=int)
        displacements = np.asarray(displacement_log[agent.uuv_id], dtype=float)
        reports.append(
            {
                "uuv_id": agent.uuv_id,
                "final_x": agent.state.x,
                "final_y": agent.state.y,
                "trajectory_points": len(agent.trajectory),
                "path_length_px": trajectory_path_length(agent.trajectory),
                "mean_step_px": float(displacements.mean()),
                "observed_cells": agent.exploration_memory.observed_count(),
                "coverage_percent": agent.exploration_memory.coverage() * 100.0,
                "known_resources": len(agent.resource_memory),
                "new_cells_rollout": int(gains.sum()),
                "mean_info_gain": float(gains.mean()),
                "zero_gain_steps": int((gains == 0).sum()),
            }
        )
    if len({id(agent.exploration_memory) for agent in agents}) != len(agents):
        raise RuntimeError("Exploration-memory independence lost.")
    if len({id(agent.resource_memory) for agent in agents}) != len(agents):
        raise RuntimeError("Resource-memory independence lost.")
    if len({id(agent.trajectory) for agent in agents}) != len(agents):
        raise RuntimeError("Trajectory independence lost.")
    return {
        "reports": reports,
        "information_gain_log": information_gain_log,
        "action_log": action_log,
        "displacement_log": displacement_log,
    }


def pursue_selected_frontier_goals(
    agents,
    resources,
    selected_goals,
    max_steps=MAX_STEPS_PER_GOAL,
):
    """Extract Combo 16's sequential goal-pursuit phase."""
    reports, paths = [], {}
    for agent in agents:
        goal = selected_goals[agent.uuv_id]
        if goal is None:
            paths[agent.uuv_id] = [(agent.state.x, agent.state.y)]
            reports.append({"uuv_id": agent.uuv_id, "goal_reached": False, "steps_used": 0})
            continue
        goal_position = (goal["x"], goal["y"])
        start_position = (agent.state.x, agent.state.y)
        start_distance = euclidean_distance(start_position, goal_position)
        start_cells = agent.exploration_memory.observed_count()
        start_resources = len(agent.resource_memory)
        path, steps_used, goal_reached = [start_position], 0, False
        for _ in range(max_steps):
            distance = euclidean_distance(
                (agent.state.x, agent.state.y), goal_position
            )
            if distance <= ROLLOUT_STEP_SIZE:
                goal_reached = True
                break
            action = choose_greedy_action(agent.state, *goal_position)
            if action is None:
                goal_reached = True
                break
            next_state = move_uuv_scaled(agent.state, action, ROLLOUT_STEP_SIZE)
            agent.state = next_state
            position = (next_state.x, next_state.y)
            agent.trajectory.append(position)
            path.append(position)
            steps_used += 1
            agent.exploration_memory.update_from_fov(agent.state)
            agent.resource_memory.update(
                discover_visible_resources(agent.state, resources)
            )
        final_position = (agent.state.x, agent.state.y)
        final_distance = euclidean_distance(final_position, goal_position)
        goal_reached = goal_reached or final_distance <= ROLLOUT_STEP_SIZE
        if start_distance > ROLLOUT_STEP_SIZE and final_distance >= start_distance:
            raise RuntimeError(
                f"UUV-{agent.uuv_id}: goal pursuit failed to reduce goal distance."
            )
        paths[agent.uuv_id] = path
        reports.append(
            {
                "uuv_id": agent.uuv_id,
                "goal_x": goal_position[0],
                "goal_y": goal_position[1],
                "start_distance": start_distance,
                "final_distance": final_distance,
                "steps_used": steps_used,
                "goal_reached": goal_reached,
                "new_explored_cells": agent.exploration_memory.observed_count() - start_cells,
                "new_resources": len(agent.resource_memory) - start_resources,
                "final_x": final_position[0],
                "final_y": final_position[1],
            }
        )
    return {"reports": reports, "paths": paths}


def run_independent_frontier_exploration(
    agents,
    resources,
    planning_cycles=AUTONOMOUS_PLANNING_CYCLES,
    max_steps_per_goal=MAX_STEPS_PER_GOAL,
):
    """Extract Combo 17's agent-by-agent autonomous frontier cycles."""
    starts = {
        agent.uuv_id: {
            "coverage": agent.exploration_memory.coverage(),
            "observed_cells": agent.exploration_memory.observed_count(),
            "known_resources": len(agent.resource_memory),
            "trajectory_index": len(agent.trajectory) - 1,
        }
        for agent in agents
    }
    logs = {agent.uuv_id: [] for agent in agents}
    for agent in agents:
        for planning_cycle in range(planning_cycles):
            goal = generate_private_frontier_goal(agent)
            if goal is None:
                break
            goal_position = (goal["x"], goal["y"])
            start_position = (agent.state.x, agent.state.y)
            start_distance = euclidean_distance(start_position, goal_position)
            cells_before = agent.exploration_memory.observed_count()
            resources_before = len(agent.resource_memory)
            steps_used = 0
            for _ in range(max_steps_per_goal):
                current = (agent.state.x, agent.state.y)
                if euclidean_distance(current, goal_position) <= ROLLOUT_STEP_SIZE:
                    break
                action = choose_greedy_action(agent.state, *goal_position)
                if action is None:
                    break
                next_state = move_uuv_scaled(agent.state, action, ROLLOUT_STEP_SIZE)
                next_position = (next_state.x, next_state.y)
                if next_position == current:
                    break
                agent.state = next_state
                agent.trajectory.append(next_position)
                steps_used += 1
                agent.exploration_memory.update_from_fov(agent.state)
                agent.resource_memory.update(
                    discover_visible_resources(agent.state, resources)
                )
            logs[agent.uuv_id].append(
                {
                    "planning_cycle": planning_cycle,
                    "goal_x": goal_position[0],
                    "goal_y": goal_position[1],
                    "utility": goal["utility"],
                    "start_distance": start_distance,
                    "final_distance": euclidean_distance(
                        (agent.state.x, agent.state.y), goal_position
                    ),
                    "steps_used": steps_used,
                    "new_cells": agent.exploration_memory.observed_count() - cells_before,
                    "new_resources": len(agent.resource_memory) - resources_before,
                    "coverage": agent.exploration_memory.coverage(),
                }
            )
    for agent in agents:
        if logs[agent.uuv_id] and (
            agent.exploration_memory.observed_count()
            - starts[agent.uuv_id]["observed_cells"]
            <= 0
        ):
            raise RuntimeError(
                f"UUV-{agent.uuv_id} completed planning cycles but acquired no new territory."
            )
    return {"starts": starts, "logs": logs}


def run_safety_aware_frontier_exploration(
    agents,
    resources,
    steps=SAFE_AUTONOMY_STEPS,
):
    """Extract Combo 20's synchronized private-memory mission ordering."""
    active_goals = {agent.uuv_id: None for agent in agents}
    override_counts = {agent.uuv_id: 0 for agent in agents}
    desired_unsafe = intervention_steps = 0
    minimum_separation = np.inf
    desired_history, executed_history = [], []
    for _ in range(steps):
        for agent in agents:
            goal = active_goals[agent.uuv_id]
            if goal is not None and euclidean_distance(
                (agent.state.x, agent.state.y), (goal["x"], goal["y"])
            ) <= ROLLOUT_STEP_SIZE:
                active_goals[agent.uuv_id] = None
            if active_goals[agent.uuv_id] is None:
                active_goals[agent.uuv_id] = generate_private_frontier_goal(agent)
        desired_actions = []
        for agent in agents:
            goal = active_goals[agent.uuv_id]
            action = (
                ACTION_HOLD
                if goal is None
                else choose_greedy_action(agent.state, goal["x"], goal["y"])
            )
            desired_actions.append(ACTION_HOLD if action is None else action)
        shield = safety_shield(
            [agent.state for agent in agents],
            desired_actions,
            SAFE_SEPARATION,
            ROLLOUT_STEP_SIZE,
        )
        desired_history.append(shield["desired_minimum_distance"])
        executed_history.append(shield["minimum_distance"])
        desired_unsafe += int(shield["desired_minimum_distance"] < SAFE_SEPARATION)
        intervention_steps += int(shield["override_count"] > 0)
        minimum_separation = min(minimum_separation, shield["minimum_distance"])
        for agent, desired, executed in zip(
            agents, desired_actions, shield["executed_actions"]
        ):
            override_counts[agent.uuv_id] += int(desired != executed)
        next_states = [
            predict_safety_state(agent.state, action, ROLLOUT_STEP_SIZE)
            for agent, action in zip(agents, shield["executed_actions"])
        ]
        for agent, state in zip(agents, next_states):
            agent.state = state
            agent.trajectory.append((state.x, state.y))
        for agent in agents:
            agent.exploration_memory.update_from_fov(agent.state)
            agent.resource_memory.update(
                discover_visible_resources(agent.state, resources)
            )
    if minimum_separation < SAFE_SEPARATION:
        raise RuntimeError("Safety-aware mission violated minimum separation.")
    return {
        "active_goals": active_goals,
        "safety_overrides": override_counts,
        "desired_unsafe_timesteps": desired_unsafe,
        "intervention_timesteps": intervention_steps,
        "minimum_separation_px": minimum_separation,
        "desired_minimum_history": desired_history,
        "executed_minimum_history": executed_history,
    }


def prepare_notebook_checkpoint_24_5(
    resources,
    spawn_rng=None,
    motion_rng=None,
):
    """Replay the state-mutating Combos 10-23 that precede checkpoint 24.5."""
    agents, spawn_attempts = create_swarm(resources, spawn_rng)
    run_independent_random_rollout(agents, resources, rng=motion_rng)
    goals = {
        agent.uuv_id: generate_private_frontier_goal(agent) for agent in agents
    }
    pursue_selected_frontier_goals(agents, resources, goals)
    run_independent_frontier_exploration(agents, resources)
    run_safety_aware_frontier_exploration(agents, resources)
    positions = {
        agent.uuv_id: (agent.state.x, agent.state.y) for agent in agents
    }
    graph = build_communication_graph(positions)
    provenance = {
        agent.uuv_id: {
            resource.resource_id: "LOCAL"
            for resource in agent.resource_memory.all_resources()
        }
        for agent in agents
    }
    from .communication import (
        exchange_map_knowledge_one_hop,
        exchange_resource_knowledge_one_hop,
    )

    exchange_resource_knowledge_one_hop(graph, agents, provenance)
    known_maps = initialize_known_maps(agents)
    exchange_map_knowledge_one_hop(graph, known_maps)
    return {
        "checkpoint": create_checkpoint(agents, known_maps, provenance),
        "agents": agents,
        "known_maps": known_maps,
        "resource_provenance": provenance,
        "communication_graph": graph,
        "spawn_attempts": spawn_attempts,
    }


def _resource_memories(agents):
    return {agent.uuv_id: agent.resource_memory for agent in agents}


def _merge_local_maps(agents, known_maps):
    for agent in agents:
        known_maps[agent.uuv_id] = np.logical_or(
            known_maps[agent.uuv_id], agent.exploration_memory.observed
        )


def _sense_healthy(agents, resources, known_maps, failed_flags=None, provenance=None):
    for agent in agents:
        if failed_flags is not None and failed_flags[agent.uuv_id]:
            continue
        agent.exploration_memory.update_from_fov(agent.state)
        known_maps[agent.uuv_id] = np.logical_or(
            known_maps[agent.uuv_id], agent.exploration_memory.observed
        )
        visible = discover_visible_resources(agent.state, resources)
        agent.resource_memory.update(visible)
        if provenance is not None:
            for resource in visible:
                provenance[agent.uuv_id][resource.resource_id] = "LOCAL"


def _coverage(agents):
    union = np.logical_or.reduce(
        [agent.exploration_memory.observed for agent in agents]
    )
    return float(union.sum() / union.size)


def _mission_metrics(
    controller,
    seed,
    agents,
    start_maps,
    coverage_curve,
    hold_counts,
    safety_overrides,
    goal_changes,
    allocation_rounds,
    minimum_separation,
    hybrid_counts,
):
    final_maps = {
        agent.uuv_id: agent.exploration_memory.observed for agent in agents
    }
    start_union = np.logical_or.reduce(list(start_maps.values()))
    final_union = np.logical_or.reduce(list(final_maps.values()))
    start_coverage = float(start_union.sum() / start_union.size)
    final_coverage = float(final_union.sum() / final_union.size)
    new_masks = {
        uuv_id: final_maps[uuv_id] & (~start_maps[uuv_id])
        for uuv_id in final_maps
    }
    sum_new = sum(int(mask.sum()) for mask in new_masks.values())
    unique_new = int(np.logical_or.reduce(list(new_masks.values())).sum())
    redundancy = (sum_new - unique_new) / sum_new if sum_new > 0 else 0.0
    coverage_auc = float(np.trapezoid(np.asarray(coverage_curve) * 100.0))
    if minimum_separation < SAFE_SEPARATION:
        raise RuntimeError(f"{controller} run violated safety.")
    return {
        "seed": seed,
        "controller": controller,
        "start_coverage_pct": start_coverage * 100.0,
        "final_coverage_pct": final_coverage * 100.0,
        "coverage_gain_pp": (final_coverage - start_coverage) * 100.0,
        "coverage_auc": coverage_auc,
        "incremental_redundancy": redundancy,
        "unique_new_cells": unique_new,
        "hold_steps": sum(hold_counts.values()),
        "safety_overrides": sum(safety_overrides.values()),
        "goal_changes": goal_changes,
        "allocation_rounds": allocation_rounds,
        "min_separation_px": minimum_separation,
        "hybrid_agreements": hybrid_counts["LEARNED_AGREEMENT"],
        "hybrid_overrides": hybrid_counts["LEARNED_OVERRIDE"],
        "hybrid_fallbacks": hybrid_counts["CLASSICAL_FALLBACK"],
    }


def run_controller_mission(
    controller,
    channel_seed,
    checkpoint,
    resources,
    steps=180,
    model=None,
    feature_mean=None,
    feature_std=None,
    device="cpu",
):
    """Combo 44 mission runner with a supplied immutable common checkpoint."""
    if controller not in {"Classical", "Learned", "Hybrid"}:
        raise ValueError(f"Unknown controller: {controller}")
    if controller != "Classical" and model is None:
        raise ValueError("Learned and Hybrid missions require the validated model.")
    agents, known_maps, provenance = restore_checkpoint(checkpoint)
    _merge_local_maps(agents, known_maps)
    start_maps = {
        agent.uuv_id: agent.exploration_memory.observed.copy() for agent in agents
    }
    channel = DegradedCommunicationChannel(
        [agent.uuv_id for agent in agents],
        COMMUNICATION_RANGE,
        PACKET_LOSS_PROBABILITY,
        MIN_COMM_DELAY,
        MAX_COMM_DELAY,
        BLACKOUT_START_PROBABILITY,
        BLACKOUT_MIN_DURATION,
        BLACKOUT_MAX_DURATION,
        np.random.default_rng(channel_seed),
    )
    goals = {agent.uuv_id: None for agent in agents}
    previous_edges = None
    last_allocation_step = -ALLOCATION_COOLDOWN
    allocation_rounds = goal_changes = 0
    hold_counts = {agent.uuv_id: 0 for agent in agents}
    safety_overrides = {agent.uuv_id: 0 for agent in agents}
    minimum_separation = np.inf
    coverage_curve = []
    hybrid_counts = {
        "LEARNED_AGREEMENT": 0,
        "LEARNED_OVERRIDE": 0,
        "CLASSICAL_FALLBACK": 0,
    }
    for timestep in range(steps):
        channel.deliver_due(timestep, known_maps, _resource_memories(agents))
        _merge_local_maps(agents, known_maps)
        positions = {
            agent.uuv_id: (agent.state.x, agent.state.y) for agent in agents
        }
        operational_graph = channel.transmit_round(
            timestep, positions, known_maps, _resource_memories(agents)
        )["operational_graph"]
        goal_invalidated = False
        for agent in agents:
            if goals[agent.uuv_id] is not None and not goal_is_still_useful(
                agent, goals[agent.uuv_id], known_maps
            ):
                goals[agent.uuv_id] = None
                goal_invalidated = True
        edges = frozenset(tuple(sorted(edge)) for edge in operational_graph.edges())
        topology_changed = previous_edges is None or edges != previous_edges
        previous_edges = edges
        missing_goal = any(goals[agent.uuv_id] is None for agent in agents)
        should_allocate = timestep % ALLOCATION_INTERVAL == 0 or (
            (topology_changed or goal_invalidated or missing_goal)
            and timestep - last_allocation_step >= ALLOCATION_COOLDOWN
        )
        if should_allocate:
            if controller == "Classical":
                new_goals, changes, _, _, _ = perform_stable_allocation_round(
                    operational_graph, goals, agents, known_maps
                )
            elif controller == "Learned":
                new_goals, changes, _ = perform_learned_allocation_round_40(
                    operational_graph, goals, agents, known_maps, model,
                    feature_mean, feature_std, device
                )
            else:
                new_goals, changes, counts = perform_hybrid_allocation_round_44(
                    operational_graph, goals, agents, known_maps, model,
                    feature_mean, feature_std, device
                )
                for key in hybrid_counts:
                    hybrid_counts[key] += counts[key]
            goals = new_goals
            goal_changes += changes
            allocation_rounds += 1
            last_allocation_step = timestep
        desired_actions = []
        for agent in agents:
            goal = goals[agent.uuv_id]
            if goal is None:
                action = ACTION_HOLD
                hold_counts[agent.uuv_id] += 1
            else:
                action = choose_greedy_action(agent.state, goal["x"], goal["y"])
                if action is None:
                    action = ACTION_HOLD
                    hold_counts[agent.uuv_id] += 1
            desired_actions.append(action)
        shield = safety_shield(
            [agent.state for agent in agents], desired_actions,
            SAFE_SEPARATION, ROLLOUT_STEP_SIZE
        )
        minimum_separation = min(minimum_separation, shield["minimum_distance"])
        for agent, desired, executed in zip(agents, desired_actions, shield["executed_actions"]):
            if desired != executed:
                safety_overrides[agent.uuv_id] += 1
        next_states = [
            predict_safety_state(agent.state, action, ROLLOUT_STEP_SIZE)
            for agent, action in zip(agents, shield["executed_actions"])
        ]
        for agent, state in zip(agents, next_states):
            agent.state = state
            agent.trajectory.append((state.x, state.y))
        _sense_healthy(agents, resources, known_maps, provenance=provenance)
        coverage_curve.append(_coverage(agents))
    return _mission_metrics(
        controller, channel_seed, agents, start_maps, coverage_curve,
        hold_counts, safety_overrides, goal_changes, allocation_rounds,
        minimum_separation, hybrid_counts
    )


def run_fault_recovery_mission(
    checkpoint,
    resources,
    model,
    feature_mean,
    feature_std,
    steps=SHOWCASE_STEPS_46,
    seed=SHOWCASE_SEED_46,
    failed_uuv_id=FAILED_UUV_ID_46,
    failure_time=PHYSICAL_FAILURE_TIME_46,
    heartbeat_threshold=HEARTBEAT_THRESHOLD_46,
    majority_confirmations=MAJORITY_CONFIRMATIONS_46,
    device="cpu",
):
    """Combo 46 ordering without visualization side effects."""
    agents, known_maps, provenance = restore_checkpoint(checkpoint)
    _merge_local_maps(agents, known_maps)
    agent_by_id = {agent.uuv_id: agent for agent in agents}
    ids = [agent.uuv_id for agent in agents]
    rng = np.random.default_rng(seed)
    failed_flags = {uuv_id: False for uuv_id in ids}
    suspicion = {detector: {peer: 0 for peer in ids if peer != detector} for detector in ids}
    declared = {detector: set() for detector in ids}
    detection_time = {detector: {} for detector in ids}
    heartbeat_queue = []
    heartbeat_message_id = 0
    heartbeat_blackout = LinkBlackoutManager(
        ids, BLACKOUT_START_PROBABILITY, BLACKOUT_MIN_DURATION,
        BLACKOUT_MAX_DURATION, rng
    )
    channel = DegradedCommunicationChannel(
        ids, COMMUNICATION_RANGE, PACKET_LOSS_PROBABILITY,
        MIN_COMM_DELAY, MAX_COMM_DELAY, BLACKOUT_START_PROBABILITY,
        BLACKOUT_MIN_DURATION, BLACKOUT_MAX_DURATION, rng
    )
    goals = {uuv_id: None for uuv_id in ids}
    last_allocation_step = -ALLOCATION_COOLDOWN
    previous_edges = None
    fault_confirmed = False
    confirmation_time = None
    failure_position = known_snapshot = goal_snapshot = None
    orphan_task = orphan_source = orphan_successor = None
    orphan_active = orphan_completed = False
    orphan_completion_time = None
    mission_min_separation = np.inf
    healthy_motion_steps_after_failure = 0
    team_discovered_resources = {}
    safety_overrides = {uuv_id: 0 for uuv_id in ids}
    hold_counts = {uuv_id: 0 for uuv_id in ids}
    allocation_rounds = goal_changes = 0
    for timestep in range(steps):
        if timestep == failure_time:
            failed_agent = agent_by_id[failed_uuv_id]
            failure_position = (failed_agent.state.x, failed_agent.state.y)
            known_snapshot = known_maps[failed_uuv_id].copy()
            goal_snapshot = deepcopy(goals[failed_uuv_id])
            failed_flags[failed_uuv_id] = True
            goals[failed_uuv_id] = None
        deliver_due_failure_aware_33(
            channel, timestep, known_maps, _resource_memories(agents), failed_flags
        )
        heartbeat_queue, fresh_pairs = deliver_heartbeats(
            heartbeat_queue, timestep, failed_flags
        )
        _merge_local_maps(agents, known_maps)
        physical_positions = {
            agent.uuv_id: (agent.state.x, agent.state.y) for agent in agents
        }
        geometric_graph = build_communication_graph(physical_positions)
        heartbeat_graph = heartbeat_blackout.step(geometric_graph)["operational_graph"]
        for uuv_a, uuv_b in heartbeat_graph.edges():
            for sender, receiver in ((uuv_a, uuv_b), (uuv_b, uuv_a)):
                if failed_flags[sender] or failed_flags[receiver]:
                    continue
                heartbeat_message_id, _ = queue_heartbeat(
                    heartbeat_queue, sender, receiver, timestep, rng,
                    heartbeat_message_id
                )
        healthy_ids = [uuv_id for uuv_id in ids if not failed_flags[uuv_id]]
        for detector in healthy_ids:
            for peer in ids:
                if peer == detector or peer in declared[detector]:
                    continue
                if not geometric_graph.has_edge(detector, peer):
                    continue
                if (detector, peer) in fresh_pairs:
                    suspicion[detector][peer] = 0
                else:
                    suspicion[detector][peer] += 1
                if suspicion[detector][peer] >= heartbeat_threshold:
                    declared[detector].add(peer)
                    detection_time[detector][peer] = timestep
        if not fault_confirmed and timestep > failure_time:
            detectors = majority_detectors(healthy_ids, declared, failed_uuv_id)
            if len(detectors) >= majority_confirmations:
                fault_confirmed = True
                confirmation_time = timestep
                orphan_task, orphan_source = create_safe_orphan_task(
                    agent_by_id[failed_uuv_id], goal_snapshot, known_snapshot,
                    failure_position
                )
                orphan_successor, _ = choose_orphan_successor(
                    agents, orphan_task, failed_flags, known_maps
                )
                orphan_active = True
                goals[orphan_successor] = deepcopy(orphan_task)
        healthy_positions = {uuv_id: physical_positions[uuv_id] for uuv_id in healthy_ids}
        if len(healthy_positions) >= 2:
            operational_graph = channel.transmit_round(
                timestep, healthy_positions, known_maps, _resource_memories(agents)
            )["operational_graph"]
        else:
            operational_graph = nx.Graph(); operational_graph.add_nodes_from(healthy_ids)
        goal_invalidated = False
        for agent in agents:
            uuv_id = agent.uuv_id
            if failed_flags[uuv_id] or (orphan_active and uuv_id == orphan_successor):
                continue
            if goals[uuv_id] is not None and not goal_is_still_useful(agent, goals[uuv_id], known_maps):
                goals[uuv_id] = None; goal_invalidated = True
        operational_graph.add_nodes_from(healthy_ids)
        edges = frozenset(tuple(sorted(edge)) for edge in operational_graph.edges())
        topology_changed = previous_edges is None or edges != previous_edges
        previous_edges = edges
        missing_goal = any(goals[uuv_id] is None for uuv_id in healthy_ids)
        should_allocate = timestep % ALLOCATION_INTERVAL == 0 or (
            (topology_changed or goal_invalidated or missing_goal)
            and timestep - last_allocation_step >= ALLOCATION_COOLDOWN
        )
        if should_allocate and healthy_ids:
            new_goals, changes, _ = perform_hybrid_allocation_round_44(
                operational_graph, goals, agents, known_maps, model,
                feature_mean, feature_std, device
            )
            for uuv_id in healthy_ids:
                goals[uuv_id] = new_goals.get(uuv_id)
            goals[failed_uuv_id] = None
            if orphan_active:
                goals[orphan_successor] = deepcopy(orphan_task)
            goal_changes += changes; allocation_rounds += 1; last_allocation_step = timestep
        desired_actions = []
        for agent in agents:
            if failed_flags[agent.uuv_id]:
                action = ACTION_HOLD
            elif goals[agent.uuv_id] is None:
                action = ACTION_HOLD; hold_counts[agent.uuv_id] += 1
            else:
                action = choose_greedy_action(
                    agent.state, goals[agent.uuv_id]["x"], goals[agent.uuv_id]["y"]
                )
                if action is None:
                    action = ACTION_HOLD; hold_counts[agent.uuv_id] += 1
            desired_actions.append(action)
        shield = failure_aware_safety_shield_33(
            [agent.state for agent in agents], desired_actions,
            [failed_flags[agent.uuv_id] for agent in agents],
            SAFE_SEPARATION, ROLLOUT_STEP_SIZE
        )
        mission_min_separation = min(mission_min_separation, shield["minimum_distance"])
        for agent, desired, executed in zip(agents, desired_actions, shield["executed_actions"]):
            if not failed_flags[agent.uuv_id] and desired != executed:
                safety_overrides[agent.uuv_id] += 1
        previous_positions = {agent.uuv_id: (agent.state.x, agent.state.y) for agent in agents}
        next_states = predict_joint_states_33(
            [agent.state for agent in agents], shield["executed_actions"], ROLLOUT_STEP_SIZE
        )
        for agent, state in zip(agents, next_states):
            agent.state = state; agent.trajectory.append((state.x, state.y))
        if timestep >= failure_time and any(
            (agent.state.x, agent.state.y) != previous_positions[agent.uuv_id]
            for agent in agents if not failed_flags[agent.uuv_id]
        ):
            healthy_motion_steps_after_failure += 1
        _sense_healthy(agents, resources, known_maps, failed_flags, provenance)
        for agent in agents:
            if failed_flags[agent.uuv_id]:
                continue
            for resource in discover_visible_resources(agent.state, resources):
                team_discovered_resources[resource.resource_id] = resource
        if orphan_active:
            successor = agent_by_id[orphan_successor]
            if euclidean_distance(
                (successor.state.x, successor.state.y),
                (orphan_task["x"], orphan_task["y"])
            ) <= ROLLOUT_STEP_SIZE:
                orphan_completed = True; orphan_active = False
                orphan_completion_time = timestep; goals[orphan_successor] = None
    failed_agent = agent_by_id[failed_uuv_id]
    failed_displacement = (
        euclidean_distance(failure_position, (failed_agent.state.x, failed_agent.state.y))
        if failure_position is not None else np.nan
    )
    false_positives = sum(
        peer != failed_uuv_id
        for detector in ids if detector != failed_uuv_id
        for peer in declared[detector]
    )
    return {
        "mission_steps": steps,
        "failed_uuv": failed_uuv_id,
        "physical_failure_time": failure_time,
        "fault_confirmed": fault_confirmed,
        "majority_confirmation_time": confirmation_time,
        "detection_delay": None if confirmation_time is None else confirmation_time - failure_time,
        "false_positives": false_positives,
        "orphan_task": orphan_task,
        "orphan_source": orphan_source,
        "orphan_successor": orphan_successor,
        "orphan_completed": orphan_completed,
        "completion_time": orphan_completion_time,
        "total_recovery_delay": None if orphan_completion_time is None else orphan_completion_time - failure_time,
        "failed_displacement_px": float(failed_displacement),
        "final_coverage": _coverage(agents),
        "resources_discovered": len(team_discovered_resources),
        "resources_total": len(resources),
        "healthy_motion_steps_after_failure": healthy_motion_steps_after_failure,
        "safety_overrides": sum(safety_overrides.values()),
        "minimum_separation_px": mission_min_separation,
        "allocation_rounds": allocation_rounds,
        "goal_changes": goal_changes,
        "agents": agents,
        "known_maps": known_maps,
        "detection_time": detection_time,
    }
