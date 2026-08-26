"""Read-only architecture, model, benchmark, recovery, and artifact checks."""

from pathlib import Path

import numpy as np

from .config import (
    COMMUNICATION_RANGE,
    FOV_HEIGHT,
    FOV_WIDTH,
    HYBRID_CLASSICAL_REGRET_MAX_43,
    MAX_COMM_DELAY,
    MIN_COMM_DELAY,
    PACKET_LOSS_PROBABILITY,
    ROLLOUT_STEP_SIZE,
    SAFE_SEPARATION,
)


def validate_architecture(agents):
    checks = {
        "four_agents": len(agents) == 4,
        "fov_width_128": FOV_WIDTH == 128,
        "fov_height_128": FOV_HEIGHT == 128,
        "rollout_step_32": ROLLOUT_STEP_SIZE == 32,
        "safe_separation_128": bool(np.isclose(SAFE_SEPARATION, 128.0)),
        "communication_range_400": bool(np.isclose(COMMUNICATION_RANGE, 400.0)),
        "packet_loss_025": bool(np.isclose(PACKET_LOSS_PROBABILITY, 0.25)),
        "minimum_delay_1": MIN_COMM_DELAY == 1,
        "maximum_delay_5": MAX_COMM_DELAY == 5,
    }
    return checks, all(checks.values())


def validate_learned_model(metrics, top1_accuracy, parameter_count):
    checks = {
        "test_r2_gt_099": metrics["R2"] > 0.99,
        "test_spearman_gt_099": metrics["Spearman"] > 0.99,
        "top1_gt_090": top1_accuracy > 0.90,
        "model_parameters_positive": parameter_count > 0,
    }
    return checks, all(checks.values())


def validate_hybrid_regret(result):
    if result["classical_regret"] > HYBRID_CLASSICAL_REGRET_MAX_43 + 1e-9:
        raise RuntimeError("Hybrid gate allowed excessive classical regret.")
    return True


def controller_benchmark_winners(summary_df):
    """Combo 48 winner extraction, retaining pandas idxmin/idxmax tie behavior."""
    return {
        "mean_final_coverage": summary_df.loc[
            summary_df["final_coverage_mean"].idxmax(), "controller"
        ],
        "coverage_auc": summary_df.loc[
            summary_df["coverage_auc_mean"].idxmax(), "controller"
        ],
        "lowest_redundancy": summary_df.loc[
            summary_df["redundancy_mean"].idxmin(), "controller"
        ],
        "lowest_coverage_variance": summary_df.loc[
            summary_df["final_coverage_std"].idxmin(), "controller"
        ],
    }


def validate_hybrid_benchmark(summary_df):
    winners = controller_benchmark_winners(summary_df)
    checks = {
        "hybrid_best_mean_coverage": winners["mean_final_coverage"] == "Hybrid",
        "hybrid_best_coverage_auc": winners["coverage_auc"] == "Hybrid",
        "hybrid_lowest_redundancy": winners["lowest_redundancy"] == "Hybrid",
        "hybrid_lowest_coverage_variance": winners["lowest_coverage_variance"] == "Hybrid",
    }
    return checks, all(checks.values())


def validate_fault_recovery(metrics):
    checks = {
        "failure_confirmed_after_failure": metrics["majority_confirmation_time"] is not None
        and metrics["majority_confirmation_time"] > metrics["physical_failure_time"],
        "recovery_task_created": metrics["orphan_task"] is not None,
        "healthy_successor_selected": metrics["orphan_successor"] is not None
        and metrics["orphan_successor"] != metrics["failed_uuv"],
        "recovery_completed": bool(metrics["orphan_completed"]),
        "failed_uuv_immobile": bool(np.isclose(metrics["failed_displacement_px"], 0.0)),
        "safety_preserved": metrics["minimum_separation_px"] >= SAFE_SEPARATION,
        "all_resources_discovered": metrics["resources_discovered"] == metrics["resources_total"],
        "healthy_swarm_continued": metrics["healthy_motion_steps_after_failure"] > 0,
    }
    return checks, all(checks.values())


def validate_artifacts(paths):
    checks = {
        name: Path(path).exists() and Path(path).stat().st_size > 0
        for name, path in paths.items()
    }
    return checks, all(checks.values())


def validate_private_knowledge(agents, known_maps):
    for agent in agents:
        local = agent.exploration_memory.observed
        known = known_maps[agent.uuv_id]
        if np.any(local & (~known)):
            raise RuntimeError("Known map lost locally observed cells.")
        ids = [resource.resource_id for resource in agent.resource_memory.all_resources()]
        if len(ids) != len(set(ids)):
            raise RuntimeError("Duplicate resource IDs exist in private memory.")
    return True
