"""Notebook-compatible metrics, summary, and PyTorch artifact export."""

import json
from pathlib import Path

import pandas as pd
import torch

from .config import (
    BLACKOUT_START_PROBABILITY,
    COMMUNICATION_RANGE,
    FOV_HEIGHT,
    FOV_WIDTH,
    MAX_COMM_DELAY,
    METRICS_CSV_NAME,
    MIN_COMM_DELAY,
    MODEL_NAME,
    PACKET_LOSS_PROBABILITY,
    ROLLOUT_STEP_SIZE,
    SAFE_SEPARATION,
    SUMMARY_JSON_NAME,
)
from .learning import model_export_payload


def controller_export_records(summary_df):
    return [
        {
            "controller": str(row["controller"]),
            "final_coverage_mean_pct": float(row["final_coverage_mean"]),
            "final_coverage_std_pct": float(row["final_coverage_std"]),
            "coverage_auc_mean": float(row["coverage_auc_mean"]),
            "redundancy_mean": float(row["redundancy_mean"]),
            "redundancy_std": float(row["redundancy_std"]),
            "hold_mean": float(row["hold_mean"]),
            "safety_override_mean": float(row["safety_override_mean"]),
            "goal_changes_mean": float(row["goal_changes_mean"]),
            "min_separation_mean_px": float(row["min_separation_mean"]),
        }
        for _, row in summary_df.iterrows()
    ]


def fault_recovery_export_record(mission_result):
    """Convert Combo 46 runtime fields to Combo 47's exact JSON schema."""
    orphan = mission_result["orphan_task"]
    return {
        "mission_steps": int(mission_result["mission_steps"]),
        "failed_uuv": int(mission_result["failed_uuv"]),
        "physical_failure_time": int(mission_result["physical_failure_time"]),
        "majority_confirmation_time": int(mission_result["majority_confirmation_time"]),
        "detection_delay": int(mission_result["detection_delay"]),
        "orphan_source": str(mission_result["orphan_source"]),
        "orphan_goal_x": int(orphan["x"]),
        "orphan_goal_y": int(orphan["y"]),
        "recovery_uuv": int(mission_result["orphan_successor"]),
        "completion_time": int(mission_result["completion_time"]),
        "total_recovery_delay": int(mission_result["total_recovery_delay"]),
        "failed_displacement_px": float(mission_result["failed_displacement_px"]),
        "final_coverage_pct": float(mission_result["final_coverage"] * 100.0),
        "resources_discovered": int(mission_result["resources_discovered"]),
        "resources_total": int(mission_result["resources_total"]),
        "safety_overrides": int(mission_result["safety_overrides"]),
        "minimum_separation_px": float(mission_result["minimum_separation_px"]),
    }


def export_final_artifacts(
    output_dir,
    summary_df,
    learned_metrics,
    top1_accuracy,
    model,
    feature_mean,
    feature_std,
    hybrid_authority,
    fault_metrics,
    artifact_links=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / METRICS_CSV_NAME
    json_path = output_dir / SUMMARY_JSON_NAME
    model_path = output_dir / MODEL_NAME
    controller_records = controller_export_records(summary_df)
    pd.DataFrame(controller_records).to_csv(csv_path, index=False)
    model_metrics = {
        "test_mse": float(learned_metrics["MSE"]),
        "test_mae": float(learned_metrics["MAE"]),
        "test_r2": float(learned_metrics["R2"]),
        "test_pearson": float(learned_metrics["Pearson"]),
        "test_spearman": float(learned_metrics["Spearman"]),
        "top1_agreement": float(top1_accuracy),
        "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    summary = {
        "project": "MultiUUV-Link",
        "prototype_status": "validated_research_prototype",
        "architecture": {
            "agents": 4,
            "local_fov_pixels": [FOV_WIDTH, FOV_HEIGHT],
            "motion_step_pixels": ROLLOUT_STEP_SIZE,
            "safe_separation_pixels": SAFE_SEPARATION,
            "communication_range_pixels": COMMUNICATION_RANGE,
            "packet_loss_probability": PACKET_LOSS_PROBABILITY,
            "communication_delay_range": [MIN_COMM_DELAY, MAX_COMM_DELAY],
            "blackout_start_probability": BLACKOUT_START_PROBABILITY,
        },
        "learned_model": model_metrics,
        "controllers": controller_records,
        "hybrid_authority": hybrid_authority,
        "fault_recovery_showcase": fault_metrics,
        "scientific_interpretation": {
            "learned_model_role": "The PyTorch model learns frontier-value ranking from classical supervision.",
            "closed_loop_result": "Fully learned ranking did not consistently outperform the classical planner across matched stochastic communication seeds.",
            "final_architecture_choice": "Confidence-gated hybrid planning retains the learned model while using classical fallbacks to bound ranking error.",
        },
        "artifacts": artifact_links or {},
    }
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    torch.save(model_export_payload(model, feature_mean, feature_std), model_path)
    validate_artifact_paths([csv_path, json_path, model_path])
    return {"metrics_csv": csv_path, "summary_json": json_path, "model": model_path}


def validate_artifact_paths(paths):
    for path in map(Path, paths):
        if not path.exists():
            raise RuntimeError(f"Artifact was not created: {path}")
        if path.stat().st_size <= 0:
            raise RuntimeError(f"Artifact is empty: {path}")
    return True
