"""Controlled paired controller benchmarks and notebook-equivalent summaries."""

import numpy as np
import pandas as pd

from .config import (
    BENCHMARK_SEEDS_42,
    BENCHMARK_SEEDS_44,
    BENCHMARK_STEPS_42,
    BENCHMARK_STEPS_44,
)
from .missions import run_controller_mission


def time_to_coverage_41(coverage_curve, threshold):
    matches = np.where(np.asarray(coverage_curve) >= threshold)[0]
    return None if len(matches) == 0 else int(matches[0])


def run_paired_controller_benchmark(
    checkpoint,
    resources,
    model,
    feature_mean,
    feature_std,
    seeds=BENCHMARK_SEEDS_44,
    steps=BENCHMARK_STEPS_44,
    device="cpu",
):
    records = []
    for seed in seeds:
        for controller in ("Classical", "Learned", "Hybrid"):
            records.append(
                run_controller_mission(
                    controller,
                    seed,
                    checkpoint,
                    resources,
                    steps,
                    model,
                    feature_mean,
                    feature_std,
                    device,
                )
            )
    return pd.DataFrame(records)


def run_classical_learned_benchmark_42(
    checkpoint,
    resources,
    model,
    feature_mean,
    feature_std,
    seeds=BENCHMARK_SEEDS_42,
    steps=BENCHMARK_STEPS_42,
    device="cpu",
):
    records = []
    for seed in seeds:
        records.append(
            run_controller_mission(
                "Classical", seed, checkpoint, resources, steps,
                model, feature_mean, feature_std, device
            )
        )
        records.append(
            run_controller_mission(
                "Learned", seed, checkpoint, resources, steps,
                model, feature_mean, feature_std, device
            )
        )
    return pd.DataFrame(records)


def paired_classical_learned_summary_42(benchmark_df, seeds=BENCHMARK_SEEDS_42):
    metric_columns = [
        "final_coverage_pct",
        "coverage_gain_pp",
        "coverage_auc",
        "incremental_redundancy",
        "unique_new_cells",
        "hold_steps",
        "safety_overrides",
        "goal_changes",
        "min_separation_px",
    ]
    controller_summary = benchmark_df.groupby("controller")[metric_columns].agg(
        ["mean", "std"]
    )
    classical = benchmark_df[benchmark_df["controller"] == "Classical"].set_index("seed")
    learned = benchmark_df[benchmark_df["controller"] == "Learned"].set_index("seed")
    paired_records = []
    for seed in seeds:
        classical_row, learned_row = classical.loc[seed], learned.loc[seed]
        paired_records.append(
            {
                "seed": seed,
                "final_coverage_diff_pp": learned_row["final_coverage_pct"]
                - classical_row["final_coverage_pct"],
                "coverage_auc_diff": learned_row["coverage_auc"]
                - classical_row["coverage_auc"],
                "redundancy_diff": learned_row["incremental_redundancy"]
                - classical_row["incremental_redundancy"],
                "hold_diff": learned_row["hold_steps"] - classical_row["hold_steps"],
                "safety_override_diff": learned_row["safety_overrides"]
                - classical_row["safety_overrides"],
                "goal_change_diff": learned_row["goal_changes"]
                - classical_row["goal_changes"],
            }
        )
    paired = pd.DataFrame(paired_records)
    summary = {
        "mean_final_coverage_diff_pp": float(paired["final_coverage_diff_pp"].mean()),
        "mean_coverage_auc_diff": float(paired["coverage_auc_diff"].mean()),
        "mean_redundancy_diff": float(paired["redundancy_diff"].mean()),
        "median_redundancy_diff": float(paired["redundancy_diff"].median()),
        "learned_lower_redundancy_seeds": int((paired["redundancy_diff"] < 0).sum()),
        "learned_higher_coverage_auc_seeds": int((paired["coverage_auc_diff"] > 0).sum()),
        "learned_higher_final_coverage_seeds": int(
            (paired["final_coverage_diff_pp"] > 0).sum()
        ),
    }
    return controller_summary, paired, summary


def summarize_controller_benchmark(benchmark_df):
    rows = []
    for controller in ("Classical", "Learned", "Hybrid"):
        subset = benchmark_df[benchmark_df["controller"] == controller]
        rows.append(
            {
                "controller": controller,
                "final_coverage_mean": subset["final_coverage_pct"].mean(),
                "final_coverage_std": subset["final_coverage_pct"].std(),
                "coverage_auc_mean": subset["coverage_auc"].mean(),
                "redundancy_mean": subset["incremental_redundancy"].mean(),
                "redundancy_std": subset["incremental_redundancy"].std(),
                "hold_mean": subset["hold_steps"].mean(),
                "safety_override_mean": subset["safety_overrides"].mean(),
                "goal_changes_mean": subset["goal_changes"].mean(),
                "min_separation_mean": subset["min_separation_px"].mean(),
            }
        )
    return pd.DataFrame(rows)


def hybrid_authority_summary(benchmark_df):
    hybrid = benchmark_df[benchmark_df["controller"] == "Hybrid"]
    agreements = int(hybrid["hybrid_agreements"].sum())
    overrides = int(hybrid["hybrid_overrides"].sum())
    fallbacks = int(hybrid["hybrid_fallbacks"].sum())
    total = agreements + overrides + fallbacks
    return {
        "learned_agreements": agreements,
        "learned_overrides": overrides,
        "classical_fallbacks": fallbacks,
        "total_decisions": total,
        "learned_authorized_fraction": (
            (agreements + overrides) / total if total > 0 else 0.0
        ),
    }
