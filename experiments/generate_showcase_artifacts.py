"""Generate README-ready MultiUUV-Link showcase artifacts.

Outputs:
- Fault-recovery mission GIF
- Controller benchmark metrics CSV
- Fault-recovery metrics JSON
- README-ready Markdown summary
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from multiuuv_link.benchmarks import (
    run_paired_controller_benchmark,
    summarize_controller_benchmark,
)
from multiuuv_link.environment import generate_resources, load_seabed
from multiuuv_link.learning import load_frontier_value_model
from multiuuv_link.missions import (
    prepare_notebook_checkpoint_24_5,
    run_fault_recovery_mission,
)
from multiuuv_link.models import UUVState
from multiuuv_link.visualization import render_mission_frame, save_gif


def build_fault_recovery_gif(
    seabed_rgb,
    checkpoint,
    result,
    output_path,
    frame_stride=4,
    fps=8,
):
    """Reconstruct mission motion from stored trajectories for README display."""

    final_agents = result["agents"]
    initial_agents = {
        agent.uuv_id: agent for agent in checkpoint["agents"]
    }

    mission_steps = int(result["mission_steps"])
    failed_uuv = int(result["failed_uuv"])
    failure_time = int(result["physical_failure_time"])
    confirmation_time = result["majority_confirmation_time"]
    completion_time = result["completion_time"]

    frames = []

    timesteps = list(range(0, mission_steps + 1, frame_stride))
    if timesteps[-1] != mission_steps:
        timesteps.append(mission_steps)

    for timestep in timesteps:
        snapshot_agents = []

        for final_agent in final_agents:
            agent = deepcopy(final_agent)

            # The fault mission starts after the notebook-equivalent checkpoint.
            checkpoint_length = len(initial_agents[agent.uuv_id].trajectory)

            trajectory_index = min(
                checkpoint_length - 1 + timestep,
                len(final_agent.trajectory) - 1,
            )

            mission_path = final_agent.trajectory[
                checkpoint_length - 1 : trajectory_index + 1
            ]

            if not mission_path:
                mission_path = [
                    (
                        initial_agents[agent.uuv_id].state.x,
                        initial_agents[agent.uuv_id].state.y,
                    )
                ]

            agent.trajectory = list(mission_path)

            x, y = mission_path[-1]
            agent.state = UUVState(
                uuv_id=agent.uuv_id,
                x=int(x),
                y=int(y),
            )

            snapshot_agents.append(agent)

        failed_flags = {
            agent.uuv_id: (
                agent.uuv_id == failed_uuv
                and timestep >= failure_time
            )
            for agent in snapshot_agents
        }

        if timestep < failure_time:
            phase = "Nominal Mission"

        elif confirmation_time is None or timestep < confirmation_time:
            phase = "Physical Failure | Awaiting Distributed Confirmation"

        elif completion_time is None or timestep < completion_time:
            phase = "Confirmed Failure | Orphan Task Reallocation"

        else:
            phase = "Recovery Complete"

        frame = render_mission_frame(
            seabed_rgb=seabed_rgb,
            agents=snapshot_agents,
            timestep=timestep,
            failed_flags=failed_flags,
            title=f"MultiUUV-Link | {phase}",
        )

        frames.append(frame)

    save_gif(frames, output_path, fps=fps)


def make_readme_markdown(
    summary_df,
    fault_result,
    gif_relative_path,
):
    """Build a compact README-ready showcase section."""

    rows = []

    for _, row in summary_df.iterrows():
        rows.append(
            "| "
            f"{row['controller']} | "
            f"{row['final_coverage_mean']:.2f}% | "
            f"{row['coverage_auc_mean']:.1f} | "
            f"{row['redundancy_mean']:.3f} | "
            f"{row['safety_override_mean']:.1f} | "
            f"{row['min_separation_mean']:.2f} px |"
        )

    benchmark_table = "\n".join(rows)

    markdown = f"""## MultiUUV-Link Showcase

### Communication-Aware Decentralized Multi-UUV Mission

![MultiUUV-Link fault-recovery showcase]({gif_relative_path})

The showcase demonstrates four UUVs operating with local perception,
persistent private knowledge, intermittent communication, distributed task
allocation, collision-aware motion, learned frontier ranking, and
failure-aware mission recovery.

### Controller Benchmark

| Controller | Final Coverage | Coverage AUC | Redundancy | Safety Overrides | Mean Minimum Separation |
| --- | ---: | ---: | ---: | ---: | ---: |
{benchmark_table}

### Fault-Recovery Demonstration

| Metric | Result |
| --- | ---: |
| Physical failure time | {fault_result['physical_failure_time']} |
| Majority confirmation time | {fault_result['majority_confirmation_time']} |
| Detection delay | {fault_result['detection_delay']} steps |
| Recovery UUV | UUV-{fault_result['orphan_successor']} |
| Recovery completed | {fault_result['orphan_completed']} |
| Completion time | {fault_result['completion_time']} |
| Total recovery delay | {fault_result['total_recovery_delay']} steps |
| Failed-UUV displacement | {fault_result['failed_displacement_px']:.2f} px |
| Final coverage | {100.0 * fault_result['final_coverage']:.2f}% |
| Resources discovered | {fault_result['resources_discovered']}/{fault_result['resources_total']} |
| Minimum separation | {fault_result['minimum_separation_px']:.2f} px |
| Safety overrides | {fault_result['safety_overrides']} |
| False-positive failures | {fault_result['false_positives']} |

> Results above are generated from a fresh modular execution and may differ
> numerically from the original notebook demonstration because the historical
> notebook contained stochastic states that were not fully checkpointed.
"""

    return markdown


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("experiments/output/frontier_value_model.pt"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/readme_showcase"),
    )

    parser.add_argument("--scenario-seed", type=int, default=42)
    parser.add_argument("--spawn-seed", type=int, default=42)
    parser.add_argument("--motion-seed", type=int, default=42)

    parser.add_argument("--benchmark-steps", type=int, default=180)

    parser.add_argument("--gif-stride", type=int, default=4)
    parser.add_argument("--gif-fps", type=int, default=8)

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("MultiUUV-Link | README Showcase Generator")
    print("=" * 76)

    # ------------------------------------------------------------------
    # Load map + create deterministic fresh scenario
    # ------------------------------------------------------------------

    print("\n[1/5] Preparing scenario...")

    _, seabed_rgb = load_seabed()

    resources = generate_resources(
        1448,
        1086,
        rng=np.random.default_rng(args.scenario_seed),
    )

    prepared = prepare_notebook_checkpoint_24_5(
        resources,
        np.random.default_rng(args.spawn_seed),
        np.random.default_rng(args.motion_seed),
    )

    checkpoint = prepared["checkpoint"]

    # ------------------------------------------------------------------
    # Load trained learned frontier model
    # ------------------------------------------------------------------

    print("[2/5] Loading learned frontier model...")

    model, feature_mean, feature_std, _ = load_frontier_value_model(
        args.model
    )

    # ------------------------------------------------------------------
    # Controller benchmark
    # ------------------------------------------------------------------

    print("[3/5] Running Classical / Learned / Hybrid benchmark...")

    benchmark_df = run_paired_controller_benchmark(
        checkpoint,
        resources,
        model,
        feature_mean,
        feature_std,
        steps=args.benchmark_steps,
    )

    summary_df = summarize_controller_benchmark(benchmark_df)

    benchmark_runs_path = (
        args.output_dir / "controller_benchmark_runs.csv"
    )

    benchmark_summary_path = (
        args.output_dir / "controller_benchmark_summary.csv"
    )

    benchmark_df.to_csv(benchmark_runs_path, index=False)
    summary_df.to_csv(benchmark_summary_path, index=False)

    print("\nController summary:")
    print(summary_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Fault recovery
    # ------------------------------------------------------------------

    print("\n[4/5] Running fault-recovery mission...")

    fault_result = run_fault_recovery_mission(
        checkpoint,
        resources,
        model,
        feature_mean,
        feature_std,
    )

    fault_metrics = {
        key: value
        for key, value in fault_result.items()
        if key not in {"agents", "known_maps", "detection_time"}
    }

    fault_metrics_path = (
        args.output_dir / "fault_recovery_metrics.json"
    )

    with fault_metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            fault_metrics,
            file,
            indent=2,
            default=lambda obj: (
                obj.item()
                if isinstance(obj, np.generic)
                else str(obj)
            ),
        )

    # ------------------------------------------------------------------
    # GIF
    # ------------------------------------------------------------------

    print("[5/5] Rendering README showcase GIF...")

    gif_path = (
        args.output_dir / "multiuuv_fault_recovery_showcase.gif"
    )

    build_fault_recovery_gif(
        seabed_rgb=seabed_rgb,
        checkpoint=checkpoint,
        result=fault_result,
        output_path=gif_path,
        frame_stride=args.gif_stride,
        fps=args.gif_fps,
    )

    # ------------------------------------------------------------------
    # README-ready Markdown
    # ------------------------------------------------------------------

    readme_snippet_path = (
        args.output_dir / "README_SHOWCASE.md"
    )

    gif_relative_path = (
        "artifacts/readme_showcase/"
        "multiuuv_fault_recovery_showcase.gif"
    )

    readme_snippet = make_readme_markdown(
        summary_df,
        fault_result,
        gif_relative_path,
    )

    readme_snippet_path.write_text(
        readme_snippet,
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    output_files = [
        benchmark_runs_path,
        benchmark_summary_path,
        fault_metrics_path,
        gif_path,
        readme_snippet_path,
    ]

    for path in output_files:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(
                f"Artifact generation failed: {path}"
            )

    print("\n" + "=" * 76)
    print("SHOWCASE COMPLETE")
    print("=" * 76)

    for path in output_files:
        size_kb = path.stat().st_size / 1024.0
        print(f"{path}  [{size_kb:.1f} KB]")

    print("\nREADME headline metrics:")
    print(
        f"  final coverage          "
        f"{fault_result['final_coverage'] * 100:.2f}%"
    )
    print(
        f"  resources discovered    "
        f"{fault_result['resources_discovered']}/"
        f"{fault_result['resources_total']}"
    )
    print(
        f"  minimum separation      "
        f"{fault_result['minimum_separation_px']:.2f} px"
    )
    print(
        f"  failed displacement     "
        f"{fault_result['failed_displacement_px']:.2f} px"
    )
    print(
        f"  orphan task completed   "
        f"{fault_result['orphan_completed']}"
    )

    print(
        "\nPaste the contents of "
        f"{readme_snippet_path} into README.md."
    )


if __name__ == "__main__":
    main()