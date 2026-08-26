"""Run the validated fault-detection and recovery mission without writing artifacts."""

import argparse
from pathlib import Path

import numpy as np

from multiuuv_link.environment import generate_resources
from multiuuv_link.learning import load_frontier_value_model
from multiuuv_link.missions import (
    prepare_notebook_checkpoint_24_5,
    run_fault_recovery_mission,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scenario-seed", type=int, default=42)
    parser.add_argument("--spawn-seed", type=int, default=42)
    parser.add_argument("--motion-seed", type=int, default=42)
    args = parser.parse_args()
    resources = generate_resources(
        1448, 1086, rng=np.random.default_rng(args.scenario_seed)
    )
    checkpoint = prepare_notebook_checkpoint_24_5(
        resources,
        np.random.default_rng(args.spawn_seed),
        np.random.default_rng(args.motion_seed),
    )["checkpoint"]
    model, mean, std, _ = load_frontier_value_model(args.model)
    result = run_fault_recovery_mission(checkpoint, resources, model, mean, std)
    for key, value in result.items():
        if key not in {"agents", "known_maps", "detection_time"}:
            print(f"{key}={value}")


if __name__ == "__main__":
    main()
