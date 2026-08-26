"""Run the paired classical/learned/hybrid benchmark from a common checkpoint."""

import argparse
from pathlib import Path

import numpy as np

from multiuuv_link.benchmarks import (
    run_paired_controller_benchmark,
    summarize_controller_benchmark,
)
from multiuuv_link.environment import generate_resources
from multiuuv_link.learning import load_frontier_value_model
from multiuuv_link.missions import prepare_notebook_checkpoint_24_5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=180)
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
    benchmark = run_paired_controller_benchmark(
        checkpoint, resources, model, mean, std, steps=args.steps
    )
    print(benchmark.to_string(index=False))
    print(summarize_controller_benchmark(benchmark).to_string(index=False))


if __name__ == "__main__":
    main()
