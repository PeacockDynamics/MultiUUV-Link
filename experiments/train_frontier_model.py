"""Reproduce the notebook's classically supervised training procedure."""

import argparse
from pathlib import Path

import numpy as np
import torch

from multiuuv_link.environment import generate_resources
from multiuuv_link.learning import (
    generate_frontier_dataset,
    model_export_payload,
    train_frontier_value_network,
)
from multiuuv_link.missions import prepare_notebook_checkpoint_24_5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario-seed", type=int, default=42)
    parser.add_argument("--spawn-seed", type=int, default=42)
    parser.add_argument("--motion-seed", type=int, default=42)
    args = parser.parse_args()
    resources = generate_resources(
        1448, 1086, rng=np.random.default_rng(args.scenario_seed)
    )
    prepared = prepare_notebook_checkpoint_24_5(
        resources,
        np.random.default_rng(args.spawn_seed),
        np.random.default_rng(args.motion_seed),
    )
    agents, known_maps = prepared["agents"], prepared["known_maps"]
    dataset = generate_frontier_dataset(agents, known_maps)
    result = train_frontier_value_network(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        model_export_payload(
            result["model"], result["feature_mean"], result["feature_std"]
        ),
        args.output,
    )
    print(result["metrics"]["test"])
    print(f"top1_accuracy={result['top1_accuracy']:.6f}")


if __name__ == "__main__":
    main()
