"""Classically supervised frontier-value dataset, network, training, and inference."""

import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .communication import build_communication_graph
from .config import (
    BATCH_SIZE_38,
    FEATURE_COLUMNS_37,
    LEARN_DATASET_SEED,
    MAP_DIAGONAL,
    MAX_DATASET_CANDIDATES,
    MAX_SYNTHETIC_RETENTION,
    MIN_SYNTHETIC_RETENTION,
    NUM_EPOCHS_38,
    SYNTHETIC_STATES_PER_UUV,
    TARGET_COLUMN_37,
    TRAIN_SEED_38,
)
from .frontier import (
    extract_frontier_mask,
    sample_frontier_candidates,
    score_frontier_candidate_with_memory,
)
from .memory import PlanningMemory


def build_synthetic_known_map_37(source_mask, retention_probability, rng):
    keep_mask = rng.random(source_mask.shape) < retention_probability
    return source_mask & keep_mask


def build_frontier_features_37(
    agent, candidate, planning_memory, communication_degree, num_agents
):
    scored = score_frontier_candidate_with_memory(agent, candidate, planning_memory)
    features = {
        "information_ratio": float(scored["info_ratio"]),
        "distance_norm": float(scored["distance_px"] / MAP_DIAGONAL),
        "redundancy_ratio": float(scored["redundancy_ratio"]),
        "local_coverage": float(agent.exploration_memory.coverage()),
        "known_coverage": float(planning_memory.coverage()),
        "comm_degree_norm": float(communication_degree / max(1, num_agents - 1)),
    }
    return features, float(scored["utility"]), scored


def generate_frontier_dataset(
    agents,
    known_maps,
    seed=LEARN_DATASET_SEED,
    synthetic_states_per_uuv=SYNTHETIC_STATES_PER_UUV,
    max_candidates=MAX_DATASET_CANDIDATES,
):
    """Execute Combo 37 without mutating live agents or known maps."""
    rng = np.random.default_rng(seed)
    positions = {
        agent.uuv_id: (agent.state.x, agent.state.y) for agent in agents
    }
    graph = build_communication_graph(positions)
    degree = {agent.uuv_id: int(graph.degree[agent.uuv_id]) for agent in agents}
    records = []
    for agent in agents:
        source_known_map = known_maps[agent.uuv_id].copy()
        for synthetic_index in range(synthetic_states_per_uuv):
            retention_probability = float(
                rng.uniform(MIN_SYNTHETIC_RETENTION, MAX_SYNTHETIC_RETENTION)
            )
            synthetic_mask = build_synthetic_known_map_37(
                source_known_map, retention_probability, rng
            )
            planning_memory = PlanningMemory(agent.uuv_id, synthetic_mask)
            candidates = sample_frontier_candidates(
                extract_frontier_mask(planning_memory),
                (agent.state.x, agent.state.y),
                min_spacing=96,
                max_candidates=max_candidates,
            )
            for candidate_index, candidate in enumerate(candidates):
                features, target, _ = build_frontier_features_37(
                    agent,
                    candidate,
                    planning_memory,
                    degree[agent.uuv_id],
                    len(agents),
                )
                records.append(
                    {
                        "uuv_id": agent.uuv_id,
                        "synthetic_state": synthetic_index,
                        "retention_probability": retention_probability,
                        "candidate_index": candidate_index,
                        "candidate_x": int(candidate["x"]),
                        "candidate_y": int(candidate["y"]),
                        **features,
                        TARGET_COLUMN_37: target,
                    }
                )
    dataset = pd.DataFrame(records)
    if dataset.empty:
        raise RuntimeError("Combo 37 generated an empty dataset.")
    features = dataset[FEATURE_COLUMNS_37].to_numpy(dtype=np.float32)
    targets = dataset[TARGET_COLUMN_37].to_numpy(dtype=np.float32).reshape(-1, 1)
    if not np.isfinite(features).all():
        raise RuntimeError("Non-finite frontier features detected.")
    if not np.isfinite(targets).all():
        raise RuntimeError("Non-finite training targets detected.")
    if set(dataset["uuv_id"].unique()) != {agent.uuv_id for agent in agents}:
        raise RuntimeError("Dataset does not contain all UUV IDs.")
    return dataset


def group_tuple_set_38(group_keys, indices):
    return set(
        tuple(row)
        for row in group_keys.iloc[indices][["uuv_id", "synthetic_state"]].to_numpy()
    )


def split_frontier_dataset(dataset, seed=TRAIN_SEED_38):
    group_keys = dataset[["uuv_id", "synthetic_state"]].drop_duplicates().reset_index(drop=True)
    indices = np.arange(len(group_keys))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    num_train = int(round(0.70 * len(indices)))
    num_val = int(round(0.15 * len(indices)))
    train_groups = group_tuple_set_38(group_keys, indices[:num_train])
    val_groups = group_tuple_set_38(group_keys, indices[num_train : num_train + num_val])
    test_groups = group_tuple_set_38(group_keys, indices[num_train + num_val :])
    pairs = list(zip(dataset["uuv_id"].astype(int), dataset["synthetic_state"].astype(int)))
    masks = {
        "train": np.asarray([pair in train_groups for pair in pairs], dtype=bool),
        "validation": np.asarray([pair in val_groups for pair in pairs], dtype=bool),
        "test": np.asarray([pair in test_groups for pair in pairs], dtype=bool),
    }
    return masks, {"train": train_groups, "validation": val_groups, "test": test_groups}


def normalize_features_38(X_np, feature_mean_np, feature_std_np):
    return ((X_np - feature_mean_np) / feature_std_np).astype(np.float32)


class FrontierValueNetwork38(nn.Module):
    def __init__(self, input_dim=6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


def regression_metrics_38(y_true, y_pred):
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    true_ranks = pd.Series(y_true).rank().to_numpy()
    pred_ranks = pd.Series(y_pred).rank().to_numpy()
    spearman = float(np.corrcoef(true_ranks, pred_ranks)[0, 1])
    return {"MSE": mse, "MAE": mae, "R2": r2, "Pearson": pearson, "Spearman": spearman}


def train_frontier_value_network(
    dataset,
    seed=TRAIN_SEED_38,
    epochs=NUM_EPOCHS_38,
    batch_size=BATCH_SIZE_38,
    device=None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    masks, groups = split_frontier_dataset(dataset, seed)
    X = dataset[FEATURE_COLUMNS_37].to_numpy(dtype=np.float32)
    y = dataset[TARGET_COLUMN_37].to_numpy(dtype=np.float32).reshape(-1, 1)
    X_train, y_train = X[masks["train"]], y[masks["train"]]
    feature_mean = X_train.mean(axis=0, keepdims=True)
    feature_std = X_train.std(axis=0, keepdims=True)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
    tensors = {}
    arrays = {}
    for split in ("train", "validation", "test"):
        split_X = X[masks[split]]
        split_y = y[masks[split]]
        arrays[split] = (split_X, split_y)
        tensors[split] = (
            torch.tensor(normalize_features_38(split_X, feature_mean, feature_std), dtype=torch.float32),
            torch.tensor(split_y, dtype=torch.float32),
        )
    train_loader = DataLoader(
        TensorDataset(*tensors["train"]),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = FrontierValueNetwork38(len(FEATURE_COLUMNS_37)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_losses, val_losses = [], []
    val_X, val_y = (tensor.to(device) for tensor in tensors["validation"])
    for _ in range(epochs):
        model.train()
        loss_sum, samples = 0.0, 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * batch_X.shape[0]
            samples += batch_X.shape[0]
        train_losses.append(loss_sum / samples)
        model.eval()
        with torch.no_grad():
            val_losses.append(criterion(model(val_X), val_y).item())
    metrics, predictions = {}, {}
    model.eval()
    with torch.no_grad():
        for split in ("train", "validation", "test"):
            pred = model(tensors[split][0].to(device)).cpu().numpy().reshape(-1)
            predictions[split] = pred
            metrics[split] = regression_metrics_38(arrays[split][1].reshape(-1), pred)
    test_frame = dataset.loc[masks["test"]].copy().reset_index(drop=True)
    test_frame["predicted_utility"] = predictions["test"]
    total_groups = matches = 0
    for _, group in test_frame.groupby(["uuv_id", "synthetic_state"]):
        total_groups += 1
        matches += int(group[TARGET_COLUMN_37].idxmax() == group["predicted_utility"].idxmax())
    return {
        "model": model,
        "device": device,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "masks": masks,
        "groups": groups,
        "metrics": metrics,
        "top1_accuracy": matches / total_groups,
        "top1_matches": matches,
        "top1_groups": total_groups,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_loss_history": train_losses,
        "val_loss_history": val_losses,
    }


def model_export_payload(model, feature_mean, feature_std, training_seed=TRAIN_SEED_38):
    return {
        "model_state_dict": model.state_dict(),
        "feature_columns": list(FEATURE_COLUMNS_37),
        "feature_mean": np.asarray(feature_mean, dtype=np.float32),
        "feature_std": np.asarray(feature_std, dtype=np.float32),
        "architecture": {
            "input_dim": len(FEATURE_COLUMNS_37),
            "hidden_layers": [32, 32],
            "output_dim": 1,
            "activation": "ReLU",
        },
        "training_seed": int(training_seed),
    }


def load_frontier_value_model(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = FrontierValueNetwork38(payload["architecture"]["input_dim"])
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, np.asarray(payload["feature_mean"]), np.asarray(payload["feature_std"]), payload
