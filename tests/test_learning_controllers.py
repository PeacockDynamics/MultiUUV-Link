import numpy as np

from multiuuv_link.config import FEATURE_COLUMNS_37, HYBRID_CLASSICAL_REGRET_MAX_43
from multiuuv_link.controllers import (
    choose_hybrid_frontier_43,
    generate_goal_from_knowledge_hybrid_43,
    generate_goal_from_knowledge_learned_39,
)
from multiuuv_link.learning import (
    FrontierValueNetwork38,
    generate_frontier_dataset,
    model_export_payload,
    split_frontier_dataset,
)
from multiuuv_link.validation import validate_hybrid_regret, validate_learned_model


def _scores(classical, learned):
    classical_records = [
        {"candidate_id": i, "x": i * 100, "y": 0, "utility": value}
        for i, value in enumerate(classical)
    ]
    learned_records = [
        {"candidate_id": i, "x": i * 100, "y": 0, "learned_utility": value}
        for i, value in enumerate(learned)
    ]
    return classical_records, learned_records


def test_network_architecture_and_export_payload():
    model = FrontierValueNetwork38()
    assert sum(parameter.numel() for parameter in model.parameters()) == 1313
    payload = model_export_payload(model, np.zeros((1, 6)), np.ones((1, 6)))
    assert payload["feature_columns"] == FEATURE_COLUMNS_37
    assert payload["architecture"] == {
        "input_dim": 6,
        "hidden_layers": [32, 32],
        "output_dim": 1,
        "activation": "ReLU",
    }


def test_synthetic_dataset_is_finite_private_and_group_split_has_no_leakage(small_swarm):
    agents, known_maps = small_swarm
    local_before = [agent.exploration_memory.observed.copy() for agent in agents]
    known_before = {key: value.copy() for key, value in known_maps.items()}
    dataset = generate_frontier_dataset(
        agents, known_maps, synthetic_states_per_uuv=4, max_candidates=6
    )
    assert not dataset.empty
    assert np.isfinite(dataset[FEATURE_COLUMNS_37].to_numpy()).all()
    assert set(dataset["uuv_id"]) == {0, 1, 2, 3}
    masks, groups = split_frontier_dataset(dataset)
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])
    assert sum(mask.sum() for mask in masks.values()) == len(dataset)
    for agent, snapshot in zip(agents, local_before):
        assert np.array_equal(agent.exploration_memory.observed, snapshot)
    for key in known_maps:
        assert np.array_equal(known_maps[key], known_before[key])


def test_validated_model_artifact_matches_notebook_thresholds(validated_model):
    model, _, _, payload = validated_model
    metrics = {"R2": 0.9999339298806628, "Spearman": 0.999954983132577}
    checks, valid = validate_learned_model(
        metrics, 0.9583333333333334, sum(p.numel() for p in model.parameters())
    )
    assert valid and all(checks.values())
    assert payload["training_seed"] == 380038


def test_hybrid_agreement_prefers_shared_candidate():
    classical, learned = _scores([0.9, 0.8], [0.9, 0.7])
    result = choose_hybrid_frontier_43(classical, learned)
    assert result["decision_source"] == "LEARNED_AGREEMENT"
    assert result["selected"]["candidate_id"] == 0
    assert validate_hybrid_regret(result)


def test_hybrid_override_requires_margin_and_regret_bounds():
    classical, learned = _scores([0.90, 0.89], [0.20, 0.24])
    accepted = choose_hybrid_frontier_43(classical, learned)
    assert accepted["decision_source"] == "LEARNED_OVERRIDE"
    assert accepted["classical_regret"] <= HYBRID_CLASSICAL_REGRET_MAX_43
    classical, learned = _scores([0.90, 0.87], [0.20, 0.24])
    rejected = choose_hybrid_frontier_43(classical, learned)
    assert rejected["decision_source"] == "CLASSICAL_FALLBACK"
    assert rejected["selected"]["candidate_id"] == 0


def test_hybrid_falls_back_when_learned_margin_is_below_point_zero_three():
    classical, learned = _scores([0.90, 0.89], [0.20, 0.225])
    result = choose_hybrid_frontier_43(classical, learned)
    assert result["decision_source"] == "CLASSICAL_FALLBACK"


def test_classical_and_learned_goal_generation_use_identical_candidate_order(small_swarm, validated_model):
    agents, known_maps = small_swarm
    model, mean, std, _ = validated_model
    agent = agents[0]
    learned = generate_goal_from_knowledge_learned_39(
        agent, known_maps[0], 3, model, mean, std, 4
    )
    hybrid = generate_goal_from_knowledge_hybrid_43(
        agent, known_maps[0], 3, model, mean, std, 4
    )
    learned_xy = [(r["x"], r["y"]) for r in learned["candidates"]]
    hybrid_xy = [(r["x"], r["y"]) for r in hybrid["candidates"]]
    assert learned_xy == hybrid_xy
