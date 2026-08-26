import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from multiuuv_link.artifacts import (
    export_final_artifacts,
    fault_recovery_export_record,
    validate_artifact_paths,
)
from multiuuv_link.benchmarks import (
    hybrid_authority_summary,
    paired_classical_learned_summary_42,
    run_classical_learned_benchmark_42,
    run_paired_controller_benchmark,
    summarize_controller_benchmark,
    time_to_coverage_41,
)
from multiuuv_link.communication import deliver_due_failure_aware_33
from multiuuv_link.fault_recovery import (
    choose_orphan_successor,
    deliver_heartbeats,
    majority_detectors,
    queue_heartbeat,
    orphan_goal_safe_from_failed_uuv_36,
    update_failure_suspicion,
)
from multiuuv_link.memory import ResourceMemory
from multiuuv_link.missions import (
    create_checkpoint,
    get_swarm_state_signature,
    restore_checkpoint,
    run_independent_random_rollout,
    run_safety_aware_frontier_exploration,
)
from multiuuv_link.validation import (
    validate_architecture,
    validate_fault_recovery,
    validate_hybrid_benchmark,
    validate_private_knowledge,
)
from multiuuv_link.visualization import plot_agent_knowledge, render_mission_frame, save_gif


def test_checkpoint_is_deep_copy_and_restoration_is_repeatable(small_swarm):
    agents, known_maps = small_swarm
    checkpoint = create_checkpoint(agents, known_maps)
    restored_a, maps_a, _ = restore_checkpoint(checkpoint)
    restored_b, maps_b, _ = restore_checkpoint(checkpoint)
    restored_a[0].state.x += 1
    maps_a[0][:] = False
    assert get_swarm_state_signature(restored_b, maps_b) == get_swarm_state_signature(
        checkpoint["agents"], checkpoint["known_exploration_maps"]
    )


def test_short_paired_benchmark_uses_all_three_controllers(small_swarm, resources, validated_model):
    agents, known_maps = small_swarm
    checkpoint = create_checkpoint(agents, known_maps)
    model, mean, std, _ = validated_model
    frame = run_paired_controller_benchmark(
        checkpoint, resources, model, mean, std, seeds=[420001], steps=2
    )
    assert list(frame["controller"]) == ["Classical", "Learned", "Hybrid"]
    assert (frame["min_separation_px"] >= 128.0).all()
    summary = summarize_controller_benchmark(frame)
    assert list(summary["controller"]) == ["Classical", "Learned", "Hybrid"]
    authority = hybrid_authority_summary(frame)
    assert authority["total_decisions"] >= 0
    ab = run_classical_learned_benchmark_42(
        checkpoint, resources, model, mean, std, seeds=[420001], steps=1
    )
    controller_summary, paired, paired_summary = paired_classical_learned_summary_42(
        ab, seeds=[420001]
    )
    assert list(ab["controller"]) == ["Classical", "Learned"]
    assert list(paired["seed"]) == [420001]
    assert set(controller_summary.index) == {"Classical", "Learned"}
    assert set(paired_summary) == {
        "mean_final_coverage_diff_pp", "mean_coverage_auc_diff",
        "mean_redundancy_diff", "median_redundancy_diff",
        "learned_lower_redundancy_seeds", "learned_higher_coverage_auc_seeds",
        "learned_higher_final_coverage_seeds",
    }


def test_time_to_coverage_first_crossing():
    assert time_to_coverage_41([0.1, 0.4, 0.6, 0.8], 0.6) == 2
    assert time_to_coverage_41([0.1, 0.2], 0.6) is None


def test_heartbeat_detection_and_majority_are_monotonic():
    rng = np.random.default_rng(7)
    queue = []
    message_id, queued = queue_heartbeat(
        queue, 1, 0, 0, rng, 0, loss_probability=0.0, min_delay=1, max_delay=1
    )
    assert queued and message_id == 1
    queue, fresh = deliver_heartbeats(queue, 1, {0: False, 1: False, 2: False})
    suspicion = {0: {1: 0, 2: 0}, 2: {0: 0, 1: 0}}
    declared = {0: set(), 2: set()}
    detection = {0: {}, 2: {}}
    update_failure_suspicion([0, 2], fresh, suspicion, declared, detection, 2, 2)
    assert 1 not in declared[0]
    update_failure_suspicion([0, 2], set(), suspicion, declared, detection, 3, 2)
    update_failure_suspicion([0, 2], set(), suspicion, declared, detection, 4, 2)
    assert 1 in declared[0] and 1 in declared[2]
    assert majority_detectors([0, 2], declared, 1) == [0, 2]


def test_failed_receiver_drops_due_packets_without_knowledge_gain():
    class Channel:
        def __init__(self):
            self.queue = [{
                "message_id": 0, "sender": 0, "receiver": 1, "message_type": "map",
                "payload": np.ones((2, 2), bool), "send_time": 0, "arrival_time": 1,
            }]
            self.delivery_records = []
            self.delivered_packets = 0

    channel = Channel()
    maps = {0: np.zeros((2, 2), bool), 1: np.zeros((2, 2), bool)}
    memories = {0: ResourceMemory(0), 1: ResourceMemory(1)}
    delivered, discarded = deliver_due_failure_aware_33(
        channel, 1, maps, memories, {0: False, 1: True}
    )
    assert delivered == [] and discarded == 1
    assert not maps[1].any()


def test_failed_uuv_is_excluded_from_orphan_successor(small_swarm):
    agents, known_maps = small_swarm
    successor, scores = choose_orphan_successor(
        agents, {"x": 480, "y": 480}, {0: False, 1: True, 2: False, 3: False}, known_maps
    )
    assert successor != 1
    assert all(record["uuv_id"] != 1 for record in scores)
    assert orphan_goal_safe_from_failed_uuv_36({"x": 260, "y": 100}, (100, 100))
    assert not orphan_goal_safe_from_failed_uuv_36({"x": 259, "y": 100}, (100, 100))


def test_validated_fault_artifact_satisfies_recovery_invariants():
    source = Path("artifacts/multiuuv_final_summary_prototype.json")
    fault = json.loads(source.read_text(encoding="utf-8"))["fault_recovery_showcase"]
    metrics = {
        "majority_confirmation_time": fault["majority_confirmation_time"],
        "physical_failure_time": fault["physical_failure_time"],
        "orphan_task": {"x": fault["orphan_goal_x"], "y": fault["orphan_goal_y"]},
        "orphan_successor": fault["recovery_uuv"],
        "failed_uuv": fault["failed_uuv"],
        "orphan_completed": fault["completion_time"] is not None,
        "failed_displacement_px": fault["failed_displacement_px"],
        "minimum_separation_px": fault["minimum_separation_px"],
        "resources_discovered": fault["resources_discovered"],
        "resources_total": fault["resources_total"],
        "healthy_motion_steps_after_failure": 1,
    }
    checks, valid = validate_fault_recovery(metrics)
    assert valid and all(checks.values())


def test_validated_controller_artifact_preserves_all_hybrid_winners():
    source = Path("artifacts/multiuuv_final_summary_prototype.json")
    controllers = json.loads(source.read_text(encoding="utf-8"))["controllers"]
    summary = pd.DataFrame(controllers).rename(columns={
        "final_coverage_mean_pct": "final_coverage_mean",
        "final_coverage_std_pct": "final_coverage_std",
    })
    checks, valid = validate_hybrid_benchmark(summary)
    assert valid and all(checks.values())


def test_fault_runtime_to_artifact_schema_conversion():
    runtime = {
        "mission_steps": 260, "failed_uuv": 1, "physical_failure_time": 70,
        "majority_confirmation_time": 169, "detection_delay": 99,
        "orphan_source": "generated_safe_frontier", "orphan_task": {"x": 421, "y": 261},
        "orphan_successor": 0, "completion_time": 192, "total_recovery_delay": 122,
        "failed_displacement_px": 0.0, "final_coverage": 0.968154462114506,
        "resources_discovered": 10, "resources_total": 10, "safety_overrides": 25,
        "minimum_separation_px": 131.51805959639154,
    }
    record = fault_recovery_export_record(runtime)
    assert record["orphan_goal_x"] == 421 and record["recovery_uuv"] == 0
    assert np.isclose(record["final_coverage_pct"], 96.8154462114506)


def test_early_and_safety_mission_phases_preserve_independent_state(resources):
    from multiuuv_link.missions import create_swarm

    agents, _ = create_swarm(
        resources, np.random.default_rng(8), width=512, height=512, min_separation=128
    )
    rollout = run_independent_random_rollout(
        agents, resources, steps=2, rng=np.random.default_rng(9)
    )
    assert all(report["trajectory_points"] == 3 for report in rollout["reports"])
    safety = run_safety_aware_frontier_exploration(agents, resources, steps=2)
    assert safety["minimum_separation_px"] >= 128.0


def test_artifact_export_schema_writes_only_requested_directory(tmp_path, validated_model):
    model, mean, std, _ = validated_model
    summary = pd.DataFrame([
        {"controller": name, "final_coverage_mean": 90.0, "final_coverage_std": 1.0,
         "coverage_auc_mean": 100.0, "redundancy_mean": 0.2, "redundancy_std": 0.01,
         "hold_mean": 2.0, "safety_override_mean": 1.0, "goal_changes_mean": 3.0,
         "min_separation_mean": 128.0}
        for name in ("Classical", "Learned", "Hybrid")
    ])
    metrics = {"MSE": 0.0, "MAE": 0.0, "R2": 1.0, "Pearson": 1.0, "Spearman": 1.0}
    paths = export_final_artifacts(
        tmp_path, summary, metrics, 1.0, model, mean, std,
        {"total_decisions": 0},
        {"failed_uuv": 1},
    )
    assert validate_artifact_paths(paths.values())
    payload = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert payload["architecture"]["safe_separation_pixels"] == 128.0
    assert payload["learned_model"]["model_parameters"] == 1313


def test_visualization_is_read_only_and_gif_export_is_explicit(tmp_path, small_swarm, resources):
    agents, known_maps = small_swarm
    before = get_swarm_state_signature(agents, known_maps)
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    figure, axis = plt.subplots()
    plot_agent_knowledge(image, agents[0], ax=axis)
    plt.close(figure)
    graph = nx.Graph()
    graph.add_nodes_from(a.uuv_id for a in agents)
    frame = render_mission_frame(
        image, agents, 0, graph, {a.uuv_id: None for a in agents}
    )
    assert frame.ndim == 3 and frame.shape[2] == 3
    gif = save_gif([frame], tmp_path / "frame.gif", fps=1)
    assert gif.exists() and gif.stat().st_size > 0
    assert get_swarm_state_signature(agents, known_maps) == before
    assert validate_private_knowledge(agents, known_maps)
    checks, valid = validate_architecture(agents)
    assert valid and all(checks.values())
