import networkx as nx
import numpy as np

from multiuuv_link.allocation import (
    candidate_fov_iou,
    deduplicate_frontier_tasks,
    perform_stable_allocation_round,
)
from multiuuv_link.communication import (
    DegradedCommunicationChannel,
    LinkBlackoutManager,
    build_communication_graph,
    exchange_map_knowledge_one_hop,
    exchange_resource_knowledge_one_hop,
    packet_delivered,
    sample_message_delay,
)
from multiuuv_link.config import MIN_ASSIGNED_GOAL_SEPARATION
from multiuuv_link.memory import ExplorationMemory, ResourceMemory
from multiuuv_link.models import Resource, UUVAgent, UUVState


def _agent(uuv_id, x, resource=None):
    agent = UUVAgent(
        uuv_id,
        UUVState(uuv_id, x, 100),
        ExplorationMemory(uuv_id, 32, 32),
        ResourceMemory(uuv_id),
        [(x, 100)],
    )
    if resource is not None:
        agent.resource_memory.update([resource])
    return agent


def test_communication_graph_is_range_limited_and_includes_isolated_nodes():
    graph = build_communication_graph({0: (0, 0), 1: (400, 0), 2: (801, 0)})
    assert set(graph.nodes()) == {0, 1, 2}
    assert set(graph.edges()) == {(0, 1)}
    assert graph[0][1]["distance"] == 400.0


def test_one_hop_map_exchange_has_no_same_round_multi_hop():
    graph = nx.Graph([(0, 1), (1, 2)])
    maps = {uuv_id: np.zeros((3, 3), dtype=bool) for uuv_id in range(3)}
    maps[0][0, 0] = maps[1][1, 1] = maps[2][2, 2] = True
    local_snapshots = {key: value.copy() for key, value in maps.items()}
    exchange_map_knowledge_one_hop(graph, maps)
    assert maps[0][1, 1] and not maps[0][2, 2]
    assert maps[1][0, 0] and maps[1][2, 2]
    assert maps[2][1, 1] and not maps[2][0, 0]
    assert all(local_snapshots[key].sum() == 1 for key in local_snapshots)


def test_one_hop_resource_exchange_uses_pre_round_snapshots():
    r0, r1, r2 = Resource(0, 0, 0, "A"), Resource(1, 1, 1, "B"), Resource(2, 2, 2, "C")
    agents = [_agent(0, 0, r0), _agent(1, 100, r1), _agent(2, 200, r2)]
    exchange_resource_knowledge_one_hop(nx.Graph([(0, 1), (1, 2)]), agents)
    assert agents[0].resource_memory.knows(1)
    assert not agents[0].resource_memory.knows(2)
    assert agents[1].resource_memory.knows(0) and agents[1].resource_memory.knows(2)
    assert agents[2].resource_memory.knows(1)
    assert not agents[2].resource_memory.knows(0)


def test_packet_loss_and_delay_match_configured_distributions():
    rng = np.random.default_rng(123)
    delivered = np.mean([packet_delivered(rng, 0.25) for _ in range(10_000)])
    assert abs(delivered - 0.75) < 0.03
    rng = np.random.default_rng(456)
    delays = np.asarray([sample_message_delay(rng, 1, 5) for _ in range(10_000)])
    assert delays.min() == 1 and delays.max() == 5
    assert abs(delays.mean() - 3.0) < 0.10


def test_blackout_suppresses_exact_forced_duration_then_recovers():
    manager = LinkBlackoutManager([0, 1], start_probability=0.0, rng=np.random.default_rng(1))
    graph = nx.Graph([(0, 1)])
    manager.force_blackout(0, 1, 3)
    edge_counts = [manager.step(graph)["operational_graph"].number_of_edges() for _ in range(4)]
    assert edge_counts == [0, 0, 0, 1]
    assert manager.total_blackout_starts == 1
    assert manager.total_recoveries == 1


def test_delayed_channel_snapshots_payload_and_delivers_at_arrival():
    channel = DegradedCommunicationChannel(
        [0, 1], packet_loss_probability=0.0, min_delay=1, max_delay=1,
        blackout_start_probability=0.0, rng=np.random.default_rng(2)
    )
    maps = {0: np.zeros((4, 4), bool), 1: np.zeros((4, 4), bool)}
    maps[0][0, 0] = True
    memories = {0: ResourceMemory(0), 1: ResourceMemory(1)}
    channel.transmit_round(0, {0: (0, 0), 1: (100, 0)}, maps, memories)
    maps[0][0, 0] = False
    assert channel.deliver_due(0, maps, memories) == []
    delivered = channel.deliver_due(1, maps, memories)
    assert len(delivered) == 4
    assert maps[1][0, 0]
    assert channel.geometric_packet_opportunities == 4
    assert channel.attempted_packets == channel.queued_packets == 4
    assert channel.dropped_packets == 0
    assert channel.delivered_packets == 4


def test_queued_message_delivery_is_stale_and_independent_of_later_topology():
    channel = DegradedCommunicationChannel(
        [0, 1], packet_loss_probability=0.0, min_delay=2, max_delay=2,
        blackout_start_probability=0.0, rng=np.random.default_rng(3)
    )
    maps = {0: np.zeros((2, 2), bool), 1: np.zeros((2, 2), bool)}
    maps[0][0, 0] = True
    memories = {0: ResourceMemory(0), 1: ResourceMemory(1)}
    channel.transmit_round(0, {0: (0, 0), 1: (100, 0)}, maps, memories)
    assert channel.deliver_due(1, maps, memories) == []
    # Delivery does not re-check that the sender and receiver are still connected.
    delivered = channel.deliver_due(2, maps, memories)
    assert len(delivered) == 4 and maps[1][0, 0]


def test_task_deduplication_is_stable_first_seen_and_iou_is_bounded():
    tasks = [
        {"x": 100, "y": 100, "source_uuv": 0},
        {"x": 120, "y": 100, "source_uuv": 1},
        {"x": 300, "y": 100, "source_uuv": 2},
    ]
    retained = deduplicate_frontier_tasks(tasks, min_distance=128)
    assert [(task["x"], task["y"]) for task in retained] == [(100, 100), (300, 100)]
    assert [task["task_id"] for task in retained] == [0, 1]
    assert candidate_fov_iou(tasks[0], tasks[0]) == 1.0
    assert 0.0 <= candidate_fov_iou(tasks[0], tasks[2]) <= 1.0


def test_component_allocation_preserves_minimum_goal_spacing(small_swarm):
    agents, known_maps = small_swarm
    graph = build_communication_graph({a.uuv_id: (a.state.x, a.state.y) for a in agents})
    assignments, _, spacing, overlap, stats = perform_stable_allocation_round(
        graph, {a.uuv_id: None for a in agents}, agents, known_maps
    )
    assert set(assignments) == {0, 1, 2, 3}
    assert all(distance >= MIN_ASSIGNED_GOAL_SEPARATION for distance in spacing)
    assert all(0.0 <= value <= 1.0 for value in overlap)
    assert stats
