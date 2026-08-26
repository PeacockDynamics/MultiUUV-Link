"""Range-limited one-hop communication and the validated degraded channel."""

from copy import deepcopy
from itertools import combinations

import networkx as nx
import numpy as np

from .config import (
    BLACKOUT_MAX_DURATION,
    BLACKOUT_MIN_DURATION,
    BLACKOUT_START_PROBABILITY,
    COMMUNICATION_RANGE,
    MAX_COMM_DELAY,
    MIN_COMM_DELAY,
    PACKET_LOSS_PROBABILITY,
)
from .motion import euclidean_distance


def build_communication_graph(
    positions, communication_range=COMMUNICATION_RANGE
):
    graph = nx.Graph()
    uuv_ids = sorted(positions.keys())
    graph.add_nodes_from(uuv_ids)
    for uuv_a, uuv_b in combinations(uuv_ids, 2):
        distance = euclidean_distance(positions[uuv_a], positions[uuv_b])
        if distance <= communication_range:
            graph.add_edge(uuv_a, uuv_b, distance=float(distance))
    return graph


def exchange_resource_knowledge_one_hop(
    graph, agents, resource_provenance=None
):
    """Apply one directed snapshot message in each direction per edge."""
    agent_by_id = {agent.uuv_id: agent for agent in agents}
    snapshots = {
        agent.uuv_id: {
            resource.resource_id: resource
            for resource in agent.resource_memory.all_resources()
        }
        for agent in agents
    }
    messages = []
    for uuv_a, uuv_b in graph.edges():
        messages.append((uuv_a, uuv_b, list(snapshots[uuv_a].values())))
        messages.append((uuv_b, uuv_a, list(snapshots[uuv_b].values())))
    gains = {agent.uuv_id: 0 for agent in agents}
    for sender, receiver, payload in messages:
        memory = agent_by_id[receiver].resource_memory
        before = len(memory)
        newly_learned = memory.update(payload)
        if resource_provenance is not None:
            for resource in newly_learned:
                resource_provenance[receiver].setdefault(resource.resource_id, "REMOTE")
        gains[receiver] += len(memory) - before
    return gains


def exchange_map_knowledge_one_hop(graph, known_maps):
    """Logical-OR pre-round snapshots; received data is never relayed this round."""
    snapshots = {uuv_id: known_map.copy() for uuv_id, known_map in known_maps.items()}
    messages = []
    for uuv_a, uuv_b in graph.edges():
        messages.append((uuv_a, uuv_b, snapshots[uuv_a].copy()))
        messages.append((uuv_b, uuv_a, snapshots[uuv_b].copy()))
    gains = {uuv_id: 0 for uuv_id in known_maps}
    for sender, receiver, payload in messages:
        before = int(known_maps[receiver].sum())
        known_maps[receiver] = np.logical_or(known_maps[receiver], payload)
        gains[receiver] += int(known_maps[receiver].sum()) - before
    return gains


def packet_delivered(rng, loss_probability=PACKET_LOSS_PROBABILITY):
    if not 0.0 <= loss_probability <= 1.0:
        raise ValueError("loss_probability must lie inside [0, 1].")
    return bool(rng.random() >= loss_probability)


def sample_message_delay(
    rng, min_delay=MIN_COMM_DELAY, max_delay=MAX_COMM_DELAY
):
    if min_delay < 0:
        raise ValueError("min_delay must be non-negative.")
    if max_delay < min_delay:
        raise ValueError("max_delay must be >= min_delay.")
    return int(rng.integers(min_delay, max_delay + 1))


def create_delayed_message(
    sender,
    receiver,
    message_type,
    payload,
    send_time,
    rng,
    loss_probability=PACKET_LOSS_PROBABILITY,
):
    if not packet_delivered(rng, loss_probability):
        return None
    delay = sample_message_delay(rng)
    return {
        "message_id": None,
        "sender": sender,
        "receiver": receiver,
        "message_type": message_type,
        "payload": deepcopy(payload),
        "send_time": int(send_time),
        "delay": int(delay),
        "arrival_time": int(send_time + delay),
    }


def canonical_link(uuv_a, uuv_b):
    return tuple(sorted((int(uuv_a), int(uuv_b))))


class LinkBlackoutManager:
    def __init__(
        self,
        uuv_ids,
        start_probability=BLACKOUT_START_PROBABILITY,
        min_duration=BLACKOUT_MIN_DURATION,
        max_duration=BLACKOUT_MAX_DURATION,
        rng=None,
    ):
        self.uuv_ids = sorted(list(uuv_ids))
        self.start_probability = float(start_probability)
        self.min_duration = int(min_duration)
        self.max_duration = int(max_duration)
        self.rng = np.random.default_rng() if rng is None else rng
        if not 0.0 <= self.start_probability <= 1.0:
            raise ValueError("Blackout probability must lie in [0, 1].")
        if self.min_duration <= 0:
            raise ValueError("Blackout duration must be positive.")
        if self.max_duration < self.min_duration:
            raise ValueError("max_duration must be >= min_duration.")
        self.remaining = {
            canonical_link(uuv_a, uuv_b): 0
            for uuv_a, uuv_b in combinations(self.uuv_ids, 2)
        }
        self.total_blackout_starts = 0
        self.total_recoveries = 0

    def is_operational(self, uuv_a, uuv_b):
        return self.remaining[canonical_link(uuv_a, uuv_b)] == 0

    def remaining_duration(self, uuv_a, uuv_b):
        return self.remaining[canonical_link(uuv_a, uuv_b)]

    def force_blackout(self, uuv_a, uuv_b, duration):
        if duration <= 0:
            raise ValueError("Forced blackout duration must be positive.")
        link = canonical_link(uuv_a, uuv_b)
        if self.remaining[link] == 0:
            self.total_blackout_starts += 1
        self.remaining[link] = int(duration)

    def step(self, geometric_graph, allow_new_blackouts=True):
        active_before = {
            link for link, remaining in self.remaining.items() if remaining > 0
        }
        newly_started = []
        if allow_new_blackouts:
            for uuv_a, uuv_b in geometric_graph.edges():
                link = canonical_link(uuv_a, uuv_b)
                if self.remaining[link] > 0:
                    continue
                if self.rng.random() < self.start_probability:
                    duration = int(
                        self.rng.integers(self.min_duration, self.max_duration + 1)
                    )
                    self.remaining[link] = duration
                    self.total_blackout_starts += 1
                    newly_started.append((link, duration))
        operational_graph = nx.Graph()
        operational_graph.add_nodes_from(geometric_graph.nodes())
        suppressed_links = []
        for uuv_a, uuv_b, data in geometric_graph.edges(data=True):
            link = canonical_link(uuv_a, uuv_b)
            if self.remaining[link] > 0:
                suppressed_links.append(link)
            else:
                operational_graph.add_edge(uuv_a, uuv_b, **deepcopy(data))
        failed_link_count = sum(
            1 for remaining in self.remaining.values() if remaining > 0
        )
        recovered_links = []
        for link in list(self.remaining.keys()):
            if self.remaining[link] <= 0:
                continue
            self.remaining[link] -= 1
            if self.remaining[link] == 0:
                recovered_links.append(link)
                self.total_recoveries += 1
        return {
            "operational_graph": operational_graph,
            "suppressed_links": suppressed_links,
            "new_blackouts": newly_started,
            "recoveries": recovered_links,
            "failed_link_count": failed_link_count,
            "active_before": active_before,
        }


class DegradedCommunicationChannel:
    def __init__(
        self,
        uuv_ids,
        communication_range=COMMUNICATION_RANGE,
        packet_loss_probability=PACKET_LOSS_PROBABILITY,
        min_delay=MIN_COMM_DELAY,
        max_delay=MAX_COMM_DELAY,
        blackout_start_probability=BLACKOUT_START_PROBABILITY,
        blackout_min_duration=BLACKOUT_MIN_DURATION,
        blackout_max_duration=BLACKOUT_MAX_DURATION,
        rng=None,
    ):
        self.uuv_ids = sorted(list(uuv_ids))
        self.communication_range = float(communication_range)
        self.packet_loss_probability = float(packet_loss_probability)
        self.min_delay = int(min_delay)
        self.max_delay = int(max_delay)
        self.rng = np.random.default_rng() if rng is None else rng
        self.blackout_manager = LinkBlackoutManager(
            self.uuv_ids,
            blackout_start_probability,
            blackout_min_duration,
            blackout_max_duration,
            self.rng,
        )
        self.queue = []
        self.next_message_id = 0
        self.geometric_packet_opportunities = 0
        self.blackout_suppressed_packets = 0
        self.attempted_packets = 0
        self.dropped_packets = 0
        self.queued_packets = 0
        self.delivered_packets = 0
        self.delivery_records = []

    @staticmethod
    def snapshot_payload(message_type, sender_id, known_maps, resource_memories):
        if message_type == "map":
            return known_maps[sender_id].copy()
        if message_type == "resource":
            return deepcopy(resource_memories[sender_id].all_resources())
        raise ValueError("Unknown message type.")

    def queue_packet(
        self, sender, receiver, message_type, payload, current_time
    ):
        self.attempted_packets += 1
        if not packet_delivered(self.rng, self.packet_loss_probability):
            self.dropped_packets += 1
            return None
        delay = sample_message_delay(self.rng, self.min_delay, self.max_delay)
        message = {
            "message_id": self.next_message_id,
            "sender": int(sender),
            "receiver": int(receiver),
            "message_type": message_type,
            "payload": deepcopy(payload),
            "send_time": int(current_time),
            "delay": int(delay),
            "arrival_time": int(current_time + delay),
        }
        self.next_message_id += 1
        self.queue.append(message)
        self.queued_packets += 1
        return message

    def transmit_round(self, current_time, positions, known_maps, resource_memories):
        geometric_graph = build_communication_graph(
            positions, self.communication_range
        )
        blackout_result = self.blackout_manager.step(
            geometric_graph, allow_new_blackouts=True
        )
        operational_graph = blackout_result["operational_graph"]
        geometric_opportunities = geometric_graph.number_of_edges() * 4
        operational_opportunities = operational_graph.number_of_edges() * 4
        suppressed = geometric_opportunities - operational_opportunities
        self.geometric_packet_opportunities += geometric_opportunities
        self.blackout_suppressed_packets += suppressed
        for uuv_a, uuv_b in operational_graph.edges():
            for sender, receiver in ((uuv_a, uuv_b), (uuv_b, uuv_a)):
                for message_type in ("map", "resource"):
                    payload = self.snapshot_payload(
                        message_type, sender, known_maps, resource_memories
                    )
                    self.queue_packet(
                        sender, receiver, message_type, payload, current_time
                    )
        return {
            "geometric_graph": geometric_graph,
            "operational_graph": operational_graph,
            "new_blackouts": blackout_result["new_blackouts"],
            "recoveries": blackout_result["recoveries"],
            "suppressed_packets": suppressed,
        }

    def deliver_due(self, current_time, known_maps, resource_memories):
        due = [m for m in self.queue if m["arrival_time"] <= current_time]
        future = [m for m in self.queue if m["arrival_time"] > current_time]
        delivered_now = []
        for message in due:
            receiver = message["receiver"]
            before_map_cells = int(known_maps[receiver].sum())
            before_resource_count = len(resource_memories[receiver])
            if message["message_type"] == "map":
                known_maps[receiver] = np.logical_or(
                    known_maps[receiver], message["payload"]
                )
            elif message["message_type"] == "resource":
                resource_memories[receiver].update(message["payload"])
            else:
                raise RuntimeError("Unknown queued message type.")
            record = {
                "message_id": message["message_id"],
                "message_type": message["message_type"],
                "sender": message["sender"],
                "receiver": receiver,
                "send_time": message["send_time"],
                "arrival_time": message["arrival_time"],
                "delivery_time": int(current_time),
                "age": int(current_time - message["send_time"]),
                "new_map_cells": int(known_maps[receiver].sum())
                - before_map_cells,
                "new_resources": len(resource_memories[receiver])
                - before_resource_count,
            }
            delivered_now.append(record)
            self.delivery_records.append(record)
            self.delivered_packets += 1
        self.queue = future
        return delivered_now


def deliver_due_failure_aware_33(
    channel, current_time, known_maps, resource_memories, failed_flags
):
    due = [m for m in channel.queue if m["arrival_time"] <= current_time]
    future = [m for m in channel.queue if m["arrival_time"] > current_time]
    delivered_now, discarded_now = [], 0
    for message in due:
        receiver = message["receiver"]
        if failed_flags.get(receiver, False):
            discarded_now += 1
            continue
        before_map = int(known_maps[receiver].sum())
        before_resources = len(resource_memories[receiver])
        if message["message_type"] == "map":
            known_maps[receiver] = np.logical_or(
                known_maps[receiver], message["payload"]
            )
        elif message["message_type"] == "resource":
            resource_memories[receiver].update(message["payload"])
        else:
            raise RuntimeError("Unknown message type in Combo 33.")
        record = {
            "message_id": message["message_id"],
            "message_type": message["message_type"],
            "sender": message["sender"],
            "receiver": receiver,
            "send_time": message["send_time"],
            "arrival_time": message["arrival_time"],
            "delivery_time": current_time,
            "age": current_time - message["send_time"],
            "new_map_cells": int(known_maps[receiver].sum()) - before_map,
            "new_resources": len(resource_memories[receiver]) - before_resources,
        }
        channel.delivery_records.append(record)
        channel.delivered_packets += 1
        delivered_now.append(record)
    channel.queue = future
    return delivered_now, discarded_now
