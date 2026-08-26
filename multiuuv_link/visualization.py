"""Read-only plotting and GIF helpers; inputs are never mutated."""

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from .config import FOV_HEIGHT, FOV_WIDTH, RESOURCE_CLASSES

MARKER_BY_CLASS = {"A": "o", "B": "^", "C": "s"}


def plot_environment(seabed_rgb, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(seabed_rgb)
    ax.set_xlabel("x [pixels]")
    ax.set_ylabel("y [pixels]")
    return ax


def plot_agent_knowledge(seabed_rgb, agent, frontier_mask=None, ax=None):
    ax = plot_environment(seabed_rgb, ax)
    observed = agent.exploration_memory.observed
    overlay = np.zeros((*observed.shape, 4), dtype=np.float32)
    overlay[~observed, 3] = 0.58
    ax.imshow(overlay)
    if frontier_mask is not None:
        y, x = np.where(frontier_mask)
        ax.scatter(x, y, s=3, marker=".", label="Frontier")
    for resource_class in RESOURCE_CLASSES:
        resources = [
            resource
            for resource in agent.resource_memory.all_resources()
            if resource.resource_class == resource_class
        ]
        if resources:
            ax.scatter(
                [resource.x for resource in resources],
                [resource.y for resource in resources],
                marker=MARKER_BY_CLASS[resource_class],
                s=120,
                facecolors="white",
                edgecolors="black",
                label=f"Known Type {resource_class}",
            )
    ax.scatter(agent.state.x, agent.state.y, marker="x", s=130, label=f"UUV-{agent.uuv_id}")
    return ax


def plot_trajectories(seabed_rgb, agents, ax=None):
    ax = plot_environment(seabed_rgb, ax)
    for agent in agents:
        path = np.asarray(agent.trajectory)
        if path.size:
            ax.plot(path[:, 0], path[:, 1], label=f"UUV-{agent.uuv_id}")
    return ax


def render_mission_frame(
    seabed_rgb,
    agents,
    timestep,
    operational_graph=None,
    goals=None,
    failed_flags=None,
    title="MultiUUV-Link Mission",
):
    """Render immutable state snapshots to an RGB array."""
    figure, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(seabed_rgb)
    by_id = {agent.uuv_id: agent for agent in agents}
    if operational_graph is not None:
        for uuv_a, uuv_b in operational_graph.edges():
            a, b = by_id[uuv_a].state, by_id[uuv_b].state
            ax.plot([a.x, b.x], [a.y, b.y], "w--", alpha=0.45)
    for agent in agents:
        path = np.asarray(agent.trajectory)
        if path.size:
            ax.plot(path[:, 0], path[:, 1], linewidth=1.5)
        failed = failed_flags is not None and failed_flags.get(agent.uuv_id, False)
        ax.scatter(agent.state.x, agent.state.y, marker="X" if failed else "x", s=130)
        if goals is not None and goals.get(agent.uuv_id) is not None:
            goal = goals[agent.uuv_id]
            ax.scatter(goal["x"], goal["y"], marker="*", s=170)
    ax.set_title(f"{title} - t={timestep}")
    ax.set_xlabel("x [pixels]")
    ax.set_ylabel("y [pixels]")
    figure.tight_layout()
    figure.canvas.draw()
    frame = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
    plt.close(figure)
    return frame


def save_gif(frames, path, fps=8):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, loop=0)
    return path


def fov_rectangle(position):
    return (
        position[0] - FOV_WIDTH // 2,
        position[1] - FOV_HEIGHT // 2,
        FOV_WIDTH,
        FOV_HEIGHT,
    )
