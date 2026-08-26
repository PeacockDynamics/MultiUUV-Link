"""Static seabed ground truth and hidden-resource scenario construction."""

from pathlib import Path

import cv2
import numpy as np

from .config import (
    CHANNELS,
    HEIGHT,
    RESOURCE_CLASSES,
    RESOURCE_COUNT_MAX,
    RESOURCE_COUNT_MIN,
    SEABED_IMAGE_PATH,
    WIDTH,
)
from .models import Resource


def load_seabed(path: str | Path = SEABED_IMAGE_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Load validated BGR simulator ground truth and a distinct RGB display copy."""
    seabed_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if seabed_bgr is None:
        raise FileNotFoundError(f"OpenCV could not load the seabed image:\n{path}")
    if seabed_bgr.ndim != 3:
        raise ValueError(f"Expected a 3D color image array, got shape {seabed_bgr.shape}")
    height, width, channels = seabed_bgr.shape
    if channels != CHANNELS:
        raise ValueError(f"Expected 3 color channels, got {channels}")
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image dimensions: {width} x {height}")
    if (width, height) != (WIDTH, HEIGHT):
        raise ValueError(
            f"Authoritative seabed dimensions changed: {(width, height)} != {(WIDTH, HEIGHT)}"
        )
    seabed_rgb = cv2.cvtColor(seabed_bgr, cv2.COLOR_BGR2RGB)
    return seabed_bgr, seabed_rgb


def generate_resources(
    width: int,
    height: int,
    min_resources: int = RESOURCE_COUNT_MIN,
    max_resources: int = RESOURCE_COUNT_MAX,
    classes=RESOURCE_CLASSES,
    rng=None,
):
    """Generate immutable hidden resources using notebook Combo 6 RNG ordering."""
    if rng is None:
        rng = np.random.default_rng()
    if min_resources <= 0:
        raise ValueError("min_resources must be positive.")
    if max_resources < min_resources:
        raise ValueError("max_resources must be greater than or equal to min_resources.")
    if max_resources > width * height:
        raise ValueError("Requested more possible resources than available unique pixels.")

    num_resources = int(rng.integers(min_resources, max_resources + 1))
    resources: list[Resource] = []
    used_positions: set[tuple[int, int]] = set()
    while len(resources) < num_resources:
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        position = (x, y)
        if position in used_positions:
            continue
        resource_class = str(rng.choice(classes))
        resources.append(Resource(len(resources), x, y, resource_class))
        used_positions.add(position)
    return resources


def validate_resources(
    resources,
    width: int = WIDTH,
    height: int = HEIGHT,
    classes=RESOURCE_CLASSES,
    min_resources: int = RESOURCE_COUNT_MIN,
    max_resources: int = RESOURCE_COUNT_MAX,
) -> bool:
    if not min_resources <= len(resources) <= max_resources:
        raise RuntimeError("Generated resource count is outside the configured range.")
    positions = [(resource.x, resource.y) for resource in resources]
    if len(set(positions)) != len(positions):
        raise RuntimeError("Duplicate resource coordinates detected.")
    for resource in resources:
        if not 0 <= resource.x < width or not 0 <= resource.y < height:
            raise RuntimeError(f"Resource {resource.resource_id} is out of bounds.")
        if resource.resource_class not in classes:
            raise RuntimeError(f"Resource {resource.resource_id} has an invalid class.")
    return True
