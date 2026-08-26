"""Local FOV geometry, padded observations, and resource discovery."""

import numpy as np

from .config import FOV_HEIGHT, FOV_WIDTH, HEIGHT, WIDTH
from .models import Resource, UUVState


def get_fov_bounds(
    state: UUVState,
    fov_width: int = FOV_WIDTH,
    fov_height: int = FOV_HEIGHT,
):
    half_w = fov_width // 2
    half_h = fov_height // 2
    x_min = state.x - half_w
    y_min = state.y - half_h
    return x_min, y_min, x_min + fov_width, y_min + fov_height


def get_observation(
    state: UUVState,
    image: np.ndarray,
    fov_width: int = FOV_WIDTH,
    fov_height: int = FOV_HEIGHT,
    world_width: int = WIDTH,
    world_height: int = HEIGHT,
):
    x_min, y_min, x_max, y_max = get_fov_bounds(state, fov_width, fov_height)
    observation = np.zeros(
        (fov_height, fov_width, image.shape[2]), dtype=image.dtype
    )
    src_x0, src_y0 = max(0, x_min), max(0, y_min)
    src_x1, src_y1 = min(world_width, x_max), min(world_height, y_max)
    dst_x0, dst_y0 = src_x0 - x_min, src_y0 - y_min
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    observation[dst_y0:dst_y1, dst_x0:dst_x1] = image[
        src_y0:src_y1, src_x0:src_x1
    ]
    valid_width = max(0, src_x1 - src_x0)
    valid_height = max(0, src_y1 - src_y0)
    valid_pixels = valid_width * valid_height
    total_pixels = fov_width * fov_height
    padded_pixels = total_pixels - valid_pixels
    return observation, {
        "nominal_bounds": (x_min, y_min, x_max, y_max),
        "source_bounds": (src_x0, src_y0, src_x1, src_y1),
        "valid_pixels": valid_pixels,
        "padded_pixels": padded_pixels,
        "padding_fraction": padded_pixels / total_pixels,
    }


def resource_in_fov(
    resource: Resource,
    state: UUVState,
    fov_width: int = FOV_WIDTH,
    fov_height: int = FOV_HEIGHT,
) -> bool:
    x_min, y_min, x_max, y_max = get_fov_bounds(state, fov_width, fov_height)
    return x_min <= resource.x < x_max and y_min <= resource.y < y_max


def discover_visible_resources(
    state: UUVState,
    resources,
    fov_width: int = FOV_WIDTH,
    fov_height: int = FOV_HEIGHT,
):
    return [
        resource
        for resource in resources
        if resource_in_fov(resource, state, fov_width, fov_height)
    ]
