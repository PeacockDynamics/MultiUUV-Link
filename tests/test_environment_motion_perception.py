from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from multiuuv_link.config import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    HEIGHT,
    WIDTH,
)
from multiuuv_link.environment import generate_resources, load_seabed, validate_resources
from multiuuv_link.models import Resource, UUVState, generate_safe_spawn_positions
from multiuuv_link.motion import (
    choose_greedy_action,
    move_uuv,
    move_uuv_scaled,
    trajectory_path_length,
)
from multiuuv_link.perception import (
    discover_visible_resources,
    get_fov_bounds,
    get_observation,
    resource_in_fov,
)


def test_authoritative_seabed_dimensions_and_color_copy():
    bgr, rgb = load_seabed()
    assert bgr.shape == (HEIGHT, WIDTH, 3)
    assert rgb.shape == bgr.shape
    assert not np.shares_memory(bgr, rgb)
    assert np.array_equal(rgb[..., 0], bgr[..., 2])
    assert np.array_equal(rgb[..., 1], bgr[..., 1])
    assert np.array_equal(rgb[..., 2], bgr[..., 0])


def test_resource_generation_is_seeded_unique_bounded_and_immutable():
    first = generate_resources(WIDTH, HEIGHT, rng=np.random.default_rng(42))
    second = generate_resources(WIDTH, HEIGHT, rng=np.random.default_rng(42))
    assert first == second
    assert validate_resources(first)
    assert 10 <= len(first) <= 25
    assert len({(r.x, r.y) for r in first}) == len(first)
    with pytest.raises(FrozenInstanceError):
        first[0].x = 0


def test_discrete_motion_clipping_ordering_and_tie_behavior():
    state = UUVState(3, 10, 10)
    assert move_uuv(state, ACTION_UP) == UUVState(3, 10, 2)
    assert move_uuv(state, ACTION_RIGHT) == UUVState(3, 18, 10)
    assert move_uuv(state, ACTION_DOWN) == UUVState(3, 10, 18)
    assert move_uuv(state, ACTION_LEFT) == UUVState(3, 2, 10)
    assert move_uuv_scaled(UUVState(3, 1, 1), ACTION_LEFT, 32) == UUVState(3, 0, 1)
    assert choose_greedy_action(state, 20, 20) == ACTION_RIGHT
    assert choose_greedy_action(state, 0, 0) == ACTION_LEFT
    assert choose_greedy_action(state, 10, 10) is None
    with pytest.raises(ValueError):
        move_uuv(state, 99)


def test_trajectory_path_length():
    assert trajectory_path_length([]) == 0.0
    assert trajectory_path_length([(0, 0), (3, 4), (6, 8)]) == 10.0


def test_safe_spawning_reproducibility_and_separation():
    positions_a, attempts_a = generate_safe_spawn_positions(
        4, WIDTH, HEIGHT, 256.0, np.random.default_rng(42)
    )
    positions_b, attempts_b = generate_safe_spawn_positions(
        4, WIDTH, HEIGHT, 256.0, np.random.default_rng(42)
    )
    assert positions_a == positions_b
    assert attempts_a == attempts_b
    for index, first in enumerate(positions_a):
        for second in positions_a[index + 1 :]:
            assert np.hypot(first[0] - second[0], first[1] - second[1]) >= 256.0


def test_fov_is_fixed_size_padded_and_uses_yx_array_indexing():
    image = np.arange(10 * 12 * 3, dtype=np.int32).reshape(10, 12, 3)
    state = UUVState(0, 0, 0)
    observation, metadata = get_observation(
        state, image, fov_width=4, fov_height=4, world_width=12, world_height=10
    )
    assert observation.shape == (4, 4, 3)
    assert metadata["nominal_bounds"] == (-2, -2, 2, 2)
    assert metadata["valid_pixels"] == 4
    assert metadata["padded_pixels"] == 12
    assert np.array_equal(observation[2:4, 2:4], image[0:2, 0:2])
    assert not observation[:2].any()


def test_full_size_corner_and_center_fov_counts():
    image = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8)
    corner, corner_meta = get_observation(UUVState(0, 0, 0), image)
    center, center_meta = get_observation(UUVState(0, WIDTH // 2, HEIGHT // 2), image)
    assert corner.shape == center.shape == (128, 128, 3)
    assert (corner_meta["valid_pixels"], corner_meta["padded_pixels"]) == (4096, 12288)
    assert (center_meta["valid_pixels"], center_meta["padded_pixels"]) == (16384, 0)


def test_resource_fov_half_open_boundaries():
    state = UUVState(0, 100, 100)
    assert get_fov_bounds(state) == (36, 36, 164, 164)
    inside = Resource(1, 36, 36, "A")
    outside = Resource(2, 164, 100, "B")
    assert resource_in_fov(inside, state)
    assert not resource_in_fov(outside, state)
    assert discover_visible_resources(state, [inside, outside]) == [inside]
