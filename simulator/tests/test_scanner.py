"""Tests for simulator.scanner: ScannerConfig and ray sweep generation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simulator.scanner import ScannerConfig, generate_rays, generate_rays_multi


class TestScannerConfig:
    def test_defaults(self):
        config = ScannerConfig(height=1.0)
        assert config.height == 1.0
        assert config.num_rays == 360
        assert config.angular_range == (0.0, 2 * np.pi)

    def test_rejects_non_positive_num_rays(self):
        with pytest.raises(ValueError):
            ScannerConfig(height=1.0, num_rays=0)
        with pytest.raises(ValueError):
            ScannerConfig(height=1.0, num_rays=-10)

    def test_rejects_degenerate_angular_range(self):
        with pytest.raises(ValueError):
            ScannerConfig(height=1.0, angular_range=(1.0, 1.0))
        with pytest.raises(ValueError):
            ScannerConfig(height=1.0, angular_range=(1.0, 0.5))

    def test_ray_angles_shape_and_spacing(self):
        config = ScannerConfig(height=1.0, num_rays=360)
        angles = config.ray_angles()
        assert angles.shape == (360,)
        np.testing.assert_allclose(float(angles[0]), 0.0, atol=1e-6)
        np.testing.assert_allclose(float(angles[1] - angles[0]), np.pi / 180, atol=1e-6)

    def test_ray_angles_excludes_endpoint_on_full_sweep(self):
        config = ScannerConfig(height=1.0, num_rays=4)
        angles = config.ray_angles()
        np.testing.assert_allclose(
            np.asarray(angles), [0.0, np.pi / 2, np.pi, 3 * np.pi / 2], atol=1e-6
        )

    def test_custom_partial_sweep(self):
        config = ScannerConfig(height=0.3, angular_range=(0.0, np.pi), num_rays=2)
        angles = config.ray_angles()
        np.testing.assert_allclose(np.asarray(angles), [0.0, np.pi / 2], atol=1e-6)


class TestGenerateRays:
    def test_output_shapes(self):
        config = ScannerConfig(height=1.0, num_rays=8)
        origins, directions = generate_rays(jnp.array([2.0, 3.0]), config)
        assert origins.shape == (8, 2)
        assert directions.shape == (8, 2)

    def test_origins_equal_scanner_position(self):
        config = ScannerConfig(height=1.0, num_rays=8)
        position = jnp.array([2.0, 3.0])
        origins, _ = generate_rays(position, config)
        np.testing.assert_allclose(np.asarray(origins), np.tile(np.asarray(position), (8, 1)))

    def test_directions_are_unit_vectors(self):
        config = ScannerConfig(height=1.0, num_rays=16)
        _, directions = generate_rays(jnp.array([0.0, 0.0]), config)
        norms = jnp.linalg.norm(directions, axis=-1)
        np.testing.assert_allclose(np.asarray(norms), np.ones(16), atol=1e-6)

    def test_directions_match_ray_angles(self):
        config = ScannerConfig(height=1.0, num_rays=4)
        _, directions = generate_rays(jnp.array([0.0, 0.0]), config)
        expected = jnp.stack([jnp.cos(config.ray_angles()), jnp.sin(config.ray_angles())], axis=-1)
        np.testing.assert_allclose(np.asarray(directions), np.asarray(expected), atol=1e-6)

    def test_jit_compilable(self):
        config = ScannerConfig(height=1.0, num_rays=16)
        jitted = jax.jit(lambda pos: generate_rays(pos, config))
        origins, directions = jitted(jnp.array([1.0, 1.0]))
        assert origins.shape == (16, 2)
        assert directions.shape == (16, 2)


class TestGenerateRaysMulti:
    def test_output_shapes(self):
        config = ScannerConfig(height=1.0, num_rays=8)
        positions = jnp.array([[0.0, 0.0], [5.0, 5.0], [2.0, -1.0]])
        origins, directions = generate_rays_multi(positions, config)
        assert origins.shape == (3, 8, 2)
        assert directions.shape == (3, 8, 2)

    def test_matches_single_scanner_generate_rays(self):
        config = ScannerConfig(height=1.0, num_rays=12)
        positions = jnp.array([[0.0, 0.0], [5.0, 5.0], [2.0, -1.0]])
        origins, directions = generate_rays_multi(positions, config)
        for i, position in enumerate(positions):
            expected_origins, expected_directions = generate_rays(position, config)
            np.testing.assert_allclose(
                np.asarray(origins[i]), np.asarray(expected_origins), atol=1e-6
            )
            np.testing.assert_allclose(
                np.asarray(directions[i]), np.asarray(expected_directions), atol=1e-6
            )

    def test_vmap_is_pure_no_python_loop_shape_mismatch(self):
        config = ScannerConfig(height=1.0, num_rays=5)
        positions = jnp.zeros((7, 2))
        origins, directions = generate_rays_multi(positions, config)
        assert origins.shape == (7, 5, 2)
        assert directions.shape == (7, 5, 2)

    def test_jit_compilable(self):
        config = ScannerConfig(height=1.0, num_rays=8)
        jitted = jax.jit(lambda pos: generate_rays_multi(pos, config))
        positions = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        origins, directions = jitted(positions)
        assert origins.shape == (2, 8, 2)
        assert directions.shape == (2, 8, 2)
