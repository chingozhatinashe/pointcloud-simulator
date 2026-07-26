"""Tests for data.normalisation: coordinate frame normalisation and transform storage."""

from __future__ import annotations

import numpy as np
import pytest

from data.normalisation import (
    _UP_AXIS_ROTATIONS,
    NormalisationTransform,
    normalise,
    rotate_to_z_up,
    to_metres,
    translate_to_origin,
)


class TestToMetres:
    def test_known_scales(self):
        points = np.array([[1.0, 2.0, 3.0]])
        scaled, scale = to_metres(points, "mm")
        assert scale == 0.001
        np.testing.assert_allclose(scaled, [[0.001, 0.002, 0.003]])

    def test_metres_is_identity(self):
        points = np.array([[1.0, 2.0, 3.0]])
        scaled, scale = to_metres(points, "m")
        assert scale == 1.0
        np.testing.assert_allclose(scaled, points)

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError):
            to_metres(np.zeros((1, 3)), "furlongs")


class TestUpAxisRotations:
    @pytest.mark.parametrize("axis", sorted(_UP_AXIS_ROTATIONS))
    def test_is_a_proper_rotation(self, axis):
        rotation = _UP_AXIS_ROTATIONS[axis]
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-10)

    @pytest.mark.parametrize(
        ("axis", "up_vector"),
        [
            ("z", [0.0, 0.0, 1.0]),
            ("-z", [0.0, 0.0, -1.0]),
            ("y", [0.0, 1.0, 0.0]),
            ("-y", [0.0, -1.0, 0.0]),
            ("x", [1.0, 0.0, 0.0]),
            ("-x", [-1.0, 0.0, 0.0]),
        ],
    )
    def test_maps_named_axis_to_positive_z(self, axis, up_vector):
        rotation = _UP_AXIS_ROTATIONS[axis]
        result = rotation @ np.array(up_vector)
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0], atol=1e-10)


class TestRotateToZUp:
    def test_default_z_up_is_identity(self):
        points = np.array([[1.0, 2.0, 3.0]])
        rotated, rotation = rotate_to_z_up(points)
        np.testing.assert_allclose(rotated, points)
        np.testing.assert_allclose(rotation, np.eye(3))

    def test_y_up_maps_y_to_z(self):
        points = np.array([[0.0, 5.0, 0.0]])
        rotated, _ = rotate_to_z_up(points, up_axis="y")
        np.testing.assert_allclose(rotated, [[0.0, 0.0, 5.0]], atol=1e-10)

    def test_unknown_up_axis_raises(self):
        with pytest.raises(ValueError):
            rotate_to_z_up(np.zeros((1, 3)), up_axis="w")


class TestTranslateToOrigin:
    def test_bounding_box_minimum_becomes_origin(self):
        points = np.array([[1.0, 2.0, 3.0], [4.0, -1.0, 5.0], [2.0, 0.0, 3.5]])
        translated, translation = translate_to_origin(points)
        np.testing.assert_allclose(translated.min(axis=0), [0.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(translation, [-1.0, 1.0, -3.0])


class TestNormalise:
    def test_rejects_empty_points(self):
        with pytest.raises(ValueError):
            normalise(np.zeros((0, 3)))

    def test_default_frame_is_already_normalised_bbox_min_at_origin(self):
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        normalised, transform = normalise(points)
        np.testing.assert_allclose(normalised.min(axis=0), [0.0, 0.0, 0.0], atol=1e-10)
        assert transform.scale == 1.0
        np.testing.assert_allclose(transform.rotation, np.eye(3))

    def test_millimetre_y_up_scan_is_normalised_correctly(self):
        # a scan in millimetres, Y-up: (x=1000mm, y=2000mm [up], z=500mm)
        points = np.array([[1000.0, 2000.0, 500.0], [3000.0, 500.0, 1500.0]])
        normalised, transform = normalise(points, unit="mm", up_axis="y")

        assert transform.scale == 0.001
        np.testing.assert_allclose(normalised.min(axis=0), [0.0, 0.0, 0.0], atol=1e-10)
        # after mm->m and Y-up->Z-up, point 0's height (old Y=2000mm=2m) should be
        # the larger of the two z values
        assert normalised[0, 2] > normalised[1, 2]

    @pytest.mark.parametrize("unit", sorted({"m", "cm", "mm", "ft"}))
    @pytest.mark.parametrize("up_axis", sorted(_UP_AXIS_ROTATIONS))
    def test_transform_round_trips_through_apply_and_invert(self, unit, up_axis):
        rng = np.random.default_rng(0)
        points = rng.uniform(-10.0, 10.0, size=(20, 3))
        normalised, transform = normalise(points, unit=unit, up_axis=up_axis)

        np.testing.assert_allclose(transform.apply(points), normalised, atol=1e-8)
        np.testing.assert_allclose(transform.invert(normalised), points, atol=1e-6)

    def test_to_dict_from_dict_round_trip_preserves_behaviour(self):
        rng = np.random.default_rng(1)
        points = rng.uniform(-5.0, 5.0, size=(10, 3))
        _, transform = normalise(points, unit="cm", up_axis="x")

        restored = NormalisationTransform.from_dict(transform.to_dict())

        np.testing.assert_allclose(restored.scale, transform.scale)
        np.testing.assert_allclose(restored.rotation, transform.rotation)
        np.testing.assert_allclose(restored.translation, transform.translation)
        np.testing.assert_allclose(restored.invert(transform.apply(points)), points, atol=1e-6)
