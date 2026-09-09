# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, sentinel

import numpy as np
import numpy.testing as nptest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.models.evidence_matching.region_proposal.context import (
    EvidenceLMRegionContext,
)

MAX_POINTS = 20
MAX_COORDINATE = 10.0

coordinates = st.floats(min_value=-MAX_COORDINATE, max_value=MAX_COORDINATE)
locations = arrays(dtype=np.float64, shape=(3,), elements=coordinates)
point_sets = st.integers(min_value=1, max_value=MAX_POINTS).flatmap(
    lambda n: arrays(dtype=np.float64, shape=(n, 3), elements=coordinates)
)
rotations = arrays(
    dtype=np.float64, shape=(3,), elements=st.floats(min_value=-180, max_value=180)
).map(lambda euler: Rotation.from_euler("xyz", euler, degrees=True))


def lm_with_pose(
    current_location: np.ndarray, mlh_location: np.ndarray, rotation: Rotation
) -> MagicMock:
    # An LM whose buffer sits at current_location and whose MLH for any
    # object places the sensor at mlh_location in the model frame.
    lm = MagicMock()
    lm.buffer.current_location.return_value = current_location
    lm.get_mlh_for_object.return_value = {
        "location": mlh_location,
        "rotation": rotation,
    }
    return lm


class GoalsTest(unittest.TestCase):
    def test_are_the_goals_the_lm_proposes(self) -> None:
        lm = MagicMock()
        lm.propose_goals.return_value = [sentinel.goal_a, sentinel.goal_b]

        self.assertEqual(
            EvidenceLMRegionContext(lm).goals, (sentinel.goal_a, sentinel.goal_b)
        )

    def test_are_empty_when_the_lm_proposes_none(self) -> None:
        lm = MagicMock()
        lm.propose_goals.return_value = []

        self.assertEqual(EvidenceLMRegionContext(lm).goals, ())


class RecognizedObjectTest(unittest.TestCase):
    def test_is_the_sole_match_once_the_terminal_state_is_match(self) -> None:
        lm = MagicMock()
        lm.terminal_state = "match"
        lm.get_possible_matches.return_value = ["mug"]

        self.assertEqual(EvidenceLMRegionContext(lm).recognized_object, "mug")

    @given(
        terminal_state=st.sampled_from([None, "no_match", "time_out", "pose_time_out"])
    )
    def test_is_none_unless_the_terminal_state_is_match(
        self,
        terminal_state: str | None,
    ) -> None:
        lm = MagicMock()
        lm.terminal_state = terminal_state
        lm.get_possible_matches.return_value = ["mug"]

        self.assertIsNone(EvidenceLMRegionContext(lm).recognized_object)

    @given(
        matches=st.one_of(
            # no match left, or still several
            st.just([]),
            st.lists(st.text(min_size=1), min_size=2, max_size=4),
        )
    )
    def test_is_none_unless_exactly_one_match_remains(self, matches: list[str]) -> None:
        lm = MagicMock()
        lm.terminal_state = "match"
        lm.get_possible_matches.return_value = matches

        self.assertIsNone(EvidenceLMRegionContext(lm).recognized_object)


class ToBodyFrameTest(unittest.TestCase):
    @given(current=locations, mlh_location=locations, rotation=rotations)
    def test_maps_the_sensor_model_location_onto_its_body_location(
        self,
        current: np.ndarray,
        mlh_location: np.ndarray,
        rotation: Rotation,
    ) -> None:
        context = EvidenceLMRegionContext(lm_with_pose(current, mlh_location, rotation))

        body = context.to_body_frame(mlh_location[None, :], "mug")

        nptest.assert_allclose(body[0], current, atol=1e-9)

    @given(
        points=point_sets, current=locations, mlh_location=locations, rotation=rotations
    )
    def test_is_rigid(
        self,
        points: np.ndarray,
        current: np.ndarray,
        mlh_location: np.ndarray,
        rotation: Rotation,
    ) -> None:
        # Pairwise distances survive the change of frame.
        context = EvidenceLMRegionContext(lm_with_pose(current, mlh_location, rotation))

        body = context.to_body_frame(points, "mug")

        model_distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
        body_distances = np.linalg.norm(body[:, None] - body[None, :], axis=-1)
        nptest.assert_allclose(body_distances, model_distances, atol=1e-8)

    @given(
        points=point_sets, current=locations, mlh_location=locations, rotation=rotations
    )
    def test_rotates_model_displacements_by_the_inverse_mlh_rotation(
        self,
        points: np.ndarray,
        current: np.ndarray,
        mlh_location: np.ndarray,
        rotation: Rotation,
    ) -> None:
        # The MLH rotation takes body displacements into the model frame, so
        # model displacements come back through its inverse (the convention
        # the goal generator uses for its targets).
        context = EvidenceLMRegionContext(lm_with_pose(current, mlh_location, rotation))

        body = context.to_body_frame(points, "mug")

        nptest.assert_allclose(
            rotation.apply(body - current), points - mlh_location, atol=1e-8
        )

    def test_calls_get_mlh_for_object(self) -> None:
        lm = lm_with_pose(np.zeros(3), np.zeros(3), Rotation.identity())
        EvidenceLMRegionContext(lm).to_body_frame(np.zeros((1, 3)), sentinel.object_id)

        lm.get_mlh_for_object.assert_called_once_with(sentinel.object_id)


class SurfaceTest(unittest.TestCase):
    def test_reads_locations_and_first_pose_vectors_of_the_first_channel(self) -> None:
        lm = MagicMock()
        memory = lm.graph_memory
        memory.get_input_channels_in_graph.return_value = [sentinel.channel, "other"]
        memory.get_locations_in_graph.return_value = np.arange(6.0).reshape(2, 3)
        pose_vectors = np.arange(18.0).reshape(2, 3, 3)
        memory.get_rotation_features_at_all_nodes.return_value = pose_vectors

        points, normals = EvidenceLMRegionContext(lm).surface(sentinel.object_id)

        memory.get_locations_in_graph.assert_called_once_with(
            sentinel.object_id, sentinel.channel
        )
        memory.get_rotation_features_at_all_nodes.assert_called_once_with(
            sentinel.object_id, sentinel.channel
        )
        nptest.assert_array_equal(points, np.arange(6.0).reshape(2, 3))
        nptest.assert_array_equal(normals, pose_vectors[:, 0, :])
