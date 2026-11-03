# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Any
from unittest import TestCase
from unittest.mock import ANY, MagicMock, patch, sentinel

import numpy as np
from hypothesis import given

from tbp.monty.frameworks.models.evidence_matching.hypotheses_updater import (
    DefaultHypothesesUpdater,
)
from tbp.monty.geometry import OrientationMatrix, SurfacePoseVectors
from tests.strategies import sentinels


class DefaultHypothesesUpdaterGetAllInformedPosesTest(TestCase):
    @given(possible_s_d=sentinels.distinct(prefix="s_d", min_size=1))
    @patch(
        "tbp.monty.frameworks.models.evidence_matching.hypotheses_updater.align_multiple_orthonormal_vectors"
    )
    @patch(
        "tbp.monty.frameworks.models.evidence_matching.hypotheses_updater.possible_sensed_directions"
    )
    def test_calls_for_two_possible_sensed_directions_when_surface_pose_vectors_and_pose_fully_defined(  # noqa: E501
        self,
        possible_sensed_directions_mock: MagicMock,
        align_multiple_orthonormal_vectors_mock: MagicMock,
        possible_s_d: list[Any],
    ) -> None:
        align_multiple_orthonormal_vectors_mock.return_value = np.zeros((1, 3, 3))
        graph_memory_mock = MagicMock()
        graph_memory_mock.get_locations_in_graph.return_value = np.zeros((1, 3))
        updater = DefaultHypothesesUpdater(
            feature_weights=MagicMock(),
            graph_memory=graph_memory_mock,
            max_match_distance=MagicMock(),
            tolerances=MagicMock(),
        )
        sensed_directions_stub = SurfacePoseVectors(np.identity(3))
        possible_sensed_directions_mock.return_value = possible_s_d

        updater._get_all_informed_possible_poses(
            graph_id=MagicMock(),
            sensed_channel_features={
                "pose_vectors": sensed_directions_stub,
                "pose_fully_defined": True,
            },
            input_channel=MagicMock(),
        )

        possible_sensed_directions_mock.assert_called_once_with(
            sensed_directions_stub, 2
        )
        assert align_multiple_orthonormal_vectors_mock.call_count == len(possible_s_d)
        for i, s_d in enumerate(possible_s_d):
            call_args = align_multiple_orthonormal_vectors_mock.call_args_list[i]
            assert call_args.args[1] == s_d

    @given(possible_s_d=sentinels.distinct(prefix="s_d", min_size=1))
    @patch(
        "tbp.monty.frameworks.models.evidence_matching.hypotheses_updater.align_multiple_orthonormal_vectors"
    )
    @patch(
        "tbp.monty.frameworks.models.evidence_matching.hypotheses_updater.possible_sensed_directions"
    )
    def test_calls_for_umbilical_num_possible_sensed_directions_when_surface_pose_vectors_and_pose_not_fully_defined(  # noqa: E501
        self,
        possible_sensed_directions_mock: MagicMock,
        align_multiple_orthonormal_vectors_mock: MagicMock,
        possible_s_d: list[Any],
    ) -> None:
        align_multiple_orthonormal_vectors_mock.return_value = np.zeros((1, 3, 3))
        graph_memory_mock = MagicMock()
        graph_memory_mock.get_locations_in_graph.return_value = np.zeros((1, 3))
        updater = DefaultHypothesesUpdater(
            feature_weights=MagicMock(),
            graph_memory=graph_memory_mock,
            max_match_distance=MagicMock(),
            tolerances=MagicMock(),
            umbilical_num_poses=sentinel.umbilical_num_poses,
        )
        sensed_directions_stub = SurfacePoseVectors(np.identity(3))
        possible_sensed_directions_mock.return_value = possible_s_d

        updater._get_all_informed_possible_poses(
            graph_id=MagicMock(),
            sensed_channel_features={
                "pose_vectors": sensed_directions_stub,
                "pose_fully_defined": False,
            },
            input_channel=MagicMock(),
        )

        possible_sensed_directions_mock.assert_called_once_with(
            sensed_directions_stub, sentinel.umbilical_num_poses
        )
        assert align_multiple_orthonormal_vectors_mock.call_count == len(possible_s_d)
        for i, s_d in enumerate(possible_s_d):
            call_args = align_multiple_orthonormal_vectors_mock.call_args_list[i]
            assert call_args.args[1] == s_d

    @patch(
        "tbp.monty.frameworks.models.evidence_matching.hypotheses_updater.align_multiple_orthonormal_vectors"
    )
    @patch(
        "tbp.monty.frameworks.models.evidence_matching.hypotheses_updater.possible_sensed_directions"
    )
    def test_uses_sensed_directions_for_orientation_matrix(
        self,
        possible_sensed_directions_mock: MagicMock,
        align_multiple_orthonormal_vectors_mock: MagicMock,
    ) -> None:
        align_multiple_orthonormal_vectors_mock.return_value = np.zeros((1, 3, 3))
        graph_memory_mock = MagicMock()
        graph_memory_mock.get_locations_in_graph.return_value = np.zeros((1, 3))
        updater = DefaultHypothesesUpdater(
            feature_weights=MagicMock(),
            graph_memory=graph_memory_mock,
            max_match_distance=MagicMock(),
            tolerances=MagicMock(),
        )
        sensed_directions_stub = OrientationMatrix(np.identity(3))

        updater._get_all_informed_possible_poses(
            graph_id=MagicMock(),
            sensed_channel_features={
                "pose_vectors": sensed_directions_stub,
            },
            input_channel=MagicMock(),
        )

        possible_sensed_directions_mock.assert_not_called()
        align_multiple_orthonormal_vectors_mock.assert_called_once_with(
            ANY, sensed_directions_stub, as_scipy=False
        )

    def test_raises_type_error_for_unsupported_pose_vector_type(self) -> None:
        updater = DefaultHypothesesUpdater(
            feature_weights=MagicMock(),
            graph_memory=MagicMock(),
            max_match_distance=MagicMock(),
            tolerances=MagicMock(),
        )

        with self.assertRaises(TypeError) as error:
            updater._get_all_informed_possible_poses(
                graph_id=MagicMock(),
                sensed_channel_features={
                    "pose_vectors": "unsupported_type",
                },
                input_channel=MagicMock(),
            )
        self.assertIn("Unsupported pose_vector type:", str(error.exception))
