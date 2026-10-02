# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial import KDTree

from tbp.monty.frameworks.models.evidence_matching.channels import PoseKind
from tbp.monty.frameworks.models.evidence_matching.feature_evidence.scorer import (
    DefaultFeatureEvidenceScorer,
)
from tbp.monty.frameworks.models.evidence_matching.hypotheses import Hypotheses
from tbp.monty.frameworks.models.evidence_matching.hypotheses_displacer import (
    DefaultHypothesesDisplacer,
)
from tbp.monty.geometry import Rotation


class DefaultHypothesesDisplacerTest(TestCase):
    def setUp(self) -> None:
        self.mock_graph_memory = Mock()
        self.mock_graph_memory.get_input_channels_in_graph = Mock(
            return_value=["channel_a", "channel_b"]
        )

        self.feature_weights = {
            "channel_a": {"pose_vectors": [1, 1]},
            "channel_b": {"pose_vectors": [1, 1]},
        }
        self.tolerances = {
            "channel_a": {},
            "channel_b": {},
        }
        self.feature_for_matching_selector = Mock(
            select=Mock(return_value={"channel_a": False, "channel_b": False})
        )
        self.feature_evidence_scorer = DefaultFeatureEvidenceScorer(
            graph_memory=self.mock_graph_memory,
            feature_weights=self.feature_weights,
            tolerances=self.tolerances,
            features_for_matching_selector=self.feature_for_matching_selector,
        )
        self.displacer = DefaultHypothesesDisplacer(
            feature_weights=self.feature_weights,
            graph_memory=self.mock_graph_memory,
            max_match_distance=0.01,
            feature_evidence_scorer=self.feature_evidence_scorer,
            past_weight=1,
            present_weight=1,
        )

    def test_clamps_neighbors_to_channel_node_count(self) -> None:
        channel_locations = np.zeros((1, 3))
        location_tree = KDTree(channel_locations)
        graph = Mock()
        graph.find_nearest_neighbors = Mock(
            side_effect=lambda search_locations, num_neighbors: location_tree.query(
                search_locations,
                k=num_neighbors,
            )[1]
        )
        self.mock_graph_memory.get_graph.return_value = graph
        self.mock_graph_memory.get_locations_in_graph.return_value = channel_locations
        self.mock_graph_memory.get_feature_array.return_value = {
            "channel_a": np.empty((1, 0))
        }
        self.mock_graph_memory.get_features_at_node.return_value = {
            "pose_vectors": np.eye(3).reshape(1, 1, 9),
            "pose_fully_defined": np.ones((1, 1, 1)),
        }

        evidence = self.displacer._calculate_evidence_for_new_locations(
            graph_id="test_object",
            input_channel="channel_a",
            pose_kind=PoseKind.SURFACE,
            search_locations=np.zeros((1, 3)),
            channel_possible_poses=np.eye(3).reshape(1, 3, 3),
            channel_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": True,
            },
        )

        self.assertEqual(
            graph.find_nearest_neighbors.call_args.kwargs["num_neighbors"],
            1,
            "a one-node graph[channel] should request for one nearest neighbor",
        )
        self.assertEqual(
            evidence.shape,
            (1,),
            "a single search location should produce exactly one evidence value",
        )

    def test_multi_channel_evidence_sums(self) -> None:
        """Test that evidence from two channels is summed and added to hypotheses.

        Sets up two channels each returning known evidence arrays, and verifies
        the total evidence is the sum of per-channel contributions.
        """
        num_hyps = 3
        hypotheses = Hypotheses(
            evidence=np.array([1.0, 2.0, 3.0]),
            locations=np.zeros((num_hyps, 3)),
            poses=np.tile(np.eye(3), (num_hyps, 1, 1)),
            possible=np.ones(num_hyps, dtype=bool),
        )

        # Channel A returns evidence [0.5, 0.5, 0.5]
        # Channel B returns evidence [1.0, 0.0, -1.0]
        evidence_by_channel = {
            "channel_a": np.array([0.5, 0.5, 0.5]),
            "channel_b": np.array([1.0, 0.0, -1.0]),
        }
        with patch.object(
            self.displacer,
            "_calculate_evidence_for_new_locations",
            side_effect=lambda **kw: evidence_by_channel[kw["input_channel"]],
        ):
            displaced = self.displacer.displace_hypotheses(
                displacement=np.zeros(3),
                hypotheses=hypotheses,
            )
            result, _telemetry = self.displacer.compute_evidence(
                features={
                    "channel_a": {"pose_fully_defined": True},
                    "channel_b": {"pose_fully_defined": True},
                },
                evidence_update_threshold=-np.inf,
                graph_id="test_object",
                hypotheses=displaced,
                pose_kinds={
                    "channel_a": PoseKind.SURFACE,
                    "channel_b": PoseKind.SURFACE,
                },
            )

        # Expected: past_weight * old_evidence + present_weight * summed_new
        # summed_new = [0.5+1.0, 0.5+0.0, 0.5+(-1.0)] = [1.5, 0.5, -0.5]
        # result = 1 * [1.0, 2.0, 3.0] + 1 * [1.5, 0.5, -0.5] = [2.5, 2.5, 2.5]
        np.testing.assert_array_almost_equal(result.evidence, [2.5, 2.5, 2.5])

    def test_prediction_error_computed_from_summed_evidence(self) -> None:
        num_hyps = 2
        hypotheses = Hypotheses(
            evidence=np.array([5.0, 1.0]),  # hyp 0 is MLH
            locations=np.zeros((num_hyps, 3)),
            poses=np.tile(np.eye(3), (num_hyps, 1, 1)),
            possible=np.ones(num_hyps, dtype=bool),
        )

        evidence_by_channel = {
            "channel_a": np.array([1.5, 0.5]),
            "channel_b": np.array([0.5, -0.5]),
        }

        with patch.object(
            self.displacer,
            "_calculate_evidence_for_new_locations",
            side_effect=lambda **kw: evidence_by_channel[kw["input_channel"]],
        ):
            displaced = self.displacer.displace_hypotheses(
                displacement=np.zeros(3),
                hypotheses=hypotheses,
            )
            _, telemetry = self.displacer.compute_evidence(
                features={
                    "channel_a": {"pose_fully_defined": True},
                    "channel_b": {"pose_fully_defined": True},
                },
                evidence_update_threshold=-np.inf,
                graph_id="test_object",
                hypotheses=displaced,
                pose_kinds={
                    "channel_a": PoseKind.SURFACE,
                    "channel_b": PoseKind.SURFACE,
                },
            )

        # MLH is index 0 (evidence 5.0), summed evidence at MLH = 1.5 + 0.5 = 2.0
        # With 2 channels (C=2), range is [-C, 2C] = [-2, 4], mapped to [0, 1]:
        # prediction_error = (-2.0 + 2*2) / (3*2) = 1/3
        self.assertAlmostEqual(telemetry.mlh_prediction_error, 1 / 3)

    def test_channel_missing_from_pose_kinds_raises(self) -> None:
        hypotheses = Hypotheses(
            evidence=np.zeros(1),
            locations=np.zeros((1, 3)),
            poses=np.eye(3).reshape(1, 3, 3),
            possible=np.ones(1, dtype=bool),
        )

        with patch.object(
            self.displacer,
            "_calculate_evidence_for_new_locations",
            return_value=np.zeros(1),
        ):
            with self.assertRaises(KeyError):
                self.displacer.compute_evidence(
                    features={
                        "channel_a": {"pose_fully_defined": True},
                        "channel_b": {"pose_fully_defined": True},
                    },
                    evidence_update_threshold=-np.inf,
                    graph_id="test_object",
                    hypotheses=hypotheses,
                    pose_kinds={"channel_a": PoseKind.SURFACE},
                )


class ObjectPoseEvidenceTest(TestCase):
    def setUp(self) -> None:
        self.displacer = DefaultHypothesesDisplacer(
            feature_weights={},
            graph_memory=Mock(),
            max_match_distance=0.01,
            feature_evidence_scorer=Mock(),
        )

    def _evidence(self, sensed: Rotation, stored: Rotation) -> float:
        query_features = {"pose_vectors": sensed.as_matrix().reshape(1, 3, 3)}
        node_features = {"pose_vectors": stored.as_matrix().reshape(1, 1, 9)}
        evidence = self.displacer._get_object_pose_evidence_matrix(
            query_features, node_features
        )
        return evidence[0, 0]

    def test_flip_about_first_pose_vector_gives_min_evidence(self) -> None:
        """The surface pose path would score this as a match."""
        flipped = Rotation.from_euler("x", 180, degrees=True)
        self.assertAlmostEqual(self._evidence(flipped, Rotation.identity()), -1.0)
