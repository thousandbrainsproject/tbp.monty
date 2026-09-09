# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Tests for the feature-based graph-mismatch logic of the EvidenceGoalGenerator.

The spatial (Euclidean) mismatch path is additionally covered end-to-end in
tests/unit/frameworks/models/evidence_matching/evidence_lm_test.py; the tests here
use mock graphs to exercise the discrete (object ID) and continuous (hue) feature
paths that are used when two graphs are spatially near-identical.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as nptest
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.goal_generation import EvidenceGoalGenerator

SENSOR_CHANNEL = "patch"
TOP_ID = "object_a"
SECOND_ID = "object_b"


class FakeGraph:
    """Minimal stand-in for a learned object graph (one input channel).

    Mirrors the parts of the object-model interface used by the GSG: `pos`, `x`,
    `feature_mapping` and `find_nearest_neighbors`.
    """

    def __init__(self, pos, features=None):
        self.pos = np.asarray(pos, dtype=float)
        self.feature_mapping = {}
        columns = []
        num_columns = 0
        for name, raw_values in (features or {}).items():
            values = np.asarray(raw_values, dtype=float)
            if values.ndim == 1:
                values = values[:, None]
            self.feature_mapping[name] = [num_columns, num_columns + values.shape[1]]
            num_columns += values.shape[1]
            columns.append(values)
        self.x = (
            np.column_stack(columns)
            if columns
            else np.zeros((len(self.pos), 0))
        )
        self._tree = KDTree(self.pos)

    def find_nearest_neighbors(
        self,
        search_locations,
        num_neighbors,
        return_distance=False,
    ):
        distances, nearest_node_ids = self._tree.query(
            search_locations, k=num_neighbors
        )
        if return_distance:
            return distances
        return nearest_node_ids


def identity_mlh(graph_id):
    """An MLH with identity rotation at the origin (transform is a no-op).

    Returns:
        The MLH dict.
    """
    return {
        "graph_id": graph_id,
        "mlh_id": 0,
        "location": np.zeros(3),
        "rotation": Rotation.identity(),
    }


def gsg_with_graphs(graphs, **gsg_kwargs) -> EvidenceGoalGenerator:
    """Build an EvidenceGoalGenerator around a mocked parent LM.

    Args:
        graphs: Nested dict of graph_id -> input_channel -> FakeGraph.
        **gsg_kwargs: Forwarded to the EvidenceGoalGenerator constructor.

    Returns:
        The GSG, with its parent LM mocked to serve the given graphs and to
        report object_a and object_b as the top-two MLH objects.
    """
    lm = MagicMock()
    lm.learning_module_id = "learning_module_2"
    lm.buffer.get_first_sensory_input_channel.return_value = SENSOR_CHANNEL
    lm.get_top_two_mlh_ids.return_value = (TOP_ID, SECOND_ID)
    lm.get_mlh_for_object.side_effect = identity_mlh
    lm._get_current_mlh.return_value = identity_mlh(TOP_ID)
    lm.get_graph.side_effect = lambda graph_id, input_channel=None: (
        graphs[graph_id]
        if input_channel is None
        else graphs[graph_id][input_channel]
    )
    lm.get_input_channels_in_graph.side_effect = lambda graph_id: list(
        graphs[graph_id].keys()
    )
    gsg = EvidenceGoalGenerator(**gsg_kwargs)
    gsg.parent_lm = lm
    gsg.focus_on_pose = False
    return gsg


def make_ctx() -> RuntimeContext:
    return RuntimeContext(rng=np.random.RandomState(42))


# A small flat point cloud used for spatially-identical sensor channels.
BASE_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.05, 0.0, 0.0],
        [0.05, 0.05, 0.0],
        [0.0, 0.05, 0.0],
    ]
)


def sensor_graph(hues=None):
    """A sensor-channel graph over BASE_POINTS, optionally with HSV features.

    Returns:
        The graph.
    """
    features = None
    if hues is not None:
        ones = np.ones_like(hues)
        features = {"hsv": np.column_stack([hues, ones, ones])}
    return FakeGraph(BASE_POINTS, features)


class SpatialPathTest(unittest.TestCase):
    def test_spatial_target_returned_when_graphs_differ_spatially(self) -> None:
        # The top graph has an extra point 5cm away from anything in the second
        # graph (like a mug handle), so the spatial path should propose it.
        top_points = np.vstack([BASE_POINTS, [0.025, 0.1, 0.0]])
        graphs = {
            TOP_ID: {SENSOR_CHANNEL: FakeGraph(top_points)},
            SECOND_ID: {SENSOR_CHANNEL: sensor_graph()},
        }
        gsg = gsg_with_graphs(graphs)

        channel, target_loc_id = gsg._compute_graph_mismatch(make_ctx())

        self.assertEqual(channel, SENSOR_CHANNEL)
        self.assertEqual(target_loc_id, len(top_points) - 1)


class DiscreteFeaturePathTest(unittest.TestCase):
    def lm_channel_graphs(self, positions, top_object_ids, second_object_ids):
        """Build matching-position LM channel graphs with the given object IDs.

        Returns:
            Tuple of (top graph, second graph).
        """
        return (
            FakeGraph(positions, {"object_id": top_object_ids}),
            FakeGraph(positions, {"object_id": second_object_ids}),
        )

    def test_target_is_model_point_at_center_of_mismatch_cluster(self) -> None:
        # Three contiguous mismatching nodes (a "logo") plus one far-away
        # mismatching outlier; the target should be the middle of the cluster,
        # and the outlier should not drag the center away.
        lm_positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.004, 0.0, 0.0],
                [0.008, 0.0, 0.0],
                [0.03, 0.03, 0.0],  # matching node
                [0.1, 0.0, 0.0],  # mismatching outlier, >1cm from the cluster
            ]
        )
        top_ids = np.array([1, 1, 1, 2, 3])
        second_ids = np.array([7, 7, 7, 2, 8])
        top_lm, second_lm = self.lm_channel_graphs(lm_positions, top_ids, second_ids)
        graphs = {
            TOP_ID: {SENSOR_CHANNEL: sensor_graph(), "learning_module_0": top_lm},
            SECOND_ID: {
                SENSOR_CHANNEL: sensor_graph(),
                "learning_module_0": second_lm,
            },
        }
        gsg = gsg_with_graphs(graphs)

        channel, target_loc_id = gsg._compute_graph_mismatch(make_ctx())

        self.assertEqual(channel, "learning_module_0")
        self.assertEqual(
            target_loc_id,
            1,
            "Target should be the model point closest to the cluster's center, "
            "with the far-away mismatching node excluded as an outlier.",
        )

    def test_channel_with_larger_mismatch_cluster_wins(self) -> None:
        cluster_of_two = np.array([[0.0, 0.0, 0.0], [0.005, 0.0, 0.0]])
        cluster_of_three = np.array(
            [[0.0, 0.02, 0.0], [0.005, 0.02, 0.0], [0.01, 0.02, 0.0]]
        )
        top_lm0, second_lm0 = self.lm_channel_graphs(
            cluster_of_two, np.array([1, 1]), np.array([2, 2])
        )
        top_lm1, second_lm1 = self.lm_channel_graphs(
            cluster_of_three, np.array([1, 1, 1]), np.array([2, 2, 2])
        )
        graphs = {
            TOP_ID: {
                SENSOR_CHANNEL: sensor_graph(),
                "learning_module_0": top_lm0,
                "learning_module_1": top_lm1,
            },
            SECOND_ID: {
                SENSOR_CHANNEL: sensor_graph(),
                "learning_module_0": second_lm0,
                "learning_module_1": second_lm1,
            },
        }
        gsg = gsg_with_graphs(graphs)

        channel, target_loc_id = gsg._compute_graph_mismatch(make_ctx())

        self.assertEqual(channel, "learning_module_1")
        self.assertEqual(target_loc_id, 1, "Middle of the three-node cluster.")

    def test_object_id_mismatch_prioritized_over_hue_mismatch(self) -> None:
        # The sensor channel has a large hue difference, but a single
        # mismatching object ID should still take precedence.
        lm_positions = np.array([[0.0, 0.0, 0.0]])
        top_lm, second_lm = self.lm_channel_graphs(
            lm_positions, np.array([1]), np.array([2])
        )
        graphs = {
            TOP_ID: {
                SENSOR_CHANNEL: sensor_graph(hues=np.array([0.0, 0.5, 0.0, 0.0])),
                "learning_module_0": top_lm,
            },
            SECOND_ID: {
                SENSOR_CHANNEL: sensor_graph(hues=np.zeros(4)),
                "learning_module_0": second_lm,
            },
        }
        gsg = gsg_with_graphs(graphs)

        channel, _ = gsg._compute_graph_mismatch(make_ctx())

        self.assertEqual(channel, "learning_module_0")

    def test_no_goal_when_object_ids_match_and_no_other_features(self) -> None:
        lm_positions = np.array([[0.0, 0.0, 0.0], [0.005, 0.0, 0.0]])
        top_lm, second_lm = self.lm_channel_graphs(
            lm_positions, np.array([1, 1]), np.array([1, 1])
        )
        graphs = {
            TOP_ID: {SENSOR_CHANNEL: sensor_graph(), "learning_module_0": top_lm},
            SECOND_ID: {
                SENSOR_CHANNEL: sensor_graph(),
                "learning_module_0": second_lm,
            },
        }
        gsg = gsg_with_graphs(graphs)

        self.assertIsNone(gsg._compute_graph_mismatch(make_ctx()))


class ContinuousFeaturePathTest(unittest.TestCase):
    def test_target_is_node_with_maximal_circular_hue_distance(self) -> None:
        # Node 1 has the largest hue difference (0.5 vs 0.9 -> 0.4); node 0's
        # difference wraps around the hue circle (0.95 vs 0.05 -> 0.1).
        top_hues = np.array([0.95, 0.5, 0.2, 0.3])
        second_hues = np.array([0.05, 0.9, 0.2, 0.3])
        graphs = {
            TOP_ID: {SENSOR_CHANNEL: sensor_graph(hues=top_hues)},
            SECOND_ID: {SENSOR_CHANNEL: sensor_graph(hues=second_hues)},
        }
        gsg = gsg_with_graphs(graphs)

        channel, target_loc_id = gsg._compute_graph_mismatch(make_ctx())

        self.assertEqual(channel, SENSOR_CHANNEL)
        self.assertEqual(target_loc_id, 1)

    def test_tied_hue_distances_resolved_to_one_of_the_tied_nodes(self) -> None:
        top_hues = np.array([0.5, 0.5, 0.0, 0.0])
        second_hues = np.array([0.9, 0.9, 0.0, 0.0])
        graphs = {
            TOP_ID: {SENSOR_CHANNEL: sensor_graph(hues=top_hues)},
            SECOND_ID: {SENSOR_CHANNEL: sensor_graph(hues=second_hues)},
        }
        gsg = gsg_with_graphs(graphs)

        _, target_loc_id = gsg._compute_graph_mismatch(make_ctx())

        self.assertIn(target_loc_id, [0, 1])

    def test_no_goal_when_hue_difference_below_threshold(self) -> None:
        top_hues = np.array([0.5, 0.5, 0.0, 0.0])
        second_hues = np.array([0.55, 0.5, 0.0, 0.0])
        graphs = {
            TOP_ID: {SENSOR_CHANNEL: sensor_graph(hues=top_hues)},
            SECOND_ID: {SENSOR_CHANNEL: sensor_graph(hues=second_hues)},
        }
        gsg = gsg_with_graphs(graphs, min_hue_mismatch=0.1)

        self.assertIsNone(gsg._compute_graph_mismatch(make_ctx()))

    def test_no_goal_when_no_valid_feature_channels(self) -> None:
        graphs = {
            TOP_ID: {SENSOR_CHANNEL: sensor_graph()},
            SECOND_ID: {SENSOR_CHANNEL: sensor_graph()},
        }
        gsg = gsg_with_graphs(graphs)

        self.assertIsNone(gsg._compute_graph_mismatch(make_ctx()))

    def test_generate_goal_returns_none_goal_when_no_mismatch(self) -> None:
        graphs = {
            TOP_ID: {SENSOR_CHANNEL: sensor_graph()},
            SECOND_ID: {SENSOR_CHANNEL: sensor_graph()},
        }
        gsg = gsg_with_graphs(graphs)

        self.assertIsNone(gsg._generate_goal(make_ctx(), observations=[]))


class TargetLocInfoTest(unittest.TestCase):
    def test_lm_channel_target_uses_surface_normal_of_nearest_sensor_node(
        self,
    ) -> None:
        # Distinct normals per sensor node, so we can verify which one is used
        normals = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        pose_vectors = np.column_stack(
            [normals, np.zeros((4, 3)), np.zeros((4, 3))]
        )
        sensor = FakeGraph(BASE_POINTS, {"pose_vectors": pose_vectors})
        # An LM-channel node sitting just next to sensor node 2
        lm_graph = FakeGraph(
            np.array([[0.051, 0.049, 0.0]]), {"object_id": np.array([1])}
        )
        graphs = {
            TOP_ID: {SENSOR_CHANNEL: sensor, "learning_module_0": lm_graph},
            SECOND_ID: {SENSOR_CHANNEL: sensor, "learning_module_0": lm_graph},
        }
        gsg = gsg_with_graphs(graphs)

        target_info = gsg._get_target_loc_info(
            target_loc_id=0, input_channel="learning_module_0"
        )

        nptest.assert_allclose(target_info["target_loc"], [0.051, 0.049, 0.0])
        nptest.assert_allclose(
            target_info["target_surface_normal"],
            normals[2],
            err_msg="Surface normal should come from the nearest sensor-channel "
            "node, not the LM-channel node.",
        )

    def test_sensor_channel_target_uses_its_own_pose_vectors(self) -> None:
        normals = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        pose_vectors = np.column_stack(
            [normals, np.zeros((4, 3)), np.zeros((4, 3))]
        )
        sensor = FakeGraph(BASE_POINTS, {"pose_vectors": pose_vectors})
        graphs = {
            TOP_ID: {SENSOR_CHANNEL: sensor},
            SECOND_ID: {SENSOR_CHANNEL: sensor},
        }
        gsg = gsg_with_graphs(graphs)

        target_info = gsg._get_target_loc_info(
            target_loc_id=3, input_channel=SENSOR_CHANNEL
        )

        nptest.assert_allclose(target_info["target_loc"], BASE_POINTS[3])
        nptest.assert_allclose(target_info["target_surface_normal"], normals[3])


class GeneratedGoalInfoTest(unittest.TestCase):
    def test_goal_info_carries_hypothesis_identity_and_predicted_displacement(
        self,
    ) -> None:
        # The top graph has an extra point 10cm away (like a mug handle) that
        # the spatial mismatch path will propose as the target location.
        extra_point = np.array([0.025, 0.1, 0.0])
        top_points = np.vstack([BASE_POINTS, extra_point])
        pose_vectors = np.column_stack(
            [np.tile([0.0, 0.0, 1.0], (len(top_points), 1)), np.zeros((5, 6))]
        )
        graphs = {
            TOP_ID: {
                SENSOR_CHANNEL: FakeGraph(
                    top_points, {"pose_vectors": pose_vectors}
                )
            },
            SECOND_ID: {SENSOR_CHANNEL: sensor_graph()},
        }
        gsg = gsg_with_graphs(graphs)
        gsg.parent_lm.get_output.return_value = MagicMock(confidence=1.0)
        sensed_location = np.array([0.01, 0.0, 0.0])
        sensory_input = MagicMock(sender_id=SENSOR_CHANNEL, location=sensed_location)

        goal = gsg._generate_goal(make_ctx(), observations=[sensory_input])

        # The identity of the hypothesis behind the goal is snapshotted so the
        # LM can attribute a failed jump to it later.
        self.assertEqual(goal.info["hypothesis_to_test_graph_id"], TOP_ID)
        self.assertEqual(goal.info["hypothesis_to_test_mlh_id"], 0)
        # The MLH is at the model origin with identity rotation, so a
        # successful jump displaces the sensor by the model-frame vector from
        # the MLH location to the target point.
        nptest.assert_allclose(goal.info["predicted_displacement"], extra_point)
        nptest.assert_allclose(
            goal.info["proposed_surface_loc"],
            sensed_location + goal.info["predicted_displacement"],
        )


if __name__ == "__main__":
    unittest.main()
