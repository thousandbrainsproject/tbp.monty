# Copyright 2025-2026 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import numpy as np
import numpy.typing as npt
from hypothesis import given, settings
from hypothesis import strategies as st

from tbp.monty.frameworks.models.object_model import (
    GraphObjectModel,
    GridObjectModel,
    GridTooSmallError,
)
from tbp.monty.frameworks.utils.object_model_utils import (
    NumpyGraph,
    orthonormal_pose_vectors,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import check_orthonormal
from tbp.monty.geometry import Rotation

RESOURCES = Path(__file__).parents[2] / "resources"  # tests/unit/resources


class ObjectModelTest(unittest.TestCase):
    def setUp(self):
        self.dummy_locs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
        self.dummy_pv = np.eye(3)
        pv = self.dummy_pv.flatten()
        self.dummy_features = {
            "pose_vectors": np.vstack([pv, pv, pv, pv]),
            "pose_fully_defined": np.array([True, True, True, True]),
            "hsv": np.array(
                [
                    [0.1, 1, 1],
                    [0.0, 1, 1],
                    [0.9, 1, 1],
                    [0.8, 1, 1],
                ]
            ),
            "curvature": np.array([0, 1, 2, 1]),
        }
        self.dummy_delta_thresholds = {
            "distance": 0.5,
            "pose_vectors": np.ones(3),
            "hsv": [0.1, 1, 1],
            "curvature": 0.1,
        }

    def test_create_graph_object_model(self):
        model = GraphObjectModel("test_model")
        self.assertIsNotNone(model)
        self.assertIsNone(model._graph)
        # make sure __repr__ works
        self.assertIsInstance(repr(model), str)

    def test_create_grid_object_model(self):
        model = GridObjectModel(
            "test_model", max_nodes=10, max_size=10, num_voxels_per_dim=10
        )
        self.assertIsNotNone(model)
        self.assertIsNone(model._graph)
        # make sure __repr__ works
        self.assertIsInstance(repr(model), str)

    def test_can_build_graph_object_model(self):
        model = GraphObjectModel("test_model")
        model.build_model(
            self.dummy_locs,
            self.dummy_features,
            k_n=None,
            graph_delta_thresholds=self.dummy_delta_thresholds,
        )
        self.assertEqual(model.num_nodes, 4, "graph model should have 4 nodes.")
        for feature in self.dummy_features:
            self.assertIn(
                feature, model.feature_mapping.keys(), f"{feature} not stored in model."
            )

    def test_can_build_grid_object_model(self):
        max_nodes, max_size, num_voxels_per_dim = 10, 10, 10
        model = GridObjectModel("test_model", max_nodes, max_size, num_voxels_per_dim)
        model.build_model(
            self.dummy_locs,
            self.dummy_features,
        )
        self.assertEqual(model.num_nodes, 4, "graph model should have 4 nodes.")
        for feature in self.dummy_features:
            self.assertIn(
                feature, model.feature_mapping.keys(), f"{feature} not stored in model."
            )

    def test_apply_delta_thresholds_correctly(self):
        # Test distance.
        model = GraphObjectModel("test_model")
        model.build_model(
            self.dummy_locs,
            self.dummy_features,
            k_n=None,
            graph_delta_thresholds={
                "distance": 2,
            },
        )
        self.assertEqual(
            model.num_nodes,
            1,
            "Graph model should have 1 node since no locations are more that 2cm away "
            f"from the first one. Model has {model.num_nodes} nodes.",
        )
        # Test pose_vectors.
        model = GraphObjectModel("test_model")
        features = copy.deepcopy(self.dummy_features)
        features["pose_vectors"][1] = np.array([0, 1, 0, 1, 0, 0, 0, 0, 1])
        model.build_model(
            self.dummy_locs,
            features,
            k_n=None,
            graph_delta_thresholds={
                "distance": 2,
                "pose_vectors": np.ones(3),
            },
        )
        self.assertEqual(
            model.num_nodes,
            2,
            "Graph model should have 2 nodes since no locations are more that 2cm away "
            "from the first one but pose vectors 0 and 1 differ enough. "
            f"Model has {model.num_nodes} nodes.",
        )
        # Test hsv.
        model = GraphObjectModel("test_model")
        model.build_model(
            self.dummy_locs,
            self.dummy_features,
            k_n=None,
            graph_delta_thresholds={
                "distance": 2,
                "hsv": [0.1, 1, 1],
            },
        )
        self.assertEqual(
            model.num_nodes,
            2,
            "Graph model should have 2 nodes since no locations are more that 2cm away "
            "from the first one but hsv 0 (0.1) and 3 (0.8) differ enough. "
            f"Model has {model.num_nodes} nodes.",
        )
        # Test curvature.
        model = GraphObjectModel("test_model")
        model.build_model(
            self.dummy_locs,
            self.dummy_features,
            k_n=None,
            graph_delta_thresholds={
                "distance": 2,
                "curvature": 1,
            },
        )
        self.assertEqual(
            model.num_nodes,
            2,
            "Graph model should have 2 nodes since no locations are more that 2cm away "
            "from the first one but curvature 0 (0) and 2 (2) differ enough. "
            f"Model has {model.num_nodes} nodes.",
        )

    def test_grid_locations_binned_correctly(self):
        # In test_can_build_grid_object_model each voxel is of size 1cm^3 so every
        # location gets its own voxel. Here, each voxel is of size 2cm^3 which means
        # that all locations should be binned into the same voxel and the location value
        # should be averaged.
        model = GridObjectModel(
            "test_model", max_nodes=10, max_size=10, num_voxels_per_dim=5
        )
        model.build_model(
            self.dummy_locs,
            self.dummy_features,
        )
        self.assertEqual(
            model.num_nodes,
            1,
            f"graph model should have 1 node but has {model.num_nodes}.",
        )
        self.assertListEqual(
            list(model.pos[0]),
            [0.5, 0.5, 0],
            "location stored in model should be average of all locations "
            f"([0.5, 0.5, 0]) but is {model.pos}.",
        )

    def test_grid_features_are_averaged_correctly(self):
        model = GridObjectModel(
            "test_model", max_nodes=10, max_size=10, num_voxels_per_dim=5
        )
        model.build_model(
            self.dummy_locs,
            self.dummy_features,
        )
        self.assertListEqual(
            list(model.get_values_for_feature("pose_vectors")[0]),
            list(self.dummy_pv.flatten()),
            "pose_vectors not averaged correctly. "
            "Since all PVs are the same, average should be same as dummy_pv.",
        )
        self.assertEqual(
            model.get_values_for_feature("pose_fully_defined")[0],
            True,
            "Average of pose_fully_defined should be True (majority vote).",
        )
        avg_hsv = model.get_values_for_feature("hsv")[0]
        self.assertListEqual(
            list(avg_hsv),
            [0.95, 1, 1],
            "hsv not averaged correctly. "
            "(keep in mind we want the circular average where 0==1).",
        )
        self.assertEqual(
            model.get_values_for_feature("curvature")[0],
            np.mean(self.dummy_features["curvature"]),
            "Curvature not averaged correctly.",
        )
        # Check different cases to make sure pose_fully_defined is averaged correctly.
        features = copy.deepcopy(self.dummy_features)
        features["pose_fully_defined"] = np.array([False, False, False, True])
        model2 = GridObjectModel(
            "test_model", max_nodes=10, max_size=10, num_voxels_per_dim=5
        )
        model2.build_model(
            self.dummy_locs,
            features,
        )
        self.assertEqual(
            model2.get_values_for_feature("pose_fully_defined")[0],
            False,
            "Average of pose_fully_defined should be False (majority vote).",
        )

        features["pose_fully_defined"][0] = True
        model2.build_model(
            self.dummy_locs,
            features,
        )
        self.assertEqual(
            model2.get_values_for_feature("pose_fully_defined")[0],
            True,
            "Average of pose_fully_defined should be True (True in case of tie).",
        )
        # Check different cases to make sure pose_vectors are averaged correctly.
        #     Check opposite surface side vectors are sorted out.
        features = copy.deepcopy(self.dummy_features)
        rot = Rotation.from_euler("z", 180, degrees=True)
        opposite_pv = rot.apply(np.eye(3))
        features["pose_vectors"][0] = opposite_pv.flatten()
        model3 = GridObjectModel(
            "test_model", max_nodes=10, max_size=10, num_voxels_per_dim=5
        )
        model3.build_model(
            self.dummy_locs,
            features,
        )
        self.assertListEqual(
            list(np.round(model3.get_values_for_feature("pose_vectors")[0], 3)),
            list(self.dummy_pv.flatten()),
            "Average of pose_vectors should still be the same. Changed pv is ignored "
            "because its angle is too far from the other ones (other surface side)",
        )
        features["pose_vectors"][1] = opposite_pv.flatten()
        features["pose_vectors"][2] = opposite_pv.flatten()
        model3.build_model(
            self.dummy_locs,
            features,
        )
        self.assertListEqual(
            list(np.round(model3.get_values_for_feature("pose_vectors")[0], 3)),
            list(np.round(opposite_pv.flatten(), 3)),
            "If majority of pose_vectors are on the opposite side of the surface (here)"
            " opposite_pv), those should become the average.",
        )
        #    Check averaged pose vector are still unit vectors.
        features = copy.deepcopy(self.dummy_features)
        rot = Rotation.from_euler("z", 20, degrees=True)
        rotated_pv = rot.apply(np.eye(3))
        features["pose_vectors"][0] = rotated_pv.flatten()
        model4 = GridObjectModel(
            "test_model", max_nodes=10, max_size=10, num_voxels_per_dim=5
        )
        model4.build_model(
            self.dummy_locs,
            features,
        )
        avg_pvs = model4.get_values_for_feature("pose_vectors")[0].reshape((3, 3))
        for pv in avg_pvs:
            self.assertAlmostEqual(
                np.linalg.norm(pv), 1.0, 3, "Average PVs are not unit vectors anymore"
            )
        self.assertTrue(check_orthonormal(avg_pvs), "Average PVs are not orthonormal")

    @settings(deadline=1000)
    @given(pose_fully_defined=st.booleans())
    def test_update_model_keeps_pose_vectors_as_rotation_matrix(
        self, pose_fully_defined
    ):
        opposite_pv = np.vstack(
            [-self.dummy_pv[0], self.dummy_pv[1], -self.dummy_pv[2]]
        )
        model = GridObjectModel(
            "test_model", max_nodes=10, max_size=10, num_voxels_per_dim=5
        )
        model.build_model(self.dummy_locs, self.dummy_features)

        for _ in range(3):
            features = copy.deepcopy(self.dummy_features)
            features["pose_vectors"] = np.vstack([opposite_pv.flatten()] * 4)
            features["pose_fully_defined"] = np.array([pose_fully_defined] * 4)
            model.update_model(
                locations=self.dummy_locs,
                features=features,
                location_rel_model=np.zeros(3),
                object_location_rel_body=np.zeros(3),
                object_rotation=Rotation.identity(),
            )

        for pose_vectors in model.get_values_for_feature("pose_vectors"):
            matrix = np.array(pose_vectors).reshape((3, 3))
            self.assertAlmostEqual(
                np.linalg.det(matrix),
                1.0,
                3,
                "Stored pose vectors are no longer a rotation matrix",
            )
            self.assertTrue(
                check_orthonormal(matrix),
                "Stored pose vectors are not orthonormal",
            )

    def test_max_nodes_applied_correctly(self):
        model = GridObjectModel(
            "test_model", max_nodes=3, max_size=10, num_voxels_per_dim=10
        )
        model.build_model(
            self.dummy_locs,
            self.dummy_features,
        )
        self.assertEqual(model.num_nodes, 3, "Max nodes not applied correctly.")

    def test_max_size_applied_correctly(self):
        """Test that GridTooSmallError is raised if locations are outside of grid."""
        model = GridObjectModel(
            "test_model", max_nodes=10, max_size=1, num_voxels_per_dim=10
        )
        with self.assertRaises(GridTooSmallError):
            model.build_model(
                self.dummy_locs,
                self.dummy_features,
            )


class GraphObjectModelGoldenTest(unittest.TestCase):
    """NumPy graph build must reproduce the frozen torch golden output.

    Characterization test over object_model_graph_golden.npz, captured pre-refactor.
    """

    def setUp(self) -> None:
        self.golden = np.load(RESOURCES / "object_model_graph_golden.npz")
        self.golden_fm = json.loads(
            (RESOURCES / "object_model_graph_golden_feature_mapping.json").read_text()
        )

        # Inputs must match the capture script exactly
        self.dummy_locs = np.array(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
        )
        pv = np.eye(3).flatten()
        self.dummy_features = {
            "pose_vectors": np.vstack([pv, pv, pv, pv]),
            "pose_fully_defined": np.array([True, True, True, True]),
            "hsv": np.array(
                [[0.1, 1.0, 1.0], [0.0, 1.0, 1.0], [0.9, 1.0, 1.0], [0.8, 1.0, 1.0]]
            ),
            "curvature": np.array([0.0, 1.0, 2.0, 1.0]),
        }
        self.dummy_thresholds = {
            "distance": 0.5,
            "pose_vectors": np.ones(3),
            "hsv": [0.1, 1.0, 1.0],
            "curvature": 0.1,
        }

        normals = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
        cds = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, -1.0, 0.0]]
        self.ppf_features = {
            "pose_vectors": np.vstack(
                [
                    orthonormal_pose_vectors(np.array(n), np.array(c))
                    for n, c in zip(normals, cds)
                ]
            ),
            "pose_fully_defined": np.array([True, True, True, True]),
            "hsv": np.array(
                [[0.1, 1.0, 1.0], [0.3, 1.0, 1.0], [0.6, 1.0, 1.0], [0.9, 1.0, 1.0]]
            ),
            "curvature": np.array([0.0, 1.0, 2.0, 3.0]),
        }
        self.ppf_locs = np.array(
            [[0.0, 0.0, 0.0], [0.2, 0.1, 0.3], [0.4, 0.5, 0.1], [0.1, 0.6, 0.7]]
        )
        self.ppf_thresholds = {
            "distance": 0.0001,
            "pose_vectors": np.ones(3),
            "hsv": [0.1, 1.0, 1.0],
            "curvature": 0.1,
        }

    def _build(
        self,
        locs: npt.NDArray[np.float64],
        feats: dict,
        thresholds: dict,
        k_n: int | None,
        ppf: bool = False,
    ) -> NumpyGraph:
        m = GraphObjectModel("golden")
        m.build_model(locs, feats, k_n=k_n, graph_delta_thresholds=thresholds)
        if ppf:
            m.add_ppf_to_graph()
        return m._graph

    def _assert_nodes(self, g: NumpyGraph, prefix: str) -> None:
        np.testing.assert_allclose(g.x, self.golden[f"{prefix}_x"], atol=1e-6, rtol=0)
        np.testing.assert_allclose(
            g.pos, self.golden[f"{prefix}_pos"], atol=1e-6, rtol=0
        )
        np.testing.assert_allclose(
            g.norm, self.golden[f"{prefix}_norm"], atol=1e-6, rtol=0
        )
        self.assertEqual(g.feature_mapping, self.golden_fm[prefix])

    def test_nodes_only(self) -> None:
        self._assert_nodes(
            self._build(
                self.dummy_locs, self.dummy_features, self.dummy_thresholds, k_n=None
            ),
            "node",
        )

    def test_nodes_and_edges(self) -> None:
        g = self._build(
            self.dummy_locs, self.dummy_features, self.dummy_thresholds, k_n=2
        )
        self._assert_nodes(g, "edge")

        # edge_index first and exact - ordering drift corrupts edge_attr rows
        np.testing.assert_array_equal(g.edge_index, self.golden["edge_edge_index"])
        np.testing.assert_allclose(
            g.edge_attr, self.golden["edge_edge_attr"], atol=1e-6, rtol=0
        )

    def test_ppf(self) -> None:
        g = self._build(
            self.ppf_locs, self.ppf_features, self.ppf_thresholds, k_n=2, ppf=True
        )
        self._assert_nodes(g, "ppf")
        np.testing.assert_array_equal(g.edge_index, self.golden["ppf_edge_index"])
        np.testing.assert_allclose(
            g.edge_attr, self.golden["ppf_edge_attr"], atol=1e-6, rtol=0
        )
