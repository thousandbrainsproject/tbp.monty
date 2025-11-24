# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from pathlib import Path

import hydra
import pytest
from omegaconf import DictConfig

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)


import shutil
import tempfile
import unittest

from tbp.monty.frameworks.experiments import (
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.frameworks.utils.graph_matching_utils import get_correct_k_n


class GraphBuildingTest(unittest.TestCase):
    def setUp(self):
        """Code that gets executed before every test."""
        self.output_dir = tempfile.mkdtemp()
        self.habitat_save_path = tempfile.mkdtemp()
        self.mesh_save_path = tempfile.mkdtemp()
        self.model_load_path = Path(self.habitat_save_path) / "pretrained"
        self.habitat_learned_rotations = [[0.0, 0.0, 0.0], [0.0, 45.0, 0.0]]

        def training_config(test_name: str) -> DictConfig:
            return hydra.compose(
                config_name="test",
                overrides=[
                    f"test=graph_building/{test_name}",
                    f"test.config.logging.output_dir={self.habitat_save_path}",
                ],
            )

        def loading_config(test_name: str) -> DictConfig:
            return hydra.compose(
                config_name="test",
                overrides=[
                    f"test=graph_building/{test_name}",
                    f"test.config.logging.output_dir={self.habitat_save_path}",
                    f"test.config.model_name_or_path={self.model_load_path}",
                ],
            )

        with hydra.initialize(version_base=None, config_path="../../conf"):
            self.supervised_pre_training_cfg = training_config(
                "supervised_pre_training"
            )
            self.spth_feat_cfg = training_config("spth_feat")
            self.load_habitat_cfg = loading_config("load_habitat")
            self.load_habitat_for_ppf_cfg = loading_config("load_habitat_for_ppf")
            self.load_habitat_for_feat_cfg = loading_config("load_habitat_for_feat")

    def tearDown(self):
        """Code that gets executed after every test."""
        shutil.rmtree(self.output_dir)

    def check_graph_formatting(self, graph, features_to_check):
        """Makes sure graph contains right feature at location information."""
        self.assertIsNot(
            graph.pos,
            None,
            "graph contains no location information",
        )
        self.assertIsNot(
            graph.x,
            None,
            "graph contains no feature information",
        )
        self.assertIsNot(
            graph.feature_mapping,
            None,
            "graph contains no feature_mapping dict",
        )
        for feature in features_to_check:
            self.assertIn(
                feature,
                graph.feature_ids_in_graph,
                f"{feature} not stored in graph",
            )
        self.assertIn(
            "principal_curvatures",
            graph.feature_ids_in_graph,
            "curvature not stored in graph",
        )
        self.assertIn(
            "node_ids",
            graph.feature_ids_in_graph,
            "node ids not stored in graph",
        )

    def build_and_save_supervised_graph(
        self,
    ) -> MontySupervisedObjectPretrainingExperiment:
        """Builds and saves a supervised graph.

        Returns:
            The experiment.
        """
        exp = hydra.utils.instantiate(self.supervised_pre_training_cfg.test)
        with exp:
            exp.model.set_experiment_mode("train")
            exp.train()
        return exp

    def build_and_save_supervised_graph_feat(
        self,
    ) -> MontySupervisedObjectPretrainingExperiment:
        """Builds and saves a supervised graph with feature matching.

        Returns:
            The experiment.
        """
        exp = hydra.utils.instantiate(self.spth_feat_cfg.test)
        with exp:
            exp.model.set_experiment_mode("train")
            exp.train()
        return exp

    def test_get_correct_k_n(self):
        # enough data points sampled, just add 1 to remove self connection
        self.assertEqual(get_correct_k_n(5, 10), 6)
        # not enough points sampled, set k_n to num points - 1
        self.assertEqual(get_correct_k_n(5, 3), 2)
        # not enough points to make edges
        self.assertEqual(get_correct_k_n(5, 2), None)

    def test_can_build_graph_habitat_supervised(self):
        exp = self.build_and_save_supervised_graph()

        cfg_object_names = list(
            self.supervised_pre_training_cfg.test.config.train_env_interface_args.object_names
        )
        self.assertListEqual(
            cfg_object_names,
            exp.model.learning_modules[0].get_all_known_object_ids(),
            "Object ids of learned objects and graphs in memory.",
        )
        for graph_id in exp.model.learning_modules[0].get_all_known_object_ids():
            graph = exp.model.learning_modules[0].get_graph(
                graph_id, input_channel="first"
            )
            # Make sure that all features that are extracted by the SM are stored in
            # the graph.
            self.check_graph_formatting(
                graph,
                features_to_check=exp.model.sensor_modules[0].features,
            )
            self.assertIsNot(
                graph.edge_index,
                None,
                "graph contains no edges",
            )
            self.assertEqual(
                graph.edge_attr.shape[1],
                3,
                "Edge attributes don't store 3d displacements",
            )

    def test_can_load_disp_graph(self):
        self.build_and_save_supervised_graph()
        exp = hydra.utils.instantiate(self.load_habitat_cfg.test)
        with exp:
            for graph_id in exp.model.learning_modules[0].get_all_known_object_ids():
                graph = exp.model.learning_modules[0].get_graph(
                    graph_id, input_channel="first"
                )
                self.check_graph_formatting(
                    graph,
                    features_to_check=exp.model.sensor_modules[0].features,
                )
            exp.evaluate()

    def test_can_load_disp_graph_for_ppf_matching(self):
        self.build_and_save_supervised_graph()
        exp = hydra.utils.instantiate(self.load_habitat_for_ppf_cfg.test)
        with exp:
            for graph_id in exp.model.learning_modules[0].get_all_known_object_ids():
                graph = exp.model.learning_modules[0].get_graph(
                    graph_id, input_channel="first"
                )
                self.check_graph_formatting(
                    graph,
                    features_to_check=exp.model.sensor_modules[0].features,
                )
                self.assertEqual(
                    graph.edge_attr.shape[1],
                    4,
                    "Edge attributes don't store 4d PPF (should be added when loading)",
                )
            exp.evaluate()

    def test_can_load_disp_graph_for_feature_matching(self):
        self.build_and_save_supervised_graph()
        exp = hydra.utils.instantiate(self.load_habitat_for_feat_cfg.test)
        with exp:
            for graph_id in exp.model.learning_modules[0].get_all_known_object_ids():
                graph = exp.model.learning_modules[0].get_graph(
                    graph_id, input_channel="first"
                )
                self.check_graph_formatting(
                    graph,
                    features_to_check=exp.model.sensor_modules[0].features,
                )
                self.assertEqual(
                    graph.edge_attr.shape[1],
                    3,
                    "Edge attributes don't store 3d displacements",
                )
            exp.evaluate()

    def test_can_extend_and_save_feat_graph(self):
        self.build_and_save_supervised_graph_feat()
        exp = hydra.utils.instantiate(self.load_habitat_for_feat_cfg.test)
        with exp:
            for graph_id in exp.model.learning_modules[0].get_all_known_object_ids():
                graph = exp.model.learning_modules[0].get_graph(
                    graph_id, input_channel="first"
                )
                self.check_graph_formatting(
                    graph,
                    features_to_check=exp.model.sensor_modules[0].features,
                )
                # TODO: not sure if we want this check. Right now it doesn't but I
                # also don't see a reason why it couldn't in the future.
                self.assertIs(
                    graph.edge_attr,
                    None,
                    "feature at location graph should not contain edges.",
                )
            exp.train()


if __name__ == "__main__":
    unittest.main()
