# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy
import shutil
import tempfile
import unittest
from pprint import pprint

import numpy as np

from tbp.monty.frameworks.config_utils.config_args import (
    LoggingConfig,
    MontyArgs,
    MontyFeatureGraphArgs,
    PatchAndViewMontyConfig,
    PretrainLoggingConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataLoaderPerObjectTrainArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import (
    MontyObjectRecognitionExperiment,
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.frameworks.models.displacement_matching import DisplacementGraphLM
from tbp.monty.frameworks.models.feature_location_matching import FeatureGraphLM
from tbp.monty.frameworks.utils.graph_matching_utils import get_correct_k_n
from tbp.monty.simulators.habitat.configs import (
    EnvInitArgsPatchViewMount,
    PatchViewFinderMountHabitatDatasetArgs,
)


class GraphLearningTest(unittest.TestCase):
    def setUp(self):
        """Code that gets executed before every test."""
        self.output_dir = tempfile.mkdtemp()
        self.habitat_save_path = tempfile.mkdtemp()
        self.mesh_save_path = tempfile.mkdtemp()

        self.habitat_learned_rotations = [[0.0, 0.0, 0.0], [0.0, 45.0, 0.0]]
        self.supervised_pre_training_in_habitat = dict(
            experiment_class=MontySupervisedObjectPretrainingExperiment,
            experiment_args=ExperimentArgs(
                do_eval=False,
                n_train_epochs=len(self.habitat_learned_rotations),
            ),
            logging_config=PretrainLoggingConfig(output_dir=self.habitat_save_path),
            monty_config=PatchAndViewMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=10),
                learning_module_configs=dict(
                    learning_module_0=dict(
                        learning_module_class=DisplacementGraphLM,
                        learning_module_args=dict(
                            k=5,
                            match_attribute="displacement",
                        ),
                    )
                ),
            ),
            dataset_class=ED.EnvironmentDataset,
            dataset_args=PatchViewFinderMountHabitatDatasetArgs(
                env_init_args=EnvInitArgsPatchViewMount(data_path=None).__dict__,
            ),
            train_dataloader_class=ED.InformedEnvironmentDataLoader,
            train_dataloader_args=EnvironmentDataLoaderPerObjectTrainArgs(
                object_names=["capsule3DSolid", "cubeSolid"],
                object_init_sampler=PredefinedObjectInitializer(
                    rotations=self.habitat_learned_rotations
                ),
            ),
            eval_dataloader_class=ED.InformedEnvironmentDataLoader,
            eval_dataloader_args=EnvironmentDataLoaderPerObjectTrainArgs(
                object_names=[],
                object_init_sampler=PredefinedObjectInitializer(),
            ),
        )

        self.load_habitat_config = dict(
            experiment_class=MontyObjectRecognitionExperiment,
            experiment_args=ExperimentArgs(
                model_name_or_path=self.habitat_save_path + "/pretrained",
            ),
            logging_config=LoggingConfig(output_dir=self.output_dir),
            monty_config=PatchAndViewMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=20),
                learning_module_configs=dict(
                    learning_module_0=dict(
                        learning_module_class=DisplacementGraphLM,
                        learning_module_args=dict(
                            k=5,
                            match_attribute="displacement",
                            tolerance=np.ones(3) * 0.0001,
                        ),
                    )
                ),
            ),
            dataset_class=ED.EnvironmentDataset,
            dataset_args=PatchViewFinderMountHabitatDatasetArgs(
                env_init_args=EnvInitArgsPatchViewMount(data_path=None).__dict__,
            ),
            train_dataloader_class=ED.InformedEnvironmentDataLoader,
            train_dataloader_args=EnvironmentDataLoaderPerObjectTrainArgs(
                object_names=["capsule3DSolid", "cubeSolid"],
                object_init_sampler=PredefinedObjectInitializer(
                    rotations=self.habitat_learned_rotations
                ),
            ),
            eval_dataloader_class=ED.InformedEnvironmentDataLoader,
            eval_dataloader_args=EnvironmentDataLoaderPerObjectTrainArgs(
                object_names=["capsule3DSolid", "cubeSolid"],
                object_init_sampler=PredefinedObjectInitializer(),
            ),
        )

        self.load_habitat_for_ppf = copy.deepcopy(self.load_habitat_config)
        self.load_habitat_for_ppf.update(
            monty_config=PatchAndViewMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=20),
                learning_module_configs=dict(
                    learning_module_0=dict(
                        learning_module_class=DisplacementGraphLM,
                        learning_module_args=dict(
                            k=5,
                            match_attribute="PPF",
                            tolerance=np.ones(4) * 0.001,
                        ),
                    )
                ),
            ),
        )

        self.load_habitat_for_feat = copy.deepcopy(self.load_habitat_config)
        self.load_habitat_for_feat.update(
            monty_config=PatchAndViewMontyConfig(
                monty_args=MontyFeatureGraphArgs(num_exploratory_steps=20),
                learning_module_configs=dict(
                    learning_module_0=dict(
                        learning_module_class=FeatureGraphLM,
                        learning_module_args=dict(
                            max_match_distance=0.01,
                            tolerances={
                                "patch": {
                                    "on_object": 0,
                                    "rgba": np.ones(4) * 10,
                                    "principal_curvatures": np.ones(2) * 5,
                                    "pose_vectors": [
                                        0.7,
                                        np.pi * 2,
                                        np.pi * 2,
                                    ],  # angular difference
                                }
                            },
                        ),
                    )
                ),
            ),
        )

        self.spth_feat = copy.deepcopy(self.supervised_pre_training_in_habitat)
        self.spth_feat.update(
            monty_config=PatchAndViewMontyConfig(
                monty_args=MontyFeatureGraphArgs(num_exploratory_steps=10),
                learning_module_configs=dict(
                    learning_module_0=dict(
                        learning_module_class=FeatureGraphLM,
                        learning_module_args=dict(
                            max_match_distance=None,  # Not matching when supervised
                            tolerances={"patch": {}},
                        ),
                    )
                ),
            ),
        )

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

    def build_and_save_supervised_graph(self):
        pprint("...parsing experiment...")
        self.exp = MontySupervisedObjectPretrainingExperiment()
        with self.exp:
            self.exp.setup_experiment(self.supervised_pre_training_in_habitat)
            self.exp.model.set_experiment_mode("train")

            pprint("...training...")
            self.exp.train()

    def build_and_save_supervised_graph_feat(self):
        pprint("...parsing experiment...")
        self.exp = MontySupervisedObjectPretrainingExperiment()
        with self.exp:
            self.exp.setup_experiment(self.spth_feat)
            self.exp.model.set_experiment_mode("train")

            pprint("...training...")
            self.exp.train()

    def test_get_correct_k_n(self):
        # enough data points sampled, just add 1 to remove self connection
        self.assertEqual(get_correct_k_n(5, 10), 6)
        # not enough points sampled, set k_n to num points - 1
        self.assertEqual(get_correct_k_n(5, 3), 2)
        # not enough points to make edges
        self.assertEqual(get_correct_k_n(5, 2), None)

    def test_can_build_graph_habitat_supervised(self):
        self.build_and_save_supervised_graph()
        pprint("...Checking graphs...")

        self.assertListEqual(
            self.supervised_pre_training_in_habitat[
                "train_dataloader_args"
            ].object_names,
            self.exp.model.learning_modules[0].get_all_known_object_ids(),
            "Object ids of learned objects and graphs in memory.",
        )
        for graph_id in self.exp.model.learning_modules[0].get_all_known_object_ids():
            graph = self.exp.model.learning_modules[0].get_graph(
                graph_id, input_channel="first"
            )
            # Make sure that all features that are extracted by the SM are stored in
            # the graph.
            self.check_graph_formatting(
                graph,
                features_to_check=self.exp.model.sensor_modules[0].features,
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
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.load_habitat_config)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(config)
            pprint("checking loaded graphs")
            for graph_id in self.exp.model.learning_modules[
                0
            ].get_all_known_object_ids():
                graph = self.exp.model.learning_modules[0].get_graph(
                    graph_id, input_channel="first"
                )
                self.check_graph_formatting(
                    graph,
                    features_to_check=self.exp.model.sensor_modules[0].features,
                )
            pprint("...evaluating on loaded models...")
            self.exp.evaluate()

    def test_can_load_disp_graph_for_ppf_matching(self):
        self.build_and_save_supervised_graph()
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.load_habitat_for_ppf)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(config)
            pprint("checking loaded graphs")
            for graph_id in self.exp.model.learning_modules[
                0
            ].get_all_known_object_ids():
                graph = self.exp.model.learning_modules[0].get_graph(
                    graph_id, input_channel="first"
                )
                self.check_graph_formatting(
                    graph,
                    features_to_check=self.exp.model.sensor_modules[0].features,
                )
                self.assertEqual(
                    graph.edge_attr.shape[1],
                    4,
                    "Edge attributes don't store 4d PPF (should be added when loading)",
                )
            pprint("...evaluating on loaded models...")
            self.exp.evaluate()

    def test_can_load_disp_graph_for_feature_matching(self):
        self.build_and_save_supervised_graph()
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.load_habitat_for_feat)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(config)
            pprint("checking loaded graphs")
            for graph_id in self.exp.model.learning_modules[
                0
            ].get_all_known_object_ids():
                graph = self.exp.model.learning_modules[0].get_graph(
                    graph_id, input_channel="first"
                )
                self.check_graph_formatting(
                    graph,
                    features_to_check=self.exp.model.sensor_modules[0].features,
                )
                self.assertEqual(
                    graph.edge_attr.shape[1],
                    3,
                    "Edge attributes don't store 3d displacements",
                )
            pprint("...evaluating on loaded models...")
            self.exp.evaluate()

    def test_can_extend_and_save_feat_graph(self):
        self.build_and_save_supervised_graph_feat()
        config = copy.deepcopy(self.load_habitat_for_feat)
        self.exp = MontyObjectRecognitionExperiment()
        with self.exp:
            self.exp.setup_experiment(config)
            pprint("checking loaded graphs")
            for graph_id in self.exp.model.learning_modules[
                0
            ].get_all_known_object_ids():
                graph = self.exp.model.learning_modules[0].get_graph(
                    graph_id, input_channel="first"
                )
                self.check_graph_formatting(
                    graph,
                    features_to_check=self.exp.model.sensor_modules[0].features,
                )
                # TODO: not sure if we want this check. Right now it doesn't but I
                # also don't see a reason why it couldn't in the future.
                self.assertIs(
                    graph.edge_attr,
                    None,
                    "feature at location graph should not contain edges.",
                )
            pprint("...evaluating on loaded models...")
            self.exp.train()


if __name__ == "__main__":
    unittest.main()
