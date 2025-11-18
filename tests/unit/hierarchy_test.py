# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy
import os
import shutil
import tempfile
import unittest
from pprint import pprint

import numpy as np
import pandas as pd
import pytest

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

from tbp.monty.frameworks.config_utils.config_args import (
    LoggingConfig,
    MontyArgs,
    PatchAndViewMontyConfig,
    PretrainLoggingConfig,
    TwoLMStackedMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_env_interface_configs import (
    EnvironmentInterfacePerObjectEvalArgs,
    EnvironmentInterfacePerObjectTrainArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
    SupervisedPretrainingExperimentArgs,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.experiments.pretraining_experiments import (
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.object_model import GridObjectModel
from tbp.monty.frameworks.utils.logging_utils import (
    load_models_from_dir,
)
from tbp.monty.simulators.habitat.configs import (
    EnvInitArgsPatchViewMount,
    EnvInitArgsTwoLMDistantStackedMount,
    PatchViewFinderMountHabitatEnvInterfaceConfig,
    TwoLMStackedDistantMountHabitatEnvInterfaceConfig,
)
from tests.unit.resources.unit_test_utils import BaseGraphTestCases


class HierarchyTest(BaseGraphTestCases.BaseGraphTest):
    def setUp(self):
        """Code that gets executed before every test."""
        super().setUp()

        self.output_dir = tempfile.mkdtemp()

        base = dict(
            experiment_class=MontyObjectRecognitionExperiment,
            experiment_args=ExperimentArgs(
                max_train_steps=30, max_eval_steps=30, max_total_steps=60
            ),
            logging_config=LoggingConfig(output_dir=self.output_dir),
            monty_config=PatchAndViewMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=20)
            ),
            env_interface_config=PatchViewFinderMountHabitatEnvInterfaceConfig(
                env_init_args=EnvInitArgsPatchViewMount(data_path=None).__dict__,
            ),
            train_env_interface_class=ED.InformedEnvironmentInterface,
            train_env_interface_args=EnvironmentInterfacePerObjectTrainArgs(
                object_names=["capsule3DSolid", "cubeSolid"],
                object_init_sampler=PredefinedObjectInitializer(),
            ),
            eval_env_interface_class=ED.InformedEnvironmentInterface,
            eval_env_interface_args=EnvironmentInterfacePerObjectEvalArgs(
                object_names=["capsule3DSolid"],
                object_init_sampler=PredefinedObjectInitializer(),
            ),
        )

        two_stacked_lms_config = dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.001,
                    tolerances={
                        "patch_0": {
                            "hsv": np.array([0.1, 1, 1]),
                            "principal_curvatures_log": np.ones(2),
                        }
                    },
                    feature_weights={},
                    max_graph_size=0.2,
                    num_model_voxels_per_dim=50,
                    max_nodes_per_graph=50,
                ),
            ),
            learning_module_1=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.001,
                    tolerances={
                        "patch_1": {
                            "hsv": np.array([0.1, 1, 1]),
                            "principal_curvatures_log": np.ones(2),
                        },
                        # object Id currently is an int representation of the strings
                        # in the object label so we keep this tolerance high. This is
                        # just until we have added a way to encode object ID with some
                        # real similarity measure.
                        "learning_module_0": {"object_id": 1},
                    },
                    feature_weights={"learning_module_0": {"object_id": 1}},
                    max_graph_size=0.3,
                    num_model_voxels_per_dim=50,
                    max_nodes_per_graph=50,
                ),
            ),
        )

        two_lms_heterarchy_config = copy.deepcopy(base)
        two_lms_heterarchy_config.update(
            experiment_args=ExperimentArgs(
                max_train_steps=30,
                max_eval_steps=30,
                max_total_steps=60,
                min_lms_match=2,
            ),
            monty_config=TwoLMStackedMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=100, min_train_steps=3),
                learning_module_configs=two_stacked_lms_config,
            ),
            env_interface_config=TwoLMStackedDistantMountHabitatEnvInterfaceConfig(
                env_init_args=EnvInitArgsTwoLMDistantStackedMount(
                    data_path=None
                ).__dict__,
            ),
        )

        two_stacked_constrained_lms_config = dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.001,
                    tolerances={
                        "patch_0": {
                            "hsv": np.array([0.1, 1, 1]),
                            "principal_curvatures_log": np.ones(2),
                        }
                    },
                    feature_weights={},
                    max_graph_size=0.3,
                    num_model_voxels_per_dim=200,
                    max_nodes_per_graph=2000,
                    object_evidence_threshold=20,
                ),
            ),
            learning_module_1=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    max_match_distance=0.001,
                    tolerances={
                        "patch_1": {
                            "hsv": np.array([0.1, 1, 1]),
                            "principal_curvatures_log": np.ones(2),
                        },
                        "learning_module_0": {"object_id": 1},
                    },
                    feature_weights={"learning_module_0": {"object_id": 1}},
                    max_graph_size=0.4,
                    num_model_voxels_per_dim=200,
                    max_nodes_per_graph=2000,
                ),
            ),
        )

        two_stacked_constrained_config = copy.deepcopy(base)
        two_stacked_constrained_config.update(
            experiment_args=SupervisedPretrainingExperimentArgs(
                max_train_steps=30,
                max_eval_steps=30,
                max_total_steps=60,
            ),
            logging_config=PretrainLoggingConfig(
                output_dir=self.output_dir,
                python_log_level="INFO",
            ),
            experiment_class=MontySupervisedObjectPretrainingExperiment,
            monty_config=TwoLMStackedMontyConfig(
                monty_args=MontyArgs(num_exploratory_steps=50),
                learning_module_configs=two_stacked_constrained_lms_config,
            ),
            env_interface_config=TwoLMStackedDistantMountHabitatEnvInterfaceConfig(
                env_init_args=EnvInitArgsTwoLMDistantStackedMount(
                    data_path=None,
                ).__dict__,
            ),
        )

        two_stacked_semisupervised_lms_config = copy.deepcopy(
            two_stacked_constrained_config
        )
        two_stacked_semisupervised_lms_config.update(
            experiment_args=SupervisedPretrainingExperimentArgs(
                supervised_lm_ids=["learning_module_1"],
                min_lms_match=2,
                model_name_or_path=os.path.join(self.output_dir, "pretrained"),
            ),
            monty_config=TwoLMStackedMontyConfig(
                # set min_train_steps to 200 to send more observations to LM_1 after
                # LM_0 has recognized the object.
                monty_args=MontyArgs(min_train_steps=200, num_exploratory_steps=0),
                learning_module_configs=two_stacked_constrained_lms_config,
            ),
        )

        two_stacked_lms_eval_config = copy.deepcopy(two_stacked_constrained_config)
        two_stacked_lms_eval_config.update(
            experiment_args=ExperimentArgs(
                do_train=False,
                min_lms_match=1,
                n_eval_epochs=2,
                model_name_or_path=os.path.join(self.output_dir, "pretrained"),
            ),
            logging_config=LoggingConfig(
                output_dir=self.output_dir,
                python_log_level="INFO",
            ),
            eval_env_interface_args=EnvironmentInterfacePerObjectEvalArgs(
                object_names=["capsule3DSolid"],
                object_init_sampler=PredefinedObjectInitializer(),
                parent_to_child_mapping={
                    "capsule3DSolid": "capsule3DSolid",
                    "cubeSolid": "cubeSolid",
                },
            ),
        )

        self.two_lms_heterarchy_config = two_lms_heterarchy_config
        self.two_stacked_constrained_config = two_stacked_constrained_config
        self.two_stacked_semisupervised_lms_config = (
            two_stacked_semisupervised_lms_config
        )
        self.two_stacked_lms_eval_config = two_stacked_lms_eval_config

    def tearDown(self):
        """Code that gets executed after every test."""
        shutil.rmtree(self.output_dir)

    def test_two_lm_heterarchy_experiment(self):
        """Test two LMs stacked on top of each other.

        LM0 receives input from SM0
        LM1 receives input from SM1 and LM0

        LM0 can store smaller models at a higher resolution and receives higher
        frequency input from SM0.
        LM1 can store larger models and a lower resolution and receives lower frequency
        input from SM1. It also receives input from LM0 once this one has a high
        confidence hypothesis.

        What happens in this experiment:
        Episodes 0-3: Both LMs have no_match and add a new model to memory.
        Episode 4: Both LMs recognize object 0 correctly and update their models.
        Episode 5: LM0 recognizes cubeSolid (new_object0) and updates its memory. LM1
            reaches a time out and does not update its memory (but has correct mlh).
        Evaluation:
            In each episode LM0 first recognizes the correct object. Since LM1 gets such
            low frequency input and stores very few points in its models it reaches
            no_match.

        NOTE: LM1 usually reaches no_match even if it knows about the object already. I
        think this is because for the first few observations it does not store features
        from LM0 yet. This would be different with a longer exploration phase that
        builds a full model of the object.

        NOTE: This test tests a lot of different things. We could split it up into many
        separate tests and test each aspect independently. However, this would increase
        computational cost since for many tests (like extending a graph correctly or
        getting the LM input) several episodes need to be run first (to build up graphs
        from which the object can be recognized in the first place). We could use mock
        data and test the LM in isolation like we already do in some places but we
        would still want to test the whole pipeline at least once. So why not make use
        of this longer run if we already have it? Maybe in the future we want to change
        this but this is my current reasoning.
        """
        pprint("...parsing experiment...")
        config = copy.deepcopy(self.two_lms_heterarchy_config)
        with MontyObjectRecognitionExperiment(config) as exp:
            pprint("...training...")
            exp.train()
            train_stats = pd.read_csv(os.path.join(exp.output_dir, "train_stats.csv"))
            self.check_hierarchical_lm_train_results(train_stats)

            models = load_models_from_dir(exp.output_dir)
            self.check_hierarchical_models(models)

            pprint("...evaluating...")
            exp.evaluate()
            eval_stats = pd.read_csv(os.path.join(exp.output_dir, "eval_stats.csv"))
            self.check_hierarchical_lm_eval_results(eval_stats)

    def test_semisupervised_stacked_lms_experiment(self):
        """Test two LMs stacked on top of each other with semisupervised learning.

        First, both LMs learn with supervision. Then, we remove the supervision for LM_0
        and only supervised LM_1 to learn a compositional object.
        Last, we load the learned models and evaluate on them.

        NOTE: It's not ideal to test compositional scenarios since we don't have access
        to compositional objects in the unit tests. So we are pretending that the cube
        and capsule are compositional and extending the graphs with the classification
        from LM_0 of the same object.

        NOTE: This test also implicitly tests a bunch of other things, such as:
        - Extending grid object models after loading them
        - Extending a graph with a new input channel
        - logging prediction errors
        """
        pprint("...supervised training...")
        config = copy.deepcopy(self.two_stacked_constrained_config)
        with MontySupervisedObjectPretrainingExperiment(config) as exp:
            exp.model.set_experiment_mode("train")
            exp.train()
            # check that both LMs have learned both objects.
            for lm_idx, lm in enumerate(exp.model.learning_modules):
                learned_objects = lm.get_all_known_object_ids()
                self.assertIn(
                    "capsule3DSolid",
                    learned_objects,
                    f"capsule3DSolid not in learned objects for LM {lm_idx}. "
                    f"Learned objects: {learned_objects}",
                )
                self.assertIn(
                    "cubeSolid",
                    learned_objects,
                    f"cubeSolid not in learned objects for LM {lm_idx}. Learned "
                    f"objects: {learned_objects}",
                )

        pprint("...semisupervised training...")
        config = copy.deepcopy(self.two_stacked_semisupervised_lms_config)
        with MontySupervisedObjectPretrainingExperiment(config) as exp:
            exp.model.set_experiment_mode("train")
            # check that models for both objects are loaded into memory correctly.
            for lm_idx, lm in enumerate(exp.model.learning_modules):
                for object_id in ["capsule3DSolid", "cubeSolid"]:
                    self.assertIn(
                        object_id,
                        lm.graph_memory.get_all_models_in_memory(),
                    )
                    # check that the correct input channel is present.
                    loaded_graph = lm.graph_memory.get_graph(graph_id=object_id)
                    self.assertIn(f"patch_{lm_idx}", loaded_graph.keys())
                    # check that it is of type GridObjectModel.
                    self.assertIsInstance(
                        loaded_graph[f"patch_{lm_idx}"], GridObjectModel
                    )
            lm_0_memory_before_learning = exp.model.learning_modules[
                0
            ].graph_memory.get_all_models_in_memory()
            exp.train()
            # check that LM_0 models were not updated
            for object_id in ["capsule3DSolid", "cubeSolid"]:
                updated_graph = exp.model.learning_modules[0].graph_memory.get_graph(
                    graph_id=object_id
                )
                self.assertEqual(updated_graph, lm_0_memory_before_learning[object_id])
            # check that LM_1 models now contain learning_module_0 input channel.
            # TODO: also get it to recognize cubeSolid
            for object_id in ["capsule3DSolid"]:
                updated_graph = exp.model.learning_modules[1].graph_memory.get_graph(
                    graph_id=object_id
                )
                self.assertIn(
                    "learning_module_0",
                    updated_graph.keys(),
                    f"learning_module_0 not in updated graph for {object_id}. Updated "
                    f"graph: {updated_graph} with keys: {updated_graph.keys()}",
                )

        pprint("...evaluating LM with compositional models...")
        config = copy.deepcopy(self.two_stacked_lms_eval_config)
        with MontyObjectRecognitionExperiment(config) as exp:
            exp.evaluate()
            pprint("... loading and checking eval statistics...")
            eval_stats = pd.read_csv(os.path.join(exp.output_dir, "eval_stats.csv"))
            episode = 0
            num_lms = len(exp.model.learning_modules)
            for lm_id in range(num_lms):
                self.assertIn(
                    eval_stats["primary_performance"][episode * 2 + lm_id],
                    ["correct", "correct_mlh"],
                    f"LM {lm_id} did not recognize the object on first episode.",
                )
            episode = 1
            for lm_id in range(num_lms):
                self.assertEqual(
                    "no_match",
                    eval_stats["primary_performance"][episode * 2 + lm_id],
                    "LMs should not recognize object on second episode as it is a "
                    "previously unseen view.",
                )
            # check that prediction errors are logged
            self.assertIn(
                "episode_avg_prediction_error",
                eval_stats.columns,
                "Prediction error is not logged.",
            )


if __name__ == "__main__":
    unittest.main()
