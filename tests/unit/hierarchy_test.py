# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import pytest

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

import shutil
import tempfile
import unittest
from pathlib import Path

import hydra
import pandas as pd

from tbp.monty.frameworks.models.object_model import GridObjectModel
from tbp.monty.frameworks.utils.logging_utils import (
    load_models_from_dir,
)
from tests.unit.resources.unit_test_utils import BaseGraphTest


class HierarchyTest(BaseGraphTest):
    def setUp(self):
        """Code that gets executed before every test."""
        super().setUp()

        self.output_dir = Path(tempfile.mkdtemp())
        self.model_path = self.output_dir / "pretrained"

        with hydra.initialize(version_base=None, config_path="../../conf"):
            self.two_lms_heterarchy_cfg = hydra.compose(
                config_name="test",
                overrides=[
                    "test=hierarchy/two_lms_heterarchy",
                    f"test.config.logging.output_dir={self.output_dir}",
                ],
            )
            self.two_lms_constrained_cfg = hydra.compose(
                config_name="test",
                overrides=[
                    "test=hierarchy/two_lms_constrained",
                    f"test.config.logging.output_dir={self.output_dir}",
                ],
            )
            self.two_lms_semisupervised_cfg = hydra.compose(
                config_name="test",
                overrides=[
                    "test=hierarchy/two_lms_semisupervised",
                    f"test.config.logging.output_dir={self.output_dir}",
                    f"test.config.model_name_or_path={self.model_path}",
                ],
            )
            self.two_lms_eval_cfg = hydra.compose(
                config_name="test",
                overrides=[
                    "test=hierarchy/two_lms_eval",
                    f"test.config.logging.output_dir={self.output_dir}",
                    f"test.config.model_name_or_path={self.model_path}",
                ],
            )

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
        exp = hydra.utils.instantiate(self.two_lms_heterarchy_cfg.test)
        with exp:
            exp.train()
            output_dir = Path(exp.output_dir)
            train_stats = pd.read_csv(output_dir / "train_stats.csv")
            self.check_hierarchical_lm_train_results(train_stats)

            models = load_models_from_dir(output_dir)
            self.check_hierarchical_models(models)

            exp.evaluate()
            eval_stats = pd.read_csv(output_dir / "eval_stats.csv")
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
        exp = hydra.utils.instantiate(self.two_lms_constrained_cfg.test)
        with exp:
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

        exp = hydra.utils.instantiate(self.two_lms_semisupervised_cfg.test)
        with exp:
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

        exp = hydra.utils.instantiate(self.two_lms_eval_cfg.test)
        with exp:
            exp.evaluate()
            eval_stats = pd.read_csv(Path(exp.output_dir) / "eval_stats.csv")
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
