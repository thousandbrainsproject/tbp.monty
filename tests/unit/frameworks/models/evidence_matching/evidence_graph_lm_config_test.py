# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import unittest

import hydra
import numpy as np
import pytest

from tbp.monty.frameworks.models.goal_state_generation import (
    EvidenceGoalStateGenerator,
    ParentLMNotProvided,
)
from tests import HYDRA_ROOT


class EvidenceGraphLMConfigTest(unittest.TestCase):
    """Test for the EvidenceGraphLM configuration instantiation of objects.

    These tests ensure that Hydra instantiates the EvidenceGraphLM objects we want
    in the same way that the old *_class, *_args pattern would.
    """

    LM_CONFIGS = (
        HYDRA_ROOT
        / "experiment"
        / "config"
        / "monty"
        / "learning_modules"
        / "learning_module"
    )

    def setUp(self):
        # This can't be setUpClass because using the GSG with an LM modifies both
        with hydra.initialize_config_dir(
            config_dir=str(self.LM_CONFIGS), version_base=None
        ):
            self.lm_config = hydra.compose(config_name="default_evidence")
            self.learning_module = hydra.utils.instantiate(self.lm_config)

    def test_default_config_instantiates_goal_state_generator(self):
        """Test that the default config instantiates the GSG correctly."""
        gsg = self.learning_module["learning_module_args"]["gsg"]

        self.assertIsInstance(gsg, EvidenceGoalStateGenerator)
        # Check a few values from the config
        self.assertEqual(gsg.elapsed_steps_factor, 10)
        self.assertEqual(gsg.min_post_goal_success_steps, 5)
        self.assertEqual(gsg.x_percent_scale_factor, 0.75)
        self.assertEqual(gsg.desired_object_distance, 0.03)

    def test_goal_state_generator_instance_is_created_without_parent_lm(self):
        """Test that the GSG instantiated by the config doesn't have the parent LM.

        We're moving to a two-step initialization for GSGs, so without the extra step
        in the parent LM to introduce itself to the GSG, the GSG shouldn't have a
        parent_lm set.
        """
        gsg = self.learning_module["learning_module_args"]["gsg"]

        self.assertIsInstance(gsg, EvidenceGoalStateGenerator)
        with pytest.raises(ParentLMNotProvided):
            gsg.parent_lm()

    def test_default_config_lm_takes_goal_state_generator_instance(self):
        """Test that the LM receives the same GSG.

        The created learning module should be created with the same instance of GSG
        that the configuration instantiates for us.
        """
        gsg = self.learning_module["learning_module_args"]["gsg"]
        # TODO: change when we switch to instantiating the LM directly
        learning_module_class = self.learning_module["learning_module_class"]
        learning_module_args = self.learning_module["learning_module_args"]
        rng = np.random.default_rng(42)
        lm = learning_module_class(rng=rng, **learning_module_args)

        self.assertIs(lm.gsg, gsg)
        self.assertIs(gsg.parent_lm, lm)
