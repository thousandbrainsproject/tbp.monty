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

from tbp.monty.frameworks.actions.action_samplers import ActionSampler
from tests import HYDRA_ROOT


class MotorSystemConfigTest(unittest.TestCase):
    """Test for the motor system configuration instantiation of objects.

    These tests ensure that Hydra instantiates the MotorSystem objects we want in the
    same way that the old *_class, *_args pattern would.
    """

    MOTOR_SYSTEM_CONFIG = (
        HYDRA_ROOT / "experiment" / "config" / "monty" / "motor_system"
    )

    @classmethod
    def setUpClass(cls):
        with hydra.initialize_config_dir(
            config_dir=str(cls.MOTOR_SYSTEM_CONFIG), version_base=None
        ):
            cls.motor_system_cfg = hydra.compose(config_name="defaults")
            cls.motor_system = hydra.utils.instantiate(cls.motor_system_cfg)

    def test_default_config_instantiates_action_sampler(self):
        """Test that the default motor system config instantiates the ActionSampler."""
        action_sampler = self.motor_system["motor_system_args"]["policy_args"][
            "action_sampler"
        ]
        self.assertIsInstance(action_sampler, ActionSampler)
        # These values come from "action_space/distant_agent.yaml"
        self.assertListEqual(
            action_sampler._action_names,
            ["look_up", "look_down", "turn_left", "turn_right"],
        )

    def test_default_config_motor_policy_takes_action_sampler_instance(self):
        """Test that the motor policy receives the same ActionSampler.

        The motor policy should be created with the same instance of the ActionSampler
        that the configuration instantiates for us.
        """
        action_sampler = self.motor_system["motor_system_args"]["policy_args"][
            "action_sampler"
        ]
        # TODO - change when we switch to instantiating the policy directly
        motor_policy_class = self.motor_system["motor_system_args"]["policy_class"]
        motor_policy_args = self.motor_system["motor_system_args"]["policy_args"]
        motor_policy = motor_policy_class(**motor_policy_args)

        self.assertIs(motor_policy.action_sampler, action_sampler)
