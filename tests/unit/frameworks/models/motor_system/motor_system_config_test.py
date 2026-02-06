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
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.motor_policies import BasePolicy
from tbp.monty.frameworks.models.motor_system import MotorSystem
from tests import HYDRA_ROOT


class MotorSystemConfigTest(unittest.TestCase):
    """Test for the motor system configuration instantiation of objects.

    These tests ensure that Hydra instantiates the MotorSystem objects we want in the
    same way that the old *_class, *_args pattern would.
    """

    MOTOR_SYSTEM_CONFIG = (
        HYDRA_ROOT / "experiment" / "config" / "monty" / "motor_system"
    )

    def setUp(self):
        with hydra.initialize_config_dir(
            config_dir=str(self.MOTOR_SYSTEM_CONFIG), version_base=None
        ):
            self.motor_system_cfg = hydra.compose(config_name="defaults")
            self.motor_system = hydra.utils.instantiate(self.motor_system_cfg)

    def test_default_config_instantiates_action_sampler(self):
        """Test that the default motor system config instantiates the ActionSampler."""
        motor_policy = self.motor_system["motor_system_args"]["policy"]
        action_sampler = motor_policy.action_sampler
        self.assertIsInstance(action_sampler, ActionSampler)
        # These values come from "action_space/distant_agent.yaml"
        self.assertListEqual(
            action_sampler._action_names,
            ["look_up", "look_down", "turn_left", "turn_right"],
        )

    def test_default_config_instantiates_motor_policy(self):
        """Test the default motor system config instantiates the MotorPolicy object."""
        motor_policy = self.motor_system["motor_system_args"]["policy"]

        self.assertIsInstance(motor_policy, BasePolicy)
        self.assertEqual(motor_policy.agent_id, AgentID("agent_id_0"))

    def test_default_config_motor_system_takes_motor_policy_instance(self):
        """Test that the motor system receives the same motor policy.

        The motor system should be created with the same instance of the MotorPolicy
        that the configuration instantiates for us.
        """
        motor_policy = self.motor_system["motor_system_args"]["policy"]

        # TODO - change when we switch to instantiating the policy directly
        motor_system_class = self.motor_system["motor_system_class"]
        motor_system_args = self.motor_system["motor_system_args"]
        motor_system: MotorSystem = motor_system_class(**motor_system_args)

        self.assertIs(motor_system._policy, motor_policy)
