# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import pytest

from tbp.monty.frameworks.agents import AgentID

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

import unittest
from typing import Any
from unittest.mock import Mock, patch

from habitat_sim import ActionSpec, Agent, AgentConfiguration
from typing_extensions import override

from tbp.monty.frameworks.actions.actions import (
    LookDown,
    LookUp,
    MoveForward,
    MoveTangentially,
    OrientHorizontal,
    OrientVertical,
    SetAgentPitch,
    SetAgentPose,
    SetSensorPitch,
    SetSensorPose,
    SetSensorRotation,
    SetYaw,
    TurnLeft,
    TurnRight,
)
from tbp.monty.simulators.habitat.actuator import (
    HabitatActuator,
    InvalidActionName,
)
from tests.unit.frameworks.actions.fakes.action import FakeAction


class FakeHabitat(HabitatActuator):
    @override
    def get_agent(self, agent_id: AgentID) -> Agent | None:
        return None


class HabitatAcutatorTest(unittest.TestCase):
    def test_action_name_concatenates_agent_id_and_name_with_period(self) -> None:
        actuator = FakeHabitat()
        action = FakeAction(agent_id=AgentID("agent1"))
        self.assertEqual(actuator.action_name(action), "agent1.fake_action")

    @patch("tests.unit.frameworks.actions.habitat.actuator_test.FakeHabitat.get_agent")
    def test_to_habitat_raises_value_error_if_action_name_not_in_action_space(
        self, mock_get_agent: Mock
    ) -> None:
        mock_agent = Mock(spec=Agent)
        mock_agent.agent_config = AgentConfiguration(action_space={})
        mock_get_agent.return_value = mock_agent

        actuator = FakeHabitat()
        action = FakeAction(agent_id=AgentID("agent1"))

        with self.assertRaises(InvalidActionName) as context:
            actuator.to_habitat(action)

        self.assertEqual(
            str(context.exception),
            "Invalid action name: agent1.fake_action",
        )


@patch("tests.unit.frameworks.actions.habitat.actuator_test.FakeHabitat.get_agent")
class HabitatActuatorsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.action_space: dict[Any, ActionSpec] = {}
        mock_agent_config = Mock(spec=AgentConfiguration)
        mock_agent_config.action_space = self.action_space
        self.mock_agent = Mock(spec=Agent)
        self.mock_agent.agent_config = mock_agent_config
        self.actuator = FakeHabitat()

    def test_actuate_look_down_acts_with_params_set(self, mock_get_agent: Mock) -> None:
        mock_get_agent.return_value = self.mock_agent

        action = LookDown(
            agent_id=AgentID("agent1"), rotation_degrees=45, constraint_degrees=90
        )
        action_name = self.actuator.action_name(action)
        self.action_space[action_name] = Mock()

        self.actuator.actuate_look_down(action)

        self.mock_agent.act.assert_called_once_with(action_name)
        self.assertEqual(self.action_space[action_name].actuation.amount, 45)
        self.assertEqual(self.action_space[action_name].actuation.constraint, 90)

    def test_actuate_look_up_acts_with_params_set(self, mock_get_agent: Mock) -> None:
        mock_get_agent.return_value = self.mock_agent

        action = LookUp(
            agent_id=AgentID("agent1"), rotation_degrees=45, constraint_degrees=90
        )
        action_name = self.actuator.action_name(action)
        self.action_space[action_name] = Mock()

        self.actuator.actuate_look_up(action)

        self.mock_agent.act.assert_called_once_with(action_name)
        self.assertEqual(self.action_space[action_name].actuation.amount, 45)
        self.assertEqual(self.action_space[action_name].actuation.constraint, 90)

    def test_actuate_move_forward_acts_with_amount_set_to_distance(
        self, mock_get_agent: Mock
    ) -> None:
        mock_get_agent.return_value = self.mock_agent

        action = MoveForward(agent_id=AgentID("agent1"), distance=1)
        action_name = self.actuator.action_name(action)
        self.action_space[action_name] = Mock()

        self.actuator.actuate_move_forward(action)

        self.mock_agent.act.assert_called_once_with(action_name)
        self.assertEqual(self.action_space[action_name].actuation.amount, 1)

    def test_actuate_move_tangentially_acts_with_params_set(
        self, mock_get_agent: Mock
    ) -> None:
        mock_get_agent.return_value = self.mock_agent

        action = MoveTangentially(
            agent_id=AgentID("agent1"), distance=1, direction=(1, 0, 0)
        )
        action_name = self.actuator.action_name(action)
        self.action_space[action_name] = Mock()

        self.actuator.actuate_move_tangentially(action)

        self.mock_agent.act.assert_called_once_with(action_name)
        self.assertEqual(self.action_space[action_name].actuation.amount, 1)
        self.assertEqual(self.action_space[action_name].actuation.constraint, (1, 0, 0))

    def test_actuate_orient_horizontal_acts_with_params_set(
        self, mock_get_agent: Mock
    ) -> None:
        mock_get_agent.return_value = self.mock_agent

        action = OrientHorizontal(
            agent_id=AgentID("agent1"),
            rotation_degrees=45,
            left_distance=1,
            forward_distance=2,
        )
        action_name = self.actuator.action_name(action)
        self.action_space[action_name] = Mock()

        self.actuator.actuate_orient_horizontal(action)

        self.mock_agent.act.assert_called_once_with(action_name)
        self.assertEqual(self.action_space[action_name].actuation.amount, 45)
        self.assertEqual(self.action_space[action_name].actuation.constraint, [1, 2])

    def test_actuate_orient_vertical_acts_with_params_set(
        self, mock_get_agent: Mock
    ) -> None:
        mock_get_agent.return_value = self.mock_agent

        action = OrientVertical(
            agent_id=AgentID("agent1"),
            rotation_degrees=45,
            down_distance=1,
            forward_distance=2,
        )
        action_name = self.actuator.action_name(action)
        self.action_space[action_name] = Mock()

        self.actuator.actuate_orient_vertical(action)

        self.mock_agent.act.assert_called_once_with(action_name)
        self.assertEqual(self.action_space[action_name].actuation.amount, 45)
        self.assertEqual(self.action_space[action_name].actuation.constraint, [1, 2])

    def test_actuate_set_agent_pitch_acts_with_amount_set_to_pitch_degrees(
        self, mock_get_agent: Mock
    ) -> None:
        mock_get_agent.return_value = self.mock_agent

        action = SetAgentPitch(agent_id=AgentID("agent1"), pitch_degrees=45)
        action_name = self.actuator.action_name(action)
        self.action_space[action_name] = Mock()

        self.actuator.actuate_set_agent_pitch(action)

        self.mock_agent.act.assert_called_once_with(action_name)
        self.assertEqual(self.action_space[action_name].actuation.amount, 45)

    def test_actuate_set_agent_pose_acts_with_params_set(
        self, mock_get_agent: Mock
    ) -> None:
        mock_get_agent.return_value = self.mock_agent

        action = SetAgentPose(
            agent_id=AgentID("agent1"), location=(1, 2, 3), rotation_quat=(0, 0, 0, 1)
        )
        action_name = self.actuator.action_name(action)
        self.action_space[action_name] = Mock()

        self.actuator.actuate_set_agent_pose(action)

        self.mock_agent.act.assert_called_once_with(action_name)
        self.assertEqual(
            self.action_space[action_name].actuation.amount, [(1, 2, 3), (0, 0, 0, 1)]
        )

    def test_actuate_set_sensor_pitch_acts_with_amount_set_to_pitch_degrees(
        self, mock_get_agent: Mock
    ) -> None:
        mock_get_agent.return_value = self.mock_agent

        action = SetSensorPitch(agent_id=AgentID("agent1"), pitch_degrees=44)
        action_name = self.actuator.action_name(action)
        self.action_space[action_name] = Mock()

        self.actuator.actuate_set_sensor_pitch(action)

        self.mock_agent.act.assert_called_once_with(action_name)
        self.assertEqual(self.action_space[action_name].actuation.amount, 44)

    def test_actuate_set_sensor_pose_acts_with_params_set(
        self, mock_get_agent: Mock
    ) -> None:
        mock_get_agent.return_value = self.mock_agent

        action = SetSensorPose(
            agent_id=AgentID("agent1"), location=(1, 2, 3), rotation_quat=(0, 0, 0, 1)
        )
        action_name = self.actuator.action_name(action)
        self.action_space[action_name] = Mock()

        self.actuator.actuate_set_sensor_pose(action)

        self.mock_agent.act.assert_called_once_with(action_name)
        self.assertEqual(
            self.action_space[action_name].actuation.amount, [(1, 2, 3), (0, 0, 0, 1)]
        )

    def test_actuate_set_sensor_rotation_acts_with_params_set(
        self, mock_get_agent: Mock
    ) -> None:
        mock_get_agent.return_value = self.mock_agent

        action = SetSensorRotation(
            agent_id=AgentID("agent1"), rotation_quat=(0, 0, 0, 1)
        )
        action_name = self.actuator.action_name(action)
        self.action_space[action_name] = Mock()

        self.actuator.actuate_set_sensor_rotation(action)

        self.mock_agent.act.assert_called_once_with(action_name)
        self.assertEqual(
            self.action_space[action_name].actuation.amount, [(0, 0, 0, 1)]
        )

    def test_actuate_set_yaw_acts_with_amount_set_to_rotation_degrees(
        self, mock_get_agent: Mock
    ) -> None:
        mock_get_agent.return_value = self.mock_agent

        action = SetYaw(agent_id=AgentID("agent1"), rotation_degrees=41)
        action_name = self.actuator.action_name(action)
        self.action_space[action_name] = Mock()

        self.actuator.actuate_set_yaw(action)

        self.mock_agent.act.assert_called_once_with(action_name)
        self.assertEqual(self.action_space[action_name].actuation.amount, 41)

    def test_actuate_turn_left_acts_with_amount_set_to_rotation_degrees(
        self, mock_get_agent: Mock
    ) -> None:
        mock_get_agent.return_value = self.mock_agent

        action = TurnLeft(agent_id=AgentID("agent1"), rotation_degrees=40)
        action_name = self.actuator.action_name(action)
        self.action_space[action_name] = Mock()

        self.actuator.actuate_turn_left(action)

        self.mock_agent.act.assert_called_once_with(action_name)
        self.assertEqual(self.action_space[action_name].actuation.amount, 40)

    def test_actuate_turn_right_acts_with_amount_set_to_rotation_degrees(
        self, mock_get_agent: Mock
    ) -> None:
        mock_get_agent.return_value = self.mock_agent

        action = TurnRight(agent_id=AgentID("agent1"), rotation_degrees=39)
        action_name = self.actuator.action_name(action)
        self.action_space[action_name] = Mock()

        self.actuator.actuate_turn_right(action)

        self.mock_agent.act.assert_called_once_with(action_name)
        self.assertEqual(self.action_space[action_name].actuation.amount, 39)


if __name__ == "__main__":
    unittest.main()
