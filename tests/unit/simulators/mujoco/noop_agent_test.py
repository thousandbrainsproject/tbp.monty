# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import quaternion as qt

from tbp.monty.frameworks.actions.actions import LookUp
from tbp.monty.frameworks.agents import AgentConfig, AgentID
from tbp.monty.frameworks.sensors import SensorConfig
from tbp.monty.math import IDENTITY_QUATERNION, ZERO_VECTOR
from tbp.monty.simulators.mujoco import MuJoCoSimulator, NoopAgent

AGENT_ID = AgentID("agent_id_0")


class NoopAgentTest(unittest.TestCase):
    def test_noop_agent_state(self) -> None:
        # test with some non-zero values
        agent_pos = (0.0, 1.5, -1.0)
        sin_pi_fourths = np.sin(np.pi / 4)
        agent_quat = (sin_pi_fourths, sin_pi_fourths, 0.0, 0.0)
        agent_args = self.default_agent_args
        agent_args.update({"position": agent_pos, "rotation": agent_quat})

        sim = MuJoCoSimulator(
            agent_configs=[
                AgentConfig(
                    agent_type=NoopAgent,
                    agent_args=agent_args,
                )
            ],
            data_path=None,
        )
        agent_state = sim.states[AGENT_ID]

        assert np.allclose(agent_state.position, agent_pos)
        assert np.allclose(qt.as_float_array(agent_state.rotation), agent_quat)

    def test_noop_agent_observation(self) -> None:
        sim = MuJoCoSimulator(
            agent_configs=[
                AgentConfig(
                    agent_type=NoopAgent,
                    agent_args=self.default_agent_args,
                )
            ],
            data_path=None,
        )
        sim.add_object("box", position=(0.0, 0.0, -5.0))

        obs = sim.observations[AGENT_ID]
        depth = obs["patch"]["depth"]
        rgba = obs["patch"]["rgba"]

        # We don't want to assert on the specifics of the data, since they may
        # be sensitive to rendering differences, but we want to get a rough idea
        # that we got back observational data.
        assert depth.shape == (64, 64)
        assert rgba.shape == (64, 64, 3)
        # assert that the min depth is the near surface of the cube
        assert np.allclose(depth.min(), 4.0)
        # assert that the max depth is beyond the back of the cube
        assert depth.max() >= 5.0
        # TODO: these might be too sensitive to variations
        assert rgba.min() == 0.0
        assert rgba.max() == 127.0

    def test_agent_that_does_not_understand_an_action(self) -> None:
        """Ensure the simulator works with an agent that doesn't respond to actions."""
        agent_mock = Mock(id=AGENT_ID)
        AgentMockClass = Mock(return_value=agent_mock)  # noqa: N806
        sim = MuJoCoSimulator(
            agent_configs=[
                AgentConfig(
                    agent_type=AgentMockClass,
                    agent_args=self.default_agent_args,
                )
            ],
            data_path=None,
        )

        action = LookUp(AGENT_ID, rotation_degrees=5.0)
        sim.step([action])

    def test_agent_action_with_attribute_error(self) -> None:
        """This test ensures that the simulator doesn't swallow agent errors."""

        def actuate_look_up(*args, **kwargs):  # noqa: ARG001
            # Simulate an attribute error as from a programming mistake
            raise AssertionError("AgentMock does not have attribute 'foo'")

        agent_mock = Mock(id=AGENT_ID)
        agent_mock.actuate_look_up = Mock(side_effect=actuate_look_up)
        AgentMockClass = Mock(return_value=agent_mock)  # noqa: N806
        sim = MuJoCoSimulator(
            agent_configs=[
                AgentConfig(
                    agent_type=AgentMockClass,
                    agent_args=self.default_agent_args,
                )
            ],
            data_path=None,
        )
        action = LookUp(AGENT_ID, rotation_degrees=5.0)

        with pytest.raises(AssertionError):
            sim.step([action])

    @property
    def default_agent_args(self) -> dict[str, Any]:
        """Creates a new dictionary of default agent args.

        This way the caller is free to modify it without having to
        make a copy.
        """
        return {
            "agent_id": AGENT_ID,
            "sensor_configs": {
                "patch": SensorConfig(
                    position=ZERO_VECTOR,
                    rotation=IDENTITY_QUATERNION,
                    resolution=(64, 64),
                    zoom=1.0,
                ),
            },
            "position": ZERO_VECTOR,
            "rotation": IDENTITY_QUATERNION,
        }
