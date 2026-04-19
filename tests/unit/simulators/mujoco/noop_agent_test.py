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
from functools import partial
from typing import Any

import numpy as np
import quaternion as qt

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.sensors import SensorConfig, SensorID
from tbp.monty.math import IDENTITY_QUATERNION, ZERO_VECTOR
from tbp.monty.simulators.mujoco.agents import NoopAgent
from tbp.monty.simulators.mujoco.simulator import DEFAULT_RESOLUTION, MuJoCoSimulator

AGENT_ID = AgentID("agent_id_0")


class NoopAgentTest(unittest.TestCase):
    def test_noop_agent_state(self) -> None:
        # test with some non-zero values
        agent_pos = (0.0, 1.5, -1.0)
        agent_quat = (np.sin(np.pi / 4), np.cos(np.pi / 4), 0.0, 0.0)
        agent_args = self.default_agent_args
        agent_args.update({"position": agent_pos, "rotation": agent_quat})

        sim = MuJoCoSimulator(
            agents=[partial(NoopAgent, **agent_args)],
            data_path=None,
        )
        agent_state = sim.states[AGENT_ID]

        assert np.allclose(agent_state.position, agent_pos)
        assert np.allclose(qt.as_float_array(agent_state.rotation), agent_quat)

    def test_noop_agent_observation(self) -> None:
        sim = MuJoCoSimulator(
            agents=[partial(NoopAgent, **self.default_agent_args)],
            data_path=None,
        )
        with sim:
            sim.add_object("box", position=(0.0, 0.0, -5.0))

            obs = sim.observations[AGENT_ID]
            depth = obs[SensorID("patch")]["depth"]
            rgba = obs[SensorID("patch")]["rgba"]

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
                    resolution=DEFAULT_RESOLUTION,
                    zoom=1.0,
                ),
            },
            "position": ZERO_VECTOR,
            "rotation": IDENTITY_QUATERNION,
        }
