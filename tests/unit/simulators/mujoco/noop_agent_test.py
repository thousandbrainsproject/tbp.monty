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

import numpy as np
import quaternion as qt
from mujoco import mjtGeom

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.environment import SemanticID
from tbp.monty.frameworks.sensors import Resolution2D, SensorConfig, SensorID
from tbp.monty.simulators.mujoco.agents import NoopAgent
from tbp.monty.simulators.mujoco.simulator import DEFAULT_RESOLUTION, MuJoCoSimulator

VIEW_FINDER_SENSOR_ID = SensorID("view_finder")
PATCH_SENSOR_ID = SensorID("patch")
AGENT_ID = AgentID("agent_id_0")


class NoopAgentTest(unittest.TestCase):
    def test_agent_state(self) -> None:
        """Test that the agent state returns expected values."""
        # test with some non-zero values
        agent_pos = (0.0, 1.5, -1.0)
        agent_quat = (np.sin(np.pi / 4), np.cos(np.pi / 4), 0.0, 0.0)

        sim = MuJoCoSimulator(
            agents=[
                partial(
                    NoopAgent,
                    agent_id=AGENT_ID,
                    sensor_configs={
                        PATCH_SENSOR_ID: SensorConfig(resolution=DEFAULT_RESOLUTION)
                    },
                    position=agent_pos,
                    rotation=agent_quat,
                )
            ],
        )
        agent_state = sim.states[AGENT_ID]

        assert np.allclose(agent_state.position, agent_pos)
        assert np.allclose(qt.as_float_array(agent_state.rotation), agent_quat)

    def test_agent_observation_single_sensor(self) -> None:
        """Test that the agent returns expected values for a single sensor.

        We're using a test "cube" to make calculating some of the expected
        values easier.
        """
        sim = MuJoCoSimulator(
            agents=[
                partial(
                    NoopAgent,
                    agent_id=AGENT_ID,
                    sensor_configs={
                        PATCH_SENSOR_ID: SensorConfig(resolution=DEFAULT_RESOLUTION)
                    },
                )
            ]
        )
        with sim:
            sim.add_object("box", position=(0.0, 0.0, -5.0))

            obs = sim.observations[AGENT_ID]
            depth = obs[PATCH_SENSOR_ID]["depth"]
            rgba = obs[PATCH_SENSOR_ID]["rgba"]

            # We don't want to assert on the specifics of the data, since they may
            # be sensitive to rendering differences, but we want to get a rough idea
            # that we got back observational data.
            assert depth.shape == (64, 64)
            assert rgba.shape == (64, 64, 4)
            # assert that the min depth is the near surface of the cube
            assert np.allclose(depth.min(), 4.0)
            # assert that the max depth is beyond the back of the cube
            assert depth.max() >= 5.0
            # TODO: these might be too sensitive to variations, such
            #   as lighting.
            assert rgba.min() == 0
            assert rgba.max() == 255

    def test_agent_observation_semantic_default_ids(self) -> None:
        """Test that the semantic sensor default ids match expectations.

        We want the semantic sensor to behave similar to how the Habitat version does
        so we need to confirm that the values returned map correctly.
        """
        sim = MuJoCoSimulator(
            agents=[
                partial(
                    NoopAgent,
                    agent_id=AGENT_ID,
                    sensor_configs={
                        PATCH_SENSOR_ID: SensorConfig(
                            resolution=DEFAULT_RESOLUTION,
                            semantic=True,
                        )
                    },
                )
            ],
        )

        with sim:
            sim.add_object("box", position=(-2.5, 0.0, -5.0))
            sim.add_object("sphere", position=(2.5, 0.0, -5.0))
            obs = sim.observations[AGENT_ID]
            semantic = obs[PATCH_SENSOR_ID]["semantic"]
            unique_ids = set(np.unique(semantic))

            assert semantic.shape == (64, 64)
            assert unique_ids == {0, mjtGeom.mjGEOM_BOX, mjtGeom.mjGEOM_SPHERE}

    def test_agent_observation_semantic_custom_ids(self) -> None:
        """Test that the semantic sensor with custom ids match expectations.

        If we give semantic ids to `add_object` we want to make sure that the
        returned semantic sensor values use those IDs and not the defaults.
        """
        sim = MuJoCoSimulator(
            agents=[
                partial(
                    NoopAgent,
                    agent_id=AGENT_ID,
                    sensor_configs={
                        PATCH_SENSOR_ID: SensorConfig(
                            resolution=DEFAULT_RESOLUTION,
                            semantic=True,
                        )
                    },
                )
            ],
        )
        box_id = SemanticID(100)
        sphere_id = SemanticID(101)

        with sim:
            sim.add_object("box", position=(-2.5, 0.0, -5.0), semantic_id=box_id)
            sim.add_object("sphere", position=(2.5, 0.0, -5.0), semantic_id=sphere_id)
            obs = sim.observations[AGENT_ID]
            semantic = obs[PATCH_SENSOR_ID]["semantic"]
            unique_ids = set(np.unique(semantic))

            assert semantic.shape == (64, 64)
            assert unique_ids == {0, box_id, sphere_id}

    def test_agent_observation_multiple_resolutions(self) -> None:
        """Test two sensors with different resolutions.

        MuJoCo camera objects have a "resolution" attribute that doesn't do anything.
        Instead, the Renderer has to have its width and height set to match the
        desired resolution. This means we need to have multiple renderers, one per
        sensor resolution.

        This test confirms that that is working properly.
        """
        sim = MuJoCoSimulator(
            agents=[
                partial(
                    NoopAgent,
                    agent_id=AGENT_ID,
                    sensor_configs={
                        PATCH_SENSOR_ID: SensorConfig(
                            resolution=DEFAULT_RESOLUTION,
                            zoom=1.0,
                        ),
                        VIEW_FINDER_SENSOR_ID: SensorConfig(
                            resolution=Resolution2D(height=256, width=256),
                            zoom=1.0,
                        ),
                    },
                )
            ],
        )

        with sim:
            obs = sim.observations[AGENT_ID]
            patch_rgba = obs[PATCH_SENSOR_ID]["rgba"]
            view_finder_rgba = obs[VIEW_FINDER_SENSOR_ID]["rgba"]

            assert patch_rgba.shape == (
                DEFAULT_RESOLUTION.height,
                DEFAULT_RESOLUTION.width,
                4,
            )
            assert view_finder_rgba.shape == (256, 256, 4)
