# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import pytest

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

import json
import os
import tempfile
import unittest
from pathlib import Path

import habitat_sim
import numpy as np
import quaternion as qt

from tbp.monty.frameworks.actions.actions import (
    LookUp,
    MoveForward,
    SetAgentPitch,
    SetAgentPose,
    SetSensorPitch,
    SetSensorPose,
    SetSensorRotation,
    SetYaw,
    TurnLeft,
)
from tbp.monty.simulators.habitat import (
    PRIMITIVE_OBJECT_TYPES,
    HabitatSim,
    SingleSensorAgent,
)


def create_agents(
    num_agents,
    resolution=(64, 64),
    semantic=False,
    action_space_type="distant_agent",
    rotation_step=10.0,
    translation_step=0.25,
) -> list[SingleSensorAgent]:
    """Create agents with RGB, Depth and optional semantic sensors.

    Args:
        num_agents: Number of agents to create
        resolution: Sensor resolution
        semantic: Whether or not to add semantic sensor
        action_space_type: Whether to use the action-space of a surface agent,
            distant agent, or an agent operating with absolute, world-coordinate actions
            only
        rotation_step: Default action rotation step in degrees
        translation_step: Default action translation step in meters

    Returns:
        The created agents.
    """
    agents = []
    for i in range(num_agents):
        cam = SingleSensorAgent(
            agent_id=f"{i}",
            sensor_id="0",
            resolution=resolution,
            semantic=semantic,
            translation_step=translation_step,
            rotation_step=rotation_step,
            action_space_type=action_space_type,
        )
        agents.append(cam)
    return agents


class HabitatSimTest(unittest.TestCase):
    def test_create_environment(self):
        # Single agent with two 5x5 cameras
        agents = create_agents(num_agents=1, resolution=(5, 5))
        # Get first agent id
        agent_id = agents[0].agent_id
        sensor_id = agents[0].sensor_id
        agent_config = agents[0].get_spec()
        with HabitatSim(agents=agents) as sim:
            # Check agent configuration
            actual_agent_config = sim.get_agent(agent_id).agent_config
            self.assertEqual(actual_agent_config, agent_config)

            # Check if 2 sensors were created for the agent (RGB and Depth)
            obs = sim.observations
            agent_obs = obs[agent_id]
            sensor_obs = agent_obs[sensor_id]
            self.assertEqual(2, len(sensor_obs))

            # Check sensor resolution
            shape = sensor_obs["depth"].shape
            self.assertSequenceEqual((5, 5), shape[:2])

            # Check default action space
            action_space = sim.action_space
            expected_actions = set(agent_config.action_space.keys())
            self.assertSetEqual(expected_actions, action_space)

            # Make sure there are no objects
            num_objs = sim.num_objects
            self.assertEqual(num_objs, 0)

    def test_multiple_agents(self):
        # 2 agents with two 5x5 cameras each
        agents = create_agents(num_agents=2, resolution=(5, 5))

        with HabitatSim(agents=agents) as sim:
            # Check agent configuration
            for agent in agents:
                actual_agent_config = sim.get_agent(agent.agent_id).agent_config
                agent_config = agent.get_spec()
                self.assertEqual(actual_agent_config, agent_config)

            # Check if 2 sensors were created for each agent
            obs = sim.observations
            for agent in agents:
                agent_obs = obs[agent.agent_id]
                sensor_obs = agent_obs[agent.sensor_id]
                self.assertEqual(2, len(sensor_obs))

    def test_add_object(self):
        agents = create_agents(num_agents=1)
        with HabitatSim(agents=agents) as sim:
            # Starts empty
            num_objs = sim.num_objects
            self.assertEqual(num_objs, 0)

            # Add single object
            sim.add_object(name="cylinder", position=(0.0, 1.0, 0.2))
            num_objs = sim.num_objects
            self.assertEqual(num_objs, 1)

            # Add another object
            sim.add_object(name="cubeSolid", position=(0.5, 1.0, 0.2))
            num_objs = sim.num_objects
            self.assertEqual(num_objs, 2)

            # Test remove objects
            sim.remove_all_objects()
            num_objs = sim.num_objects
            self.assertEqual(num_objs, 0)

    def test_primitive_objects(self):
        # Create camera agent returning semantic id
        agents = create_agents(num_agents=1, semantic=True)
        agent_id = agents[0].agent_id
        sensor_id = agents[0].sensor_id
        with HabitatSim(agents=agents) as sim:
            for obj_name, expected_obj_id in PRIMITIVE_OBJECT_TYPES.items():
                sim.remove_all_objects()
                sim.add_object(obj_name, position=(0.0, 1.5, -0.2))
                obs = sim.observations
                agent_obs = obs[agent_id]
                sensor_obs = agent_obs[sensor_id]
                semantic = sensor_obs["semantic"]
                actual = np.unique(semantic[semantic.nonzero()])
                self.assertEqual(actual, expected_obj_id)

    def test_move_and_get_agent_state(self):
        """Move agent and return agent state and sensor observations."""
        # Create camera agent returning semantic id
        rotation_degrees = 10.0
        agents = create_agents(
            num_agents=1, semantic=True, rotation_step=rotation_degrees
        )
        agent_id = agents[0].agent_id
        sensor_id = agents[0].sensor_id
        with HabitatSim(agents=agents) as sim:
            # Add a couple of objects
            cylinder = sim.add_object(name="cylinderSolid", position=(-0.2, 1.5, -0.2))
            cube = sim.add_object(name="cubeSolid", position=(0.6, 1.5, -0.6))

            # Check if observations include both objects
            expected = {cylinder.semantic_id, cube.semantic_id}
            obs = sim.observations
            agent_obs = obs[agent_id]
            sensor_obs = agent_obs[sensor_id]
            semantic = sensor_obs["semantic"]
            actual = set(semantic[semantic.nonzero()])
            self.assertSetEqual(expected, actual)

            # Turn the camera 10 degrees to the left.
            # The cube should be out of view
            turn_left = TurnLeft(agent_id=agent_id, rotation_degrees=rotation_degrees)
            obs = sim.apply_actions([turn_left])
            obs = obs[agent_id]
            expected = {cylinder.semantic_id}
            semantic = np.unique(obs[sensor_id]["semantic"])
            actual = set(semantic[semantic.nonzero()])
            self.assertSetEqual(expected, actual)

            # Reset simulator and now the cylinder and cube should be back into view
            initial_obs = sim.reset()
            obs = initial_obs[agent_id]
            expected = {cylinder.semantic_id, cube.semantic_id}
            semantic = np.unique(obs[sensor_id]["semantic"])

            actual = set(semantic[semantic.nonzero()])
            self.assertSetEqual(expected, actual)

    def test_zoom(self):
        expected_1x_zoom = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]

        expected_2x_zoom = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]

        agents = create_agents(num_agents=1, resolution=(16, 16), semantic=True)
        agent_id = agents[0].agent_id
        sensor_id = agents[0].sensor_id
        with HabitatSim(agents=agents) as sim:
            agent = sim.get_agent(agent_id)
            camera = agent._sensors[f"{sensor_id}.semantic"]

            # Place cube 0.5 meters away from camera
            sim.add_object(name="cube", position=(0.0, 1.5, -0.5), semantic_id=1)

            # Check initial cube observations before zoom
            obs = sim.observations
            camera_obs = obs[agent_id][sensor_id]["semantic"].tolist()
            self.assertListEqual(expected_1x_zoom, camera_obs)

            # Apply 2X zoom to the camera
            camera.zoom(2.0)
            obs = sim.observations
            camera_obs = obs[agent_id][sensor_id]["semantic"].tolist()
            self.assertListEqual(expected_2x_zoom, camera_obs)

            # Zoom out 0.5 restoring original zoom factor (1X)
            camera.zoom(0.5)
            obs = sim.observations
            camera_obs = obs[agent_id][sensor_id]["semantic"].tolist()
            self.assertListEqual(expected_1x_zoom, camera_obs)

    def test_states(self):
        agent_pos = np.array([2.125, 1.5, -5.278])
        agent_rot = qt.from_rotation_vector([np.pi / 2, 0.0, 0.0])
        rotation_degrees = 10.0
        translation_distance = 0.25
        agents = create_agents(
            num_agents=1,
            resolution=(16, 16),
            semantic=True,
            rotation_step=rotation_degrees,
            translation_step=translation_distance,
        )
        agent_id = agents[0].agent_id
        sensor_id = agents[0].sensor_id
        agent_config = agents[0].get_spec()
        sensor_spec = agent_config.sensor_specifications[0]
        sensor_pos = sensor_spec.position
        sensor_rot = qt.from_rotation_vector(sensor_spec.orientation)

        # Compute rotation quartenions
        turn_left_spec = agent_config.action_space[f"{agent_id}.turn_left"]
        amount = turn_left_spec.actuation.amount
        turn_left_quat = qt.from_rotation_vector([0.0, np.deg2rad(amount), 0.0])
        look_up_spec = agent_config.action_space[f"{agent_id}.look_up"]
        amount = look_up_spec.actuation.amount
        look_up_quat = qt.from_rotation_vector([np.deg2rad(amount), 0.0, 0.0])
        move_forward_spec = agent_config.action_space[f"{agent_id}.move_forward"]
        amount = move_forward_spec.actuation.amount
        move_forward_offset = [0.0, amount, 0.0]
        with HabitatSim(agents=agents) as sim:
            # Place agent at initial position and rotation
            agent_state = habitat_sim.AgentState(position=agent_pos, rotation=agent_rot)
            sim.initialize_agent(agent_id, agent_state)

            # Check initial state
            states = sim.states
            agent_state = states[agent_id]
            sensor_state = agent_state["sensors"][f"{sensor_id}.rgba"]
            self.assertEqual(agent_state["position"], agent_pos)
            self.assertTrue(qt.isclose(agent_state["rotation"], agent_rot, rtol=1e-4))
            self.assertEqual(sensor_state["position"], sensor_pos)
            self.assertTrue(qt.isclose(sensor_state["rotation"], sensor_rot, rtol=1e-4))

            # turn agent body left
            turn_left = TurnLeft(agent_id=agent_id, rotation_degrees=rotation_degrees)
            sim.apply_actions([turn_left])
            states = sim.states
            agent_state = states[agent_id]
            sensor_state = agent_state["sensors"][f"{sensor_id}.rgba"]
            expected_rot = agent_rot * turn_left_quat

            # Agent body position should stay unchanged
            # Agent body rotation should be offset by turn_left_quat
            # Sensor should stay unchanged
            self.assertTrue(
                qt.isclose(agent_state["rotation"], expected_rot, rtol=1e-4)
            )
            self.assertEqual(agent_state["position"], agent_pos)
            self.assertEqual(sensor_state["position"], sensor_pos)
            self.assertTrue(np.isclose(sensor_state["rotation"], sensor_rot, rtol=1e-4))

            # Move sensor left
            sim.reset()
            look_up = LookUp(agent_id=agent_id, rotation_degrees=rotation_degrees)
            sim.apply_actions([look_up])
            states = sim.states
            agent_state = states[agent_id]
            sensor_state = agent_state["sensors"][f"{sensor_id}.rgba"]
            expected_rot = agent_rot * turn_left_quat
            expected_rot = sensor_rot * look_up_quat

            # Agent body position should stay unchanged
            # Agent body rotation should stay unchanged
            # Sensor location should stay unchanged
            # Sensor rotation should be offset by look_up_quat
            self.assertEqual(agent_state["position"], agent_pos)
            self.assertTrue(qt.isclose(agent_state["rotation"], agent_rot, rtol=1e-4))
            self.assertEqual(sensor_state["position"], sensor_pos)
            self.assertTrue(
                qt.isclose(sensor_state["rotation"], expected_rot, rtol=1e-4)
            )

            # Move agent forward
            sim.reset()
            move_forward = MoveForward(agent_id=agent_id, distance=translation_distance)
            sim.apply_actions([move_forward])
            states = sim.states
            agent_state = states[agent_id]
            sensor_state = agent_state["sensors"][f"{sensor_id}.rgba"]

            # Agent body position should be offset by move_forward_offset
            # Agent body rotation should stay unchanged
            # Sensor location should stay unchanged
            # Sensor rotation should stay unchanged
            self.assertEqual(agent_state["position"], agent_pos + move_forward_offset)
            self.assertTrue(qt.isclose(agent_state["rotation"], agent_rot, rtol=1e-4))
            self.assertEqual(sensor_state["position"], sensor_pos)
            self.assertTrue(qt.isclose(sensor_state["rotation"], sensor_rot, rtol=1e-4))

    def test_data_path(self):
        agents = create_agents(num_agents=1)
        # Check valid data path
        with tempfile.TemporaryDirectory() as data_path:
            # Create valid habitat object
            with open(
                os.path.join(data_path, "test_obj.object_config.json"), "w"
            ) as json_file:
                json.dump(
                    {"render_asset": "icosphereSolid_subdivs_1", "mass": 1}, json_file
                )
            with HabitatSim(agents=agents, data_path=data_path) as sim:
                obj_id = sim.add_object("test_obj")
                self.assertTrue(obj_id)

        # Check valid dataset path
        with tempfile.TemporaryDirectory() as data_path:
            # Emulate YCB dataset
            dataset_path = Path(data_path) / "objects" / "ycb"
            dataset_path.mkdir(parents=True)
            # Create valid habitat object inside the dataset
            with (dataset_path / "test_obj.object_config.json").open("w") as json_file:
                json.dump(
                    {"render_asset": "icosphereSolid_subdivs_1", "mass": 1}, json_file
                )
            with HabitatSim(agents=agents, data_path=data_path) as sim:
                obj_id = sim.add_object("test_obj")
                self.assertTrue(obj_id)
        # Check invalid data path (i.e. without any valid habitat json files)
        with tempfile.TemporaryDirectory() as data_path:
            with self.assertRaises(ValueError):
                with HabitatSim(agents=agents, data_path=data_path) as sim:
                    pass

    def test_set_yaw(self):
        agent_pos = np.zeros(3)
        agent_rot = qt.one
        agents = create_agents(num_agents=1, action_space_type="absolute_only")
        agent_id = agents[0].agent_id
        sensor_id = agents[0].sensor_id
        agent_config = agents[0].get_spec()
        sensor_spec = agent_config.sensor_specifications[0]
        sensor_pos = sensor_spec.position
        sensor_rot = qt.from_rotation_vector(sensor_spec.orientation)
        with HabitatSim(agents=agents) as sim:
            # Place agent at initial position and rotation
            agent_state = habitat_sim.AgentState(position=agent_pos, rotation=agent_rot)
            sim.initialize_agent(agent_id, agent_state)

            # Make sure absolute yaw does not change over multiple calls
            for _ in range(5):
                # Set absolute yaw
                set_yaw = SetYaw(agent_id=agent_id, rotation_degrees=45.0)
                sim.apply_actions([set_yaw])

                # Agent position should stay the same
                # Sensor position and rotation should stay the same
                # Agent Z rotation should be 45 deg
                expected = qt.from_rotation_vector([0.0, 0.0, np.deg2rad(45)])
                states = sim.states
                agent_state = states[agent_id]
                sensor_state = agent_state["sensors"][f"{sensor_id}.rgba"]
                self.assertEqual(agent_state["position"], agent_pos)
                self.assertEqual(sensor_state["position"], sensor_pos)
                self.assertTrue(
                    qt.isclose(sensor_state["rotation"], sensor_rot, rtol=1e-4)
                )
                self.assertTrue(
                    qt.isclose(agent_state["rotation"], expected, rtol=1e-4)
                )

    def test_set_sensor_pitch(self):
        agent_pos = np.zeros(3)
        agent_rot = qt.one
        agents = create_agents(num_agents=1, action_space_type="absolute_only")
        agent_id = agents[0].agent_id
        sensor_id = agents[0].sensor_id
        agent_config = agents[0].get_spec()
        sensor_spec = agent_config.sensor_specifications[0]
        sensor_pos = sensor_spec.position
        with HabitatSim(agents=agents) as sim:
            # Place agent at initial position and rotation
            agent_state = habitat_sim.AgentState(position=agent_pos, rotation=agent_rot)
            sim.initialize_agent(agent_id, agent_state)

            # Make sure absolute pitch does not change over multiple calls
            for _ in range(5):
                # Set absolute pitch
                set_sensor_pitch = SetSensorPitch(agent_id=agent_id, pitch_degrees=45.0)
                sim.apply_actions([set_sensor_pitch])

                # Agent position and rotation should stay the same
                # Sensor position should stay the same
                # Sensot Y rotation should be 45 deg
                expected = qt.from_rotation_vector([0.0, np.deg2rad(45), 0.0])
                states = sim.states
                agent_state = states[agent_id]
                sensor_state = agent_state["sensors"][f"{sensor_id}.rgba"]
                self.assertEqual(agent_state["position"], agent_pos)
                self.assertEqual(sensor_state["position"], sensor_pos)
                self.assertTrue(
                    qt.isclose(agent_state["rotation"], agent_rot, rtol=1e-4)
                )
                self.assertTrue(
                    qt.isclose(sensor_state["rotation"], expected, rtol=1e-4)
                )

    def test_set_agent_pitch(self):
        agent_pos = np.zeros(3)
        agent_rot = qt.one
        agents = create_agents(num_agents=1, action_space_type="absolute_only")
        sensor_rot_initial = agent_rot  # create_agents will initialize the sensor
        # rotations to the agent_rot that is passed in
        agent_id = agents[0].agent_id
        sensor_id = agents[0].sensor_id
        agent_config = agents[0].get_spec()
        sensor_spec = agent_config.sensor_specifications[0]
        sensor_pos = sensor_spec.position
        with HabitatSim(agents=agents) as sim:
            # Place agent at initial position and rotation
            agent_state = habitat_sim.AgentState(position=agent_pos, rotation=agent_rot)
            sim.initialize_agent(agent_id, agent_state)

            # Make sure absolute pitch does not change over multiple calls
            for _ in range(5):
                # Set absolute pitch
                set_agent_pitch = SetAgentPitch(agent_id=agent_id, pitch_degrees=45.0)
                sim.apply_actions([set_agent_pitch])

                # Sensor position and rotation should stay the same
                # Agent position should stay the same
                # Agent Y rotation should be 45 deg
                expected = qt.from_rotation_vector([0.0, np.deg2rad(45), 0.0])
                states = sim.states
                agent_state = states[agent_id]
                sensor_state = agent_state["sensors"][f"{sensor_id}.rgba"]
                self.assertEqual(agent_state["position"], agent_pos)
                self.assertEqual(sensor_state["position"], sensor_pos)
                self.assertTrue(
                    qt.isclose(agent_state["rotation"], expected, rtol=1e-4)
                )
                self.assertTrue(
                    qt.isclose(sensor_state["rotation"], sensor_rot_initial, rtol=1e-4)
                )

    def test_set_sensor_rotation(self):
        agent_pos = np.zeros(3)
        agent_rot = qt.one
        agents = create_agents(num_agents=1, action_space_type="absolute_only")
        agent_id = agents[0].agent_id
        sensor_id = agents[0].sensor_id
        agent_config = agents[0].get_spec()
        sensor_spec = agent_config.sensor_specifications[0]
        sensor_pos = sensor_spec.position
        with HabitatSim(agents=agents) as sim:
            # Place agent at initial position and rotation
            agent_state = habitat_sim.AgentState(position=agent_pos, rotation=agent_rot)
            sim.initialize_agent(agent_id, agent_state)

            # Make sure absolute rotation does not change over multiple calls; position
            # should remain unchanged
            for _ in range(5):
                expected_rot = qt.from_rotation_vector(
                    [np.deg2rad(-45), np.deg2rad(45), np.deg2rad(355)]
                )

                set_sensor_rotation = SetSensorRotation(
                    agent_id=agent_id, rotation_quat=expected_rot
                )
                sim.apply_actions([set_sensor_rotation])
                states = sim.states
                agent_state = states[agent_id]
                sensor_state = agent_state["sensors"][f"{sensor_id}.rgba"]
                self.assertEqual(agent_state["position"], agent_pos)
                self.assertEqual(sensor_state["position"], sensor_pos)
                self.assertTrue(
                    qt.isclose(agent_state["rotation"], agent_rot, rtol=1e-4)
                )
                self.assertTrue(
                    qt.isclose(sensor_state["rotation"], expected_rot, rtol=1e-4)
                )

    def test_set_sensor_pose(self):
        agent_pos = np.zeros(3)
        agent_rot = qt.one
        agents = create_agents(num_agents=1, action_space_type="absolute_only")
        sensor_rot_initial = agent_rot  # create_agents will initialize the sensor
        # rotations to the agent_rot that is passed in
        agent_id = agents[0].agent_id
        sensor_id = agents[0].sensor_id
        agent_config = agents[0].get_spec()
        sensor_spec = agent_config.sensor_specifications[0]
        sensor_pos = sensor_spec.position
        with HabitatSim(agents=agents) as sim:
            # Place agent at initial position and rotation
            agent_state = habitat_sim.AgentState(position=agent_pos, rotation=agent_rot)
            sim.initialize_agent(agent_id, agent_state)

            # Make sure absolute rotation does not change over multiple calls; position
            # should remain unchanged
            for _ in range(5):
                expected_rot = qt.from_rotation_vector(
                    [np.deg2rad(-45), np.deg2rad(45), np.deg2rad(355)]
                )

                set_sensor_pose = SetSensorPose(
                    agent_id=agent_id, location=np.zeros(3), rotation_quat=expected_rot
                )
                sim.apply_actions([set_sensor_pose])
                states = sim.states
                agent_state = states[agent_id]
                sensor_state = agent_state["sensors"][f"{sensor_id}.rgba"]
                self.assertEqual(agent_state["position"], agent_pos)
                self.assertEqual(sensor_state["position"], sensor_pos)
                self.assertTrue(
                    qt.isclose(agent_state["rotation"], agent_rot, rtol=1e-4)
                )
                self.assertTrue(
                    qt.isclose(sensor_state["rotation"], expected_rot, rtol=1e-4)
                )

            # Make sure absolute position does not change over multiple calls; rotation
            # should be reset to initial value
            for _ in range(5):
                expected_pos = np.array([0.75, 1.25, -0.75])

                set_sensor_pose = SetSensorPose(
                    agent_id=agent_id,
                    location=expected_pos,
                    rotation_quat=sensor_rot_initial,
                )
                sim.apply_actions([set_sensor_pose])
                states = sim.states
                agent_state = states[agent_id]
                sensor_state = agent_state["sensors"][f"{sensor_id}.rgba"]
                self.assertEqual(agent_state["position"], agent_pos)
                self.assertEqual(sensor_state["position"], expected_pos)
                self.assertTrue(
                    qt.isclose(agent_state["rotation"], agent_rot, rtol=1e-4)
                )
                self.assertTrue(
                    qt.isclose(sensor_state["rotation"], sensor_rot_initial, rtol=1e-4)
                )

            # Make sure absolute position and rotation do not change over multiple calls
            for _ in range(5):
                expected_pos = np.array([2.5, -2.5, 2.5])
                expected_rot = qt.from_rotation_vector(
                    [np.deg2rad(180), np.deg2rad(-180), np.deg2rad(360)]
                )

                set_sensor_pose = SetSensorPose(
                    agent_id=agent_id,
                    location=expected_pos,
                    rotation_quat=expected_rot,
                )
                sim.apply_actions([set_sensor_pose])
                states = sim.states
                agent_state = states[agent_id]
                sensor_state = agent_state["sensors"][f"{sensor_id}.rgba"]
                self.assertEqual(agent_state["position"], agent_pos)
                self.assertEqual(sensor_state["position"], expected_pos)
                self.assertTrue(
                    qt.isclose(agent_state["rotation"], agent_rot, rtol=1e-4)
                )
                self.assertTrue(
                    qt.isclose(sensor_state["rotation"], expected_rot, rtol=1e-4)
                )

    def test_set_agent_pose(self):
        agent_pos = np.zeros(3)
        agent_rot = qt.one
        agents = create_agents(num_agents=1, action_space_type="absolute_only")
        sensor_rot_initial = agent_rot
        agent_id = agents[0].agent_id
        sensor_id = agents[0].sensor_id
        agent_config = agents[0].get_spec()
        sensor_spec = agent_config.sensor_specifications[0]
        sensor_pos = sensor_spec.position
        with HabitatSim(agents=agents) as sim:
            # Place agent at initial position and rotation
            agent_state = habitat_sim.AgentState(position=agent_pos, rotation=agent_rot)
            sim.initialize_agent(agent_id, agent_state)

            # Make sure absolute rotation does not change over multiple calls; position
            # should remain unchanged
            for _ in range(5):
                expected_rot = qt.from_rotation_vector(
                    [np.deg2rad(-45), np.deg2rad(45), np.deg2rad(355)]
                )

                set_agent_pose = SetAgentPose(
                    agent_id=agent_id, location=np.zeros(3), rotation_quat=expected_rot
                )
                sim.apply_actions([set_agent_pose])
                states = sim.states
                agent_state = states[agent_id]
                sensor_state = agent_state["sensors"][f"{sensor_id}.rgba"]
                self.assertEqual(agent_state["position"], agent_pos)
                self.assertEqual(sensor_state["position"], sensor_pos)
                self.assertTrue(
                    qt.isclose(agent_state["rotation"], expected_rot, rtol=1e-4)
                )
                self.assertTrue(
                    qt.isclose(sensor_state["rotation"], sensor_rot_initial, rtol=1e-4)
                )

            # Make sure absolute position does not change over multiple calls; rotation
            # should be reset to initial value
            for _ in range(5):
                expected_pos = np.array([0.75, 1.25, -0.75])

                set_agent_pose = SetAgentPose(
                    agent_id=agent_id,
                    location=expected_pos,
                    rotation_quat=sensor_rot_initial,
                )
                sim.apply_actions([set_agent_pose])
                states = sim.states
                agent_state = states[agent_id]
                sensor_state = agent_state["sensors"][f"{sensor_id}.rgba"]
                self.assertEqual(agent_state["position"], expected_pos)
                self.assertEqual(sensor_state["position"], sensor_pos)
                self.assertTrue(
                    qt.isclose(agent_state["rotation"], agent_rot, rtol=1e-4)
                )
                self.assertTrue(
                    qt.isclose(sensor_state["rotation"], sensor_rot_initial, rtol=1e-4)
                )

            # Make sure absolute position and rotation do not change over multiple calls
            for _ in range(5):
                expected_pos = np.array([2.5, -2.5, 2.5])
                expected_rot = qt.from_rotation_vector(
                    [np.deg2rad(180), np.deg2rad(-180), np.deg2rad(360)]
                )

                set_agent_pose = SetAgentPose(
                    agent_id=agent_id,
                    location=expected_pos,
                    rotation_quat=expected_rot,
                )
                sim.apply_actions([set_agent_pose])
                states = sim.states
                agent_state = states[agent_id]
                sensor_state = agent_state["sensors"][f"{sensor_id}.rgba"]
                self.assertEqual(agent_state["position"], expected_pos)
                self.assertEqual(sensor_state["position"], sensor_pos)
                self.assertTrue(
                    qt.isclose(agent_state["rotation"], expected_rot, rtol=1e-4)
                )
                self.assertTrue(
                    qt.isclose(sensor_state["rotation"], sensor_rot_initial, rtol=1e-4)
                )

    def test_agent_height(self):
        agent = SingleSensorAgent(
            agent_id="camera", sensor_id="0", agent_position=[0.0, 0.0, 0.0], height=0.0
        )
        with HabitatSim(agents=[agent]) as sim:
            states = sim.states
            agent_state = states[agent.agent_id]
            actual_height = agent_state["position"][1]
            self.assertEqual(actual_height, 0.0)

    def test_object_scale(self):
        expected_1x_zoom = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]

        expected_2x_zoom = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        agents = create_agents(num_agents=1, resolution=(16, 16), semantic=True)
        agent_id = agents[0].agent_id
        sensor_id = agents[0].sensor_id
        with HabitatSim(agents=agents) as sim:
            # Place cube 0.5 meters away from camera
            sim.add_object(name="cube", position=(0.0, 1.5, -0.5), semantic_id=1)

            # Check original cube observations without scale
            obs = sim.observations
            camera_obs = obs[agent_id][sensor_id]["semantic"].tolist()
            self.assertListEqual(expected_1x_zoom, camera_obs)

            # Apply 2X scale
            sim.remove_all_objects()

            # On the first time, the scaled object is added to habitat
            sim.add_object(
                name="cube",
                position=(0.0, 1.5, -0.5),
                scale=(2.0, 2.0, 2.0),
                semantic_id=1,
            )
            obs = sim.observations
            camera_obs = obs[agent_id][sensor_id]["semantic"].tolist()
            self.assertListEqual(expected_2x_zoom, camera_obs)

            # On the second time, the old object is accessed
            sim.remove_all_objects()
            sim.add_object(
                name="cube",
                position=(0.0, 1.5, -0.5),
                scale=(2.0, 2.0, 2.0),
                semantic_id=1,
            )
            obs = sim.observations
            camera_obs = obs[agent_id][sensor_id]["semantic"].tolist()
            self.assertListEqual(expected_2x_zoom, camera_obs)


if __name__ == "__main__":
    unittest.main()
