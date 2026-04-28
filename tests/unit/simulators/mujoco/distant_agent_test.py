# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from contextlib import contextmanager
from functools import partial
from typing import cast
from unittest import TestCase

import hypothesis.strategies as st
import numpy as np
import quaternion as qt
from hypothesis import example, given, settings
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.actions.actions import (
    LookDown,
    LookUp,
    MoveForward,
    SetAgentPose,
    SetSensorRotation,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.utils.transform_utils import (
    rotation_as_quat,
    rotation_from_quat,
)
from tbp.monty.math import QuaternionWXYZ, VectorXYZ
from tbp.monty.simulators.mujoco import MuJoCoSimulator
from tbp.monty.simulators.mujoco.agents import DistantAgent
from tests.unit.simulators.mujoco.noop_agent_test import (
    TEST_AGENT_ID,
    TEST_SENSOR_ID,
    default_agent_args,
)


@st.composite
def position(draw) -> VectorXYZ:
    x = draw(st.floats(min_value=-10.0, max_value=10.0))
    y = draw(st.floats(min_value=-10.0, max_value=10.0))
    z = draw(st.floats(min_value=-10.0, max_value=10.0))
    return x, y, z


@st.composite
def unit_quaternion(draw) -> QuaternionWXYZ:
    """Strategy to generate unit quaternions.

    Returns:
        A unit quaternion as a 4-tuple, scalar first
    """
    # We're generating the quaternions from Euler angles because generating the
    # coefficients directly results in zero-quaterions, which are invalid and raise
    # an error when trying to construct the Rotation.
    x_rad = draw(st.floats(min_value=0.0, max_value=np.pi * 2))
    y_rad = draw(st.floats(min_value=0.0, max_value=np.pi * 2))
    z_rad = draw(st.floats(min_value=0.0, max_value=np.pi * 2))
    rotation = Rotation.from_euler("xyz", [x_rad, y_rad, z_rad], degrees=False)
    normalized_rotation = rotation_as_quat(rotation)
    return cast("QuaternionWXYZ", tuple(normalized_rotation))


@st.composite
def x_rotation_quaterion(draw) -> QuaternionWXYZ:
    x_rad = draw(st.floats(min_value=0.0, max_value=np.pi * 2))
    rotation = Rotation.from_euler("xyz", [x_rad, 0.0, 0.0], degrees=False)
    normalized_rotation = rotation_as_quat(rotation)
    return cast("QuaternionWXYZ", tuple(normalized_rotation))


@contextmanager
def sim_resetter(sim: MuJoCoSimulator):
    """A context manager that resets the simulator on exit.

    Needed because Hypothesis runs all the test cases for a @given test in one run
    without calling TestCase.setUp and TestCase.tearDown in between.

    Yields:
        the sim object, for use in the `with` statement.
    """
    try:
        yield sim
    finally:
        sim.reset()


class DistantAgentTest(TestCase):
    @classmethod
    def setUpClass(cls):
        # Use a single shared simulator since Hypothesis will rerun the
        # tests multiple times, and this cuts down on the runtime, but
        # leads to other complications, see `sim_resetter` above.
        cls.sim = MuJoCoSimulator(
            agents=[partial(DistantAgent, **default_agent_args())]
        )

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def tearDown(self):
        self.sim.remove_all_objects()

    @given(
        final_position=position(),
        final_rotation=unit_quaternion(),
    )
    @settings(deadline=None)
    def test_set_agent_pose(self, final_position, final_rotation):
        action = SetAgentPose(
            agent_id=TEST_AGENT_ID,
            location=final_position,
            rotation_quat=final_rotation,
        )
        self.sim.step([action])

        agent_state = self.sim.states[TEST_AGENT_ID]
        assert agent_state.position == final_position
        assert agent_state.rotation == qt.quaternion(*final_rotation)

    @given(
        sensor_rotation=x_rotation_quaterion(),
    )
    # Example of a quaternion that is rotated to its negative, which is an
    # equivalent rotation.
    @example(sensor_rotation=(-0.4161468365471424, 0.9092974268256817, 0.0, 0.0))
    @settings(deadline=None)
    def test_set_sensor_rotation(self, sensor_rotation):
        action = SetSensorRotation(
            agent_id=TEST_AGENT_ID, rotation_quat=sensor_rotation
        )
        self.sim.step([action])

        sensor_state = self.sim.states[TEST_AGENT_ID].sensors[TEST_SENSOR_ID]
        assert sensor_state.position == (0.0, 0.0, 0.0)
        # Due to the transformations that happen to sensor.rotation, some of the
        # quaternions are rotated to their negative, which represents the same rotation.
        # Changing to SciPy rotation side-steps that issue when testing the results.
        sensor_rot = rotation_from_quat(qt.as_float_array(sensor_state.rotation))
        expected_rot = rotation_from_quat(sensor_rotation)
        assert expected_rot.approx_equal(sensor_rot)

    @given(distance=st.floats(min_value=0.0, max_value=10.0))
    @settings(deadline=None)
    def test_move_forward(self, distance):
        action = MoveForward(agent_id=TEST_AGENT_ID, distance=distance)
        with sim_resetter(self.sim):
            self.sim.step([action])

            agent_state = self.sim.states[TEST_AGENT_ID]
            # Moving forward moves the agent in the negative Z direction, so the
            # moved distance will be the negative of the requested distance.
            assert agent_state.position == (0.0, 0.0, -distance)

    @given(delta_theta=st.floats(min_value=-180.0, max_value=180.0))
    @settings(deadline=None)
    def test_turn_right(self, delta_theta):
        action = TurnRight(agent_id=TEST_AGENT_ID, rotation_degrees=delta_theta)
        with sim_resetter(self.sim):
            self.sim.step([action])

            agent_state = self.sim.states[TEST_AGENT_ID]
            agent_rot = rotation_from_quat(qt.as_float_array(agent_state.rotation))
            expected_rot = Rotation.from_euler(
                "xyz", [0.0, -delta_theta, 0.0], degrees=True
            )

            assert agent_rot.approx_equal(expected_rot)

    @given(delta_theta=st.floats(min_value=-180.0, max_value=180.0))
    @settings(deadline=None)
    def test_turn_left(self, delta_theta):
        action = TurnLeft(agent_id=TEST_AGENT_ID, rotation_degrees=delta_theta)
        with sim_resetter(self.sim):
            self.sim.step([action])

            agent_state = self.sim.states[TEST_AGENT_ID]
            agent_rot = rotation_from_quat(qt.as_float_array(agent_state.rotation))
            expected_rot = Rotation.from_euler(
                "xyz", [0.0, delta_theta, 0.0], degrees=True
            )

            assert agent_rot.approx_equal(expected_rot)

    @given(
        delta_phi=st.floats(min_value=0.0, max_value=180.0),
        phi_limit=st.floats(min_value=0.0, max_value=180.0),
    )
    @settings(deadline=None)
    def test_look_up(self, delta_phi, phi_limit):
        action = LookUp(
            agent_id=TEST_AGENT_ID,
            rotation_degrees=delta_phi,
            constraint_degrees=phi_limit,
        )
        with sim_resetter(self.sim):
            self.sim.step([action])

            sensor_state = self.sim.states[TEST_AGENT_ID].sensors[TEST_SENSOR_ID]
            sensor_rot = rotation_from_quat(qt.as_float_array(sensor_state.rotation))
            expected_rot = Rotation.from_euler(
                "xyz", [min(delta_phi, phi_limit), 0.0, 0.0], degrees=True
            )
            assert sensor_rot.approx_equal(expected_rot)

    @given(
        delta_phi=st.floats(min_value=0.0, max_value=180.0),
        phi_limit=st.floats(min_value=0.0, max_value=180.0),
    )
    @settings(deadline=None)
    def test_look_down(self, delta_phi, phi_limit):
        action = LookDown(
            agent_id=TEST_AGENT_ID,
            rotation_degrees=delta_phi,
            constraint_degrees=phi_limit,
        )
        with sim_resetter(self.sim):
            self.sim.step([action])

            sensor_state = self.sim.states[TEST_AGENT_ID].sensors[TEST_SENSOR_ID]
            sensor_rot = rotation_from_quat(qt.as_float_array(sensor_state.rotation))
            expected_rot = Rotation.from_euler(
                "xyz", [-min(delta_phi, phi_limit), 0.0, 0.0], degrees=True
            )
            assert sensor_rot.approx_equal(expected_rot)
