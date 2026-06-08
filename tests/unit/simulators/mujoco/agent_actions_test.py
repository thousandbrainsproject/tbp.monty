# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from typing import Callable, ClassVar, Generator, cast
from unittest import TestCase

import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
import pytest
import quaternion as qt
from hypothesis import example, given, settings
from parameterized import parameterized_class
from typing_extensions import Concatenate, ParamSpec, Self

from tbp.monty.frameworks.actions.actions import (
    LookDown,
    LookUp,
    MoveForward,
    MoveTangentially,
    OrientHorizontal,
    OrientVertical,
    SetAgentPose,
    SetSensorRotation,
    TurnLeft,
    TurnRight,
)
from tbp.monty.geometry import Rotation
from tbp.monty.math import (
    DEFAULT_TOLERANCE,
    IDENTITY_QUATERNION,
    ZERO_VECTOR,
    QuaternionWXYZ,
    VectorXYZ,
)
from tbp.monty.simulators.mujoco import MuJoCoSimulator
from tbp.monty.simulators.mujoco.agents import Axis, DistantAgent, SurfaceAgent
from tbp.monty.simulators.mujoco.simulator import ActuateMethodMissing
from tests.unit.simulators.mujoco.noop_agent_test import (
    TEST_AGENT_ID,
    TEST_SENSOR_ID,
    default_agent_args,
)
from tests.unit.simulators.mujoco.strategies import (
    constrained_angle,
    position,
    unit_quaternion,
    x_rotation_quaterion,
)

P = ParamSpec("P")


def sim_reset(
    fn: Callable[Concatenate[ActionsTest, P], None],
) -> Callable[Concatenate[ActionsTest, P], None]:
    """A test decorator that resets the simulator on exit.

    Needed because Hypothesis runs all the test cases for a @given test in one run
    without calling TestCase.setUp and TestCase.tearDown in between.

    Returns:
        decorated actuate test method
    """

    def wrapper(self: ActionsTest, /, *args: P.args, **kwargs: P.kwargs) -> None:
        try:
            return fn(self, *args, **kwargs)
        finally:
            self.sim.reset()

    return wrapper


@contextmanager
def skip_on_missing_actuate() -> Generator[None, None, None]:
    """A test decorator that skips tests for missing actuate methods.

    Not all agents implement all actuate methods, so we skip the tests that
    cover those methods.
    """
    try:
        yield
    except ActuateMethodMissing as e:
        pytest.skip(
            reason=f"{e.agent_class.__name__} missing {e.method_name} method "
            f"under test."
        )


def agent_test_name(
    cls: type,
    num: int,  # noqa: ARG001
    params_dict: dict[str, type],
) -> str:
    """Function used by `parameterized_class` to determine the class name for the test.

    By default, the test classes that `parameterized_class` generates are just the class
    name with a number at the end. This makes it hard to know which agent class under
    test had the failure.

    Returns:
        The base test case class name with the agent class name prepended.
    """
    agent_name = params_dict["agent_class"].__name__
    return f"{agent_name}{cls.__name__}"


def orient_horizontal_position(
    left: float = 0.0, forward: float = 0.0, theta: float = 0.0
) -> npt.NDArray[np.float64]:
    """Helper to calculate position after an OrientHorizontal action.

    Starts from the origin looking down the negative-Z axis.

    Returns:
        position after the OrientHorizontal action.
    """
    # The action results in a move left, then a yaw, then a move forward
    after_left_pos = np.array([-left, 0.0, 0.0])
    theta_rot = Rotation.from_euler("y", -theta, degrees=True)
    theta_matrix = theta_rot.as_matrix()
    axis_vector = theta_matrix[:, 2] * -forward  # 2 = z-axis

    return after_left_pos + axis_vector


def orient_vertical_position(
    down: float = 0.0, forward: float = 0.0, phi: float = 0.0
) -> npt.NDArray[np.float64]:
    """Helper to calculate position after an OrientVertical action.

    Starts from the origin looking down the negative-Z axis.

    Returns:
        position after the OrientVertical action.
    """
    # The action results in a move down, then a pitch, then a move forward
    after_down_pos = np.array([0.0, -down, 0.0])
    phi_rot = Rotation.from_euler("x", phi, degrees=True)
    phi_matrix = phi_rot.as_matrix()
    axis_vector = phi_matrix[:, 2] * -forward  # 2 = z-axis

    return after_down_pos + axis_vector


@parameterized_class(
    [{"agent_class": DistantAgent}, {"agent_class": SurfaceAgent}],
    class_name_func=agent_test_name,
)
class ActionsTest(TestCase):
    """This tests the actuate methods on the agents.

    Rather than duplicate the test code for each agent, we run this suite over
    the agent classes.
    """

    sim: ClassVar[MuJoCoSimulator]
    agent_class: ClassVar[type]

    @classmethod
    def setUpClass(cls) -> None:
        # Use a single shared simulator since Hypothesis will rerun the
        # tests multiple times, and this cuts down on the runtime, but
        # leads to other complications, see `sim_reset` above.
        cls.sim = MuJoCoSimulator(
            agents=[
                partial(
                    cls.agent_class,
                    **default_agent_args(),
                )
            ]
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def tearDown(self) -> None:
        self.sim.reset()

    @given(
        final_position=position(),
        final_rotation=unit_quaternion(),
    )
    @settings(deadline=None)
    @skip_on_missing_actuate()
    @sim_reset
    def test_set_agent_pose(
        self: Self, /, final_position: VectorXYZ, final_rotation: QuaternionWXYZ
    ) -> None:
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
    @skip_on_missing_actuate()
    @sim_reset
    def test_set_sensor_rotation(
        self: Self, /, sensor_rotation: QuaternionWXYZ
    ) -> None:
        action = SetSensorRotation(
            agent_id=TEST_AGENT_ID, rotation_quat=sensor_rotation
        )
        self.sim.step([action])

        sensor_state = self.sim.states[TEST_AGENT_ID].sensors[TEST_SENSOR_ID]
        assert sensor_state.position == (0.0, 0.0, 0.0)
        # Due to the transformations that happen to sensor.rotation, some of the
        # quaternions are rotated to their negative, which represents the same rotation.
        # Changing to Rotation side-steps that issue when testing the results.
        sensor_rot = Rotation.from_quat(qt.as_float_array(sensor_state.rotation))
        expected_rot = Rotation.from_quat(sensor_rotation)
        assert expected_rot.approx_equal(sensor_rot)

    @given(distance=st.floats(min_value=0.0, max_value=10.0))
    @settings(deadline=None)
    @skip_on_missing_actuate()
    @sim_reset
    def test_move_forward(self: Self, /, distance: float) -> None:
        action = MoveForward(agent_id=TEST_AGENT_ID, distance=distance)

        self.sim.step([action])

        agent_state = self.sim.states[TEST_AGENT_ID]
        # Moving forward moves the agent in the negative Z direction, so the
        # moved distance will be the negative of the requested distance.
        assert agent_state.position == (0.0, 0.0, -distance)

    @given(delta_theta=st.floats(min_value=-180.0, max_value=180.0))
    @settings(deadline=None)
    @skip_on_missing_actuate()
    @sim_reset
    def test_turn_right(self: Self, /, delta_theta: float) -> None:
        action = TurnRight(agent_id=TEST_AGENT_ID, rotation_degrees=delta_theta)

        self.sim.step([action])

        agent_state = self.sim.states[TEST_AGENT_ID]
        agent_rot = Rotation.from_quat(qt.as_float_array(agent_state.rotation))
        expected_rot = Rotation.from_euler(
            "xyz", [0.0, -delta_theta, 0.0], degrees=True
        )

        assert agent_rot.approx_equal(expected_rot)

    @given(delta_theta=st.floats(min_value=-180.0, max_value=180.0))
    @settings(deadline=None)
    @skip_on_missing_actuate()
    @sim_reset
    def test_turn_left(self: Self, /, delta_theta: float) -> None:
        action = TurnLeft(agent_id=TEST_AGENT_ID, rotation_degrees=delta_theta)

        self.sim.step([action])

        agent_state = self.sim.states[TEST_AGENT_ID]
        agent_rot = Rotation.from_quat(qt.as_float_array(agent_state.rotation))
        expected_rot = Rotation.from_euler("xyz", [0.0, delta_theta, 0.0], degrees=True)

        assert agent_rot.approx_equal(expected_rot)

    @given(angles=constrained_angle())
    @settings(deadline=None)
    @skip_on_missing_actuate()
    @sim_reset
    def test_look_up(self: Self, /, angles: tuple[float, float]) -> None:
        """Test LookUp actions with angles less than constraint move by that angle."""
        angle, constraint = angles
        action = LookUp(
            agent_id=TEST_AGENT_ID,
            rotation_degrees=angle,
            constraint_degrees=constraint,
        )
        expected_rot = Rotation.from_euler("xyz", [angle, 0.0, 0.0], degrees=True)

        self.sim.step([action])
        sensor_state = self.sim.states[TEST_AGENT_ID].sensors[TEST_SENSOR_ID]
        sensor_rot = Rotation.from_quat(qt.as_float_array(sensor_state.rotation))

        assert sensor_rot.approx_equal(expected_rot)

    @given(
        delta_phi1=st.floats(min_value=-180.0, max_value=180.0),
        delta_phi2=st.floats(min_value=-180.0, max_value=180.0),
        phi_limit=st.floats(min_value=0.0, max_value=180.0),
    )
    @settings(deadline=None)
    @skip_on_missing_actuate()
    @sim_reset
    def test_multiple_look_up_constrained(
        self: Self, /, delta_phi1: float, delta_phi2: float, phi_limit: float
    ) -> None:
        """Test that multiple LookUp actions are properly constrained."""
        actions = [
            LookUp(
                agent_id=TEST_AGENT_ID,
                rotation_degrees=delta_phi1,
                constraint_degrees=phi_limit,
            ),
            LookUp(
                agent_id=TEST_AGENT_ID,
                rotation_degrees=delta_phi2,
                constraint_degrees=phi_limit,
            ),
        ]

        self.sim.step(actions)
        sensor_state = self.sim.states[TEST_AGENT_ID].sensors[TEST_SENSOR_ID]
        sensor_rot = Rotation.from_quat(qt.as_float_array(sensor_state.rotation))

        pitch, _, _ = sensor_rot.as_euler("xyz", degrees=True)
        assert abs(pitch) <= phi_limit + DEFAULT_TOLERANCE

    @given(angles=constrained_angle())
    @settings(deadline=None)
    @skip_on_missing_actuate()
    @sim_reset
    def test_look_down(self: Self, /, angles: tuple[float, float]) -> None:
        """Test LookDown actions with angles less than constraint move by that angle."""
        angle, constraint = angles
        action = LookDown(
            agent_id=TEST_AGENT_ID,
            rotation_degrees=angle,
            constraint_degrees=constraint,
        )
        expected_rot = Rotation.from_euler("xyz", [-angle, 0.0, 0.0], degrees=True)

        self.sim.step([action])
        sensor_state = self.sim.states[TEST_AGENT_ID].sensors[TEST_SENSOR_ID]
        sensor_rot = Rotation.from_quat(qt.as_float_array(sensor_state.rotation))

        assert sensor_rot.approx_equal(expected_rot)

    @given(
        delta_phi1=st.floats(min_value=-180.0, max_value=180.0),
        delta_phi2=st.floats(min_value=-180.0, max_value=180.0),
        phi_limit=st.floats(min_value=0.0, max_value=180.0),
    )
    @settings(deadline=None)
    @skip_on_missing_actuate()
    @sim_reset
    def test_multiple_look_down_constrained(
        self: Self, /, delta_phi1: float, delta_phi2: float, phi_limit: float
    ) -> None:
        """Test that multiple LookDown actions are properly constrained."""
        actions = [
            LookDown(
                agent_id=TEST_AGENT_ID,
                rotation_degrees=delta_phi1,
                constraint_degrees=phi_limit,
            ),
            LookDown(
                agent_id=TEST_AGENT_ID,
                rotation_degrees=delta_phi2,
                constraint_degrees=phi_limit,
            ),
        ]

        self.sim.step(actions)
        sensor_state = self.sim.states[TEST_AGENT_ID].sensors[TEST_SENSOR_ID]
        sensor_rot = Rotation.from_quat(qt.as_float_array(sensor_state.rotation))

        pitch, _, _ = sensor_rot.as_euler("xyz", degrees=True)
        assert abs(pitch) <= phi_limit + DEFAULT_TOLERANCE

    @given(
        delta_theta=st.floats(min_value=-180.0, max_value=180.0),
        # 10 meters seems like a good upper limit given the size of our
        # test objects.
        left_distance=st.floats(min_value=0.0, max_value=10.0),
        forward_distance=st.floats(min_value=0.0, max_value=10.0),
    )
    @settings(deadline=None)
    @skip_on_missing_actuate()
    @sim_reset
    def test_orient_horizontal(
        self: Self, /, delta_theta: float, left_distance: float, forward_distance: float
    ) -> None:
        """Tests the OrientHorizontal action.

        This test could be more complicated and try to test OrientHorizontal from a
        variety of starting positions and orientations, or multiple OrientHorizontal
        actions. This would make the test more complicated and the
        `orient_horizontal_position` helper would end up duplicating the code that
        the `actuate_orient_horizontal` method uses.

        We would essentially be testing the underlying mathematical properties of
        translations and rotations, which isn't really needed.

        Testing starting from the origin tests enough to prove that the action behaves
        as we expect it to without over-complicating the test.
        """
        action = OrientHorizontal(
            agent_id=TEST_AGENT_ID,
            rotation_degrees=delta_theta,
            forward_distance=forward_distance,
            left_distance=left_distance,
        )

        self.sim.step([action])
        agent_state = self.sim.states[TEST_AGENT_ID]
        agent_rot = Rotation.from_quat(qt.as_float_array(agent_state.rotation))
        agent_pos = agent_state.position

        expected_rot = Rotation.from_euler("y", -delta_theta, degrees=True)
        expected_pos = orient_horizontal_position(
            left=left_distance,
            forward=forward_distance,
            theta=delta_theta,
        )

        assert agent_rot.approx_equal(expected_rot)
        np.testing.assert_allclose(agent_pos, expected_pos)
        # Invariant: the agent shouldn't ever leave the XZ plane as a result of
        # an OrientHorizontal action.
        assert agent_pos[Axis.Y] == expected_pos[Axis.Y], (
            "OrientHorizontal action not constrained to the XZ plane."
        )
        # Property: the distance from the starting position to the final position
        # should match the Law of Cosines.
        a, b = left_distance, forward_distance
        theta = np.deg2rad(90 - delta_theta)
        expected_distance = np.sqrt(a**2 + b**2 - 2 * a * b * np.cos(theta))
        np.testing.assert_allclose(
            np.linalg.norm(agent_pos),
            expected_distance,
            err_msg="OrientHorizontal action does not obey the Law of Cosines.",
        )

    @given(
        delta_phi=st.floats(min_value=-180.0, max_value=180.0),
        down_distance=st.floats(min_value=0.0, max_value=10.0),
        forward_distance=st.floats(min_value=0.0, max_value=10.0),
    )
    @settings(deadline=None)
    @skip_on_missing_actuate()
    @sim_reset
    def test_orient_vertical(
        self: Self, /, delta_phi: float, down_distance: float, forward_distance: float
    ) -> None:
        """Tests the OrientVertical action.

        See note in `test_orient_horizontal`.
        """
        action = OrientVertical(
            agent_id=TEST_AGENT_ID,
            rotation_degrees=delta_phi,
            forward_distance=forward_distance,
            down_distance=down_distance,
        )

        self.sim.step([action])
        agent_state = self.sim.states[TEST_AGENT_ID]
        agent_rot = Rotation.from_quat(qt.as_float_array(agent_state.rotation))
        agent_pos = agent_state.position

        expected_rot = Rotation.from_euler("x", delta_phi, degrees=True)
        expected_pos = orient_vertical_position(
            down=down_distance, forward=forward_distance, phi=delta_phi
        )

        assert agent_rot.approx_equal(expected_rot)
        np.testing.assert_allclose(agent_pos, expected_pos)
        assert agent_rot.approx_equal(expected_rot)
        np.testing.assert_allclose(agent_pos, expected_pos)
        # Invariant: the agent shouldn't ever leave the YZ plane as a result of
        # an OrientHorizontal action.
        assert agent_pos[Axis.X] == expected_pos[Axis.X], (
            "OrientVertical action not constrained to the YZ plane."
        )
        # Property: the distance from the starting position to the final position
        # should match the Law of Cosines.
        a, b = down_distance, forward_distance
        phi = np.deg2rad(90 - delta_phi)
        expected_distance = np.sqrt(a**2 + b**2 - 2 * a * b * np.cos(phi))
        np.testing.assert_allclose(
            np.linalg.norm(agent_pos),
            expected_distance,
            err_msg="OrientVertical action does not obey the Law of Cosines.",
        )

    @given(
        distance=st.floats(min_value=0.0, max_value=10.0),
        theta=st.floats(min_value=-np.pi, max_value=np.pi),
    )
    @settings(deadline=None)
    @skip_on_missing_actuate()
    @sim_reset
    def test_move_tangentially(self: Self, /, distance: float, theta: float) -> None:
        # Offset used in the code that generates MoveTangentially actions to set
        # the origin angle to the positive Y axis instead of the positive X axis.
        offset = np.pi / 2
        # Since we're starting from the origin, we can directly calculate this
        # value without having to apply it to the agent's starting position.
        direction_vector = np.array(
            [np.cos(theta - offset), np.sin(theta + offset), 0.0]
        )
        action = MoveTangentially(
            agent_id=TEST_AGENT_ID,
            distance=distance,
            direction=tuple(direction_vector),
        )

        self.sim.step([action])
        agent_state = self.sim.states[TEST_AGENT_ID]
        agent_rot = Rotation.from_quat(qt.as_float_array(agent_state.rotation))

        expected_pos = direction_vector * distance

        np.testing.assert_allclose(agent_state.position, expected_pos)

        # Invariant: MoveTangentially should not rotate the agent's orientation.
        expected_rot = Rotation.from_euler("xyz", ZERO_VECTOR)
        assert agent_rot.approx_equal(expected_rot), (
            "MoveTangentially rotated the agent."
        )

    @given(
        distance=st.floats(min_value=0.0, max_value=10.0),
        theta=st.floats(min_value=-np.pi, max_value=np.pi),
        invert_vector=st.booleans(),
    )
    @settings(deadline=None)
    @skip_on_missing_actuate()
    @sim_reset
    def test_move_tangentially_obeys_inverse_property(
        self: Self, /, distance: float, theta: float, invert_vector: bool
    ) -> None:
        # Offset used in the code that generates MoveTangentially actions to set
        # the origin angle to the positive Y axis instead of the positive X axis.
        offset = np.pi / 2
        # Since we're starting from the origin, we can directly calculate this
        # value without having to apply it to the agent's starting position.
        direction_vector = np.array(
            [np.cos(theta - offset), np.sin(theta + offset), 0.0]
        )
        action = MoveTangentially(
            agent_id=TEST_AGENT_ID, distance=distance, direction=tuple(direction_vector)
        )
        # We can only invert one of the two values to get an inverse action, so
        # we test both possibilties.
        inverse_distance = distance if invert_vector else -distance
        inverse_direction = direction_vector * -1 if invert_vector else direction_vector
        inverse_action = MoveTangentially(
            agent_id=TEST_AGENT_ID,
            distance=inverse_distance,
            direction=cast("VectorXYZ", tuple(inverse_direction)),
        )

        self.sim.step([action, inverse_action])
        agent_state = self.sim.states[TEST_AGENT_ID]
        agent_rot = Rotation.from_quat(qt.as_float_array(agent_state.rotation))

        assert agent_rot.approx_equal(Rotation.from_quat(IDENTITY_QUATERNION))
        np.testing.assert_allclose(ZERO_VECTOR, agent_state.position)
