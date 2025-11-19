# Copyright 2025 Thousand Brains Project
# Copyright 2021-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging

import numpy as np
import quaternion
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.actions.action_samplers import ConstantSampler
from tbp.monty.frameworks.actions.actions import (
    Action,
    LookDown,
    LookUp,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.models.motor_policies import BasePolicy
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    MotorSystemState,
    SensorState,
)
from tbp.monty.frameworks.models.states import GoalState

logger = logging.getLogger(__name__)


class LookAtPolicy(BasePolicy):
    """A policy that looks at a target.

    This class assumes a system similar to a 2-DOF gimbal in which the "outer" part
    can yaw left/right about the y-axis and the "inner" part can pitch up/down about
    the x-axis. This setup is typical of our distant agent in which the agent
    performs TurnLeft and TurnRight, while the sensor mounted to it performs
    LookDown and LookUp.

    The logic for orienting towards locations lies in `dynamic_call`. It returns a pair
    of TurnLeft/Right and LookDown/Up actions to take that must be applied in order.
    """

    def __init__(self, agent_id: str, sensor_module_id: str, **kwargs):
        """Initialize the look at policy.

        Args:
            agent_id: The agent ID
            sensor_module_id: The sensor module ID
            kwargs: Additional arguments to pass to the base policy.
        """
        # TODO: The BasePolicy should be refactored. Not all subclasses need the
        # arguments it requires. Here we just add some reasonable values.
        rng = kwargs.pop("rng", np.random.default_rng(42))
        action_sampler_class = kwargs.pop("action_sampler_class", ConstantSampler)
        action_sampler_args = kwargs.pop(
            "action_sampler_args",
            dict(
                actions=[TurnLeft, TurnRight, LookDown, LookUp],
                rotation_degrees=5.0,
            ),
        )
        self.desired_object_distance = kwargs.pop("desired_object_distance", 0.1)
        self.good_view_percentage = kwargs.pop("good_view_percentage", 0.5)
        self.use_goal_state_driven_actions = kwargs.pop(
            "use_goal_state_driven_actions", True
        )
        super().__init__(
            rng=rng,
            action_sampler_class=action_sampler_class,
            action_sampler_args=action_sampler_args,
            agent_id=agent_id,
            switch_frequency=kwargs.pop("switch_frequency", 0.0),
            **kwargs,
        )
        self.sensor_module_id = sensor_module_id
        self.driving_goal_state = None
        self.processed_observations = None  # unused -- here for compatibility

    def get_random_action(self, *args, **kwargs) -> Action:
        """Returns TurnLeft with 0 rotation degrees.

        Reimplemented due to issues with random number generation. And also, should
        all policies be expected to return random actions?
        """
        return TurnLeft(agent_id=self.agent_id, rotation_degrees=0)

    def reset(self) -> None:
        """Reset the look at policy."""
        super().reset()
        self.driving_goal_state = None
        self.processed_observations = None

    def set_driving_goal_state(self, goal_state: GoalState | None) -> None:
        self.driving_goal_state = goal_state

    def dynamic_call(self, state: MotorSystemState) -> tuple[Action, Action]:
        """Return turn left/right and look down/up actions to take.

        Computes two actions -- a yawing action and a pitching action -- that should
        orient the agent and sensor towards the driving goal state. They must be
        applied in the order in which they are returned.

        Note: the yawing actions must be performed by the agent, and the pitching
        actions must be performed by the sensor.

        Args:
            state: The motor system state.

        Returns:
            A tuple of actions, where the first action is one of TurnLeft or TurnRight,
            and the second action is one of LookDown or LookUp.

        Raises:
            RuntimeError: If no driving goal state is set.

        """
        if self.driving_goal_state is None:
            raise RuntimeError("No driving goal state set")

        # TODO: Remove this once we adhere to a standard format for motor system states.
        state = clean_habitat_motor_system_state(state)

        # Collect necessary agent and sensor pose information.
        # Subscripts: w=world, a=agent, s=sensor.
        agent_dict = state[self.agent_id]
        agent_pos_rel_world = agent_dict["position"]
        agent_rot_rel_world = as_scipy_rotation(agent_dict["rotation"])
        agent_to_world = RigidTransform.from_components(
            agent_pos_rel_world, agent_rot_rel_world
        )

        sensor_dict = agent_dict["sensors"][self.sensor_module_id]
        sensor_rot_rel_agent = as_scipy_rotation(sensor_dict["rotation"])

        # Get the target location in world and agent coordinates.
        target_rel_world = np.asarray(self.driving_goal_state.location)
        target_rel_agent = agent_to_world.inv()(target_rel_world)

        # Compute the target's azimuth, relative to the agent. This value is used to
        # compute the yaw action to be performed by the agent.
        agent_yaw = -np.arctan2(target_rel_agent[0], -target_rel_agent[2])

        # Compute the target's elevation, relative to the agent. Then subtract the
        # sensor's current pitch to get a pitch delta effective for the sensor. This
        # value is used to compute the look up/down action which must be performed
        # by the sensor mounted to the agent.
        target_pitch_rel_agent = np.arctan2(
            target_rel_agent[1], np.hypot(target_rel_agent[0], target_rel_agent[2])
        )
        sensor_pitch_rel_agent = sensor_rot_rel_agent.as_euler("xyz")[0]
        sensor_pitch = target_pitch_rel_agent - sensor_pitch_rel_agent
        # For some reason, the above is more stable and accurate than the below:
        # sensor_to_agent = RigidTransform.from_components(
        #     sensor_dict["position"], sensor_rot_rel_agent
        # )
        # t_s = sensor_to_agent.inv()(target_rel_agent)
        # _, py, sensor_pitch2 = cartesian_to_spherical(t_s)

        # Create actions to return to the the motor system.
        yaw_degrees = np.degrees(agent_yaw)
        if yaw_degrees >= 0:
            turn = TurnLeft(agent_id=self.agent_id, rotation_degrees=yaw_degrees)
        else:
            turn = TurnRight(agent_id=self.agent_id, rotation_degrees=-yaw_degrees)

        pitch_degrees = np.degrees(sensor_pitch)
        if pitch_degrees >= 0:
            look = LookUp(agent_id=self.agent_id, rotation_degrees=pitch_degrees)
        else:
            look = LookDown(agent_id=self.agent_id, rotation_degrees=-pitch_degrees)

        # For logging purposes only.
        self.driving_goal_state.info["attempted"] = True

        # Drop the reference to the goal state.
        self.driving_goal_state = None

        return [turn, look]


def as_scipy_rotation(
    obj: Rotation | quaternion.quaternion | ArrayLike,
    *,
    scalar_first: bool = True,
    axes: str = "xyz",
    degrees: bool = True,
) -> Rotation:
    """Convert a rotation description to a rotation matrix.

    Args:
        obj: The rotation to convert.
        scalar_first: Whether to use scalar-first order. Only used if a 4-element
            sequence is given.
        axes: The axes to use for euler angles. Only used if a 3-element sequence is
            given.
        degrees: Whether to use degrees. Only used if a 3-element sequence is given.

    Returns:
        A scipy.spatial.transform.Rotation instance.

    Raises:
        ValueError: If the argument is array-like but doesn't have the right shape.
    """
    if isinstance(obj, Rotation):
        return obj

    if isinstance(obj, quaternion.quaternion):
        return Rotation.from_quat([obj.x, obj.y, obj.z, obj.w])

    obj = np.asarray(obj)

    # - euler angles
    if obj.shape == (3,):
        return Rotation.from_euler(axes, obj, degrees=degrees)

    # - quaternion
    if obj.shape == (4,):
        if scalar_first:
            return Rotation.from_quat(np.roll(obj, -1))
        return Rotation.from_quat(obj)

    # - 3x3 rotation matrix
    if obj.shape == (3, 3):
        return Rotation.from_matrix(axes, obj, degrees=degrees)

    raise ValueError(f"Invalid rotation description: {obj}")

def clean_habitat_motor_system_state(raw_state: dict) -> MotorSystemState:
    """Clean up a Habitat motor system state dictionaries.

    Function that cleans up Habitat's MotorSystemState to a more usable format.
    For example, a single RGBD camera normally has separate, redundant positions
    and rotations. For example, "patch.depth" and "patch.rgba" are both present
    and contain the same rotation and position data. This function consolidates
    these into a single position and rotation for the sensor. Positions are also
    converted to the more usable numpy arrays (as opposed to magnum.Vector3 objects).

    Args:
        raw_state: The dirty habitat motor system dictionary.

    Returns:
        The cleaned motor system state.

    TODO: This is temporary. We should decide on a standard format for motor
    system states returned by simulators/environments and adhere to it.
    """
    state = MotorSystemState()
    for agent_id, raw_agent_state in raw_state.items():
        pos = raw_agent_state["position"]  # a magnum.Vector3
        rot = raw_agent_state["rotation"]  # a quaternion.quaternion
        agent_state = AgentState(
            {
                "position": np.array([pos.x, pos.y, pos.z]),
                "rotation": rot,
                "sensors": {},
            }
        )
        for sensor_key, raw_sensor_state in raw_agent_state["sensors"].items():
            sensor_id = sensor_key.split(".")[0]
            if sensor_id in agent_state["sensors"]:
                continue
            pos = raw_sensor_state["position"]
            rot = raw_sensor_state["rotation"]
            agent_state["sensors"][sensor_id] = SensorState(
                position=np.array([pos.x, pos.y, pos.z]),
                rotation=rot,
            )
        state[agent_id] = agent_state

    return state


class RigidTransform:
    """A rigid transform (rotation + translation)."""

    def __init__(
        self,
        translation: ArrayLike,
        rotation: Rotation | quaternion.quaternion | ArrayLike,
    ):
        # cached homogeneous transformation matrix
        self._matrix = None

        # basic
        self.translation = translation
        self.rotation = rotation

    @classmethod
    def from_components(
        cls,
        translation: ArrayLike,
        rotation: quaternion.quaternion | ArrayLike | Rotation,
    ) -> RigidTransform:
        """Implemented for compatibiility with future scipy release.

        Args:
            translation: The translation component.
            rotation: The rotation component.

        Returns:
            A RigidTransform instance.

        Note: this is a convenience method for creating a RigidTransform instance
        from its components. It is not necessary to use this method.
        """
        return cls(translation, rotation)

    @property
    def translation(self) -> np.ndarray:
        return self._translation.copy()

    @translation.setter
    def translation(self, translation: ArrayLike) -> None:
        vec = np.array(translation)
        if vec.shape != (3,):
            raise ValueError(f"Translation must be a 3-element array, got {vec.shape}")
        self._translation = vec
        self._matrix = None

    @property
    def rotation(self) -> Rotation:
        return self._rotation

    @rotation.setter
    def rotation(self, rotation: Rotation | quaternion.quaternion | ArrayLike) -> None:
        self._rotation = as_scipy_rotation(rotation)
        self._matrix = None

    def as_matrix(self) -> np.ndarray:
        """4x4 homogeneous transformation matrix."""
        return self._cached_matrix().copy()

    def apply(self, point: ArrayLike) -> np.ndarray:
        return self(point)

    def inv(self) -> RigidTransform:
        rotation_inv = self.rotation.inv()
        translation_inv = rotation_inv.apply(-self._translation)
        return RigidTransform(translation_inv, rotation_inv)

    def _cached_matrix(self) -> np.ndarray:
        """Returns a a homogeneous transformation matrix, possibly cached."""
        if self._matrix is None:
            matrix = np.eye(4)
            matrix[:3, :3] = self._rotation.as_matrix()
            matrix[:3, 3] = self._translation
            self._matrix = matrix
        return self._matrix

    def __call__(self, point: ArrayLike) -> np.ndarray:
        point = np.asarray(point)
        if point.ndim == 1:
            return self._rotation.apply(point) + self._translation
        else:
            matrix = self._cached_matrix()
            return (matrix[:3, :3] @ point.T).T + self._translation

    def __repr__(self) -> str:
        return (
            f"RigidTransform(translation={self._translation}, "
            f"rotation={self._rotation})"
        )
