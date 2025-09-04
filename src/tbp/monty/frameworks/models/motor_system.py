# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import quaternion
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as SciPyRotation

from tbp.monty.frameworks.actions.action_samplers import (
    ActionSampler,
    ConstantSampler,
)
from tbp.monty.frameworks.actions.actions import Action, LookDown, TurnLeft
from tbp.monty.frameworks.config_utils.policy_setup_utils import (
    generate_action_list,
)
from tbp.monty.frameworks.models.motor_policies import BasePolicy, MotorPolicy
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState
from tbp.monty.frameworks.models.states import GoalState, State


@dataclass
class MotorSystemTelemetry:
    state: MotorSystemState
    driving_goal_state: GoalState | None
    experiment_mode: Literal["train", "eval"] | None
    processed_observations: State | None
    action: Action | None


class MotorSystem:
    """The basic motor system implementation."""

    def __init__(
        self,
        policy: MotorPolicy,
        state: MotorSystemState | None = None,
        save_telemetry: bool = False,
    ) -> None:
        """Initialize the motor system with a motor policy.

        Args:
            policy: The default motor policy to use.
            state: The initial state of the motor system.
                Defaults to None.
            save_telemetry: Whether to save telemetry.
                Defaults to False.
        """
        self._default_policy = self._policy = policy
        self._look_at_policy = LookAtPolicy(rng=self._default_policy.rng)
        self.save_telemetry = save_telemetry
        self.reset(state)

    @property
    def agent_id(self) -> str:
        """Returns the agent ID of the motor system.

        NOTE: this assumes one agent is associated with the motor system.
        When we move to a motor system composed of many motor modules, agent IDs
        will likely be associated with the latter.
        """
        return self._policy.agent_id

    @property
    def last_action(self) -> Action | None:
        """Returns the last action taken by the motor system."""
        return self._last_action

    @property
    def policy(self) -> MotorPolicy:
        """Returns the motor policy."""
        return self._policy

    @property
    def state(self) -> MotorSystemState:
        """Returns the state of the motor system."""
        return self._state

    @state.setter
    def state(self, state: MotorSystemState | None) -> None:
        """Sets the state of the motor system."""
        self._state = state

    @property
    def telemetry(self) -> list[MotorSystemTelemetry]:
        """Returns the telemetry of the motor system."""
        return self._telemetry

    def driving_goal_state(self) -> GoalState | None:
        """Returns the driving goal state."""
        return self._driving_goal_state

    def set_driving_goal_state(self, goal_state: GoalState | None) -> None:
        """Sets the driving goal state.

        Args:
            goal_state: The goal state to drive the motor system.
        """
        self._driving_goal_state = goal_state

    def experiment_mode(self) -> Literal["train", "eval"] | None:
        """Returns the experiment mode."""
        return self._experiment_mode

    def set_experiment_mode(self, mode: Literal["train", "eval"] | None) -> None:
        """Sets the experiment mode."""
        self._experiment_mode = mode

    def processed_observations(self) -> State | None:
        """Returns the processed observations."""
        return self._processed_observations

    def set_processed_observations(self, processed_observations: State | None) -> None:
        """Sets the processed observations."""
        self._processed_observations = processed_observations
        self._policy.processed_observations = processed_observations

    def reset(self, state: MotorSystemState | None = None) -> None:
        """Reset the motor system."""
        self._policy = self._default_policy
        self._state = state
        self._driving_goal_state = None
        self._experiment_mode = None
        self._processed_observations = None
        self._last_action = None
        self._telemetry = []
        self._n_steps = 0

    def pre_episode(self) -> None:
        """Pre episode hook."""
        self.reset()
        self._policy.pre_episode()

    def post_episode(self) -> None:
        """Post episode hook."""
        self._policy.post_episode()

    def step(self) -> None:
        """Select a policy, etc.

        This must be called before `__call__()` is used.

        The important thing here is to determine whether the driving goal state
        should be attempted with the data loader's execute_jump_attempt() method.
        If so, we need to set self._policy to the appropriate `InformedPolicy` object
        and then set its `driving_goal_state` attribute. This is what the data loader
        will look for when deciding to use `execute_jump_attempt()`.

        If we don't want to attempt the driving goal state with a jump, we need to
        set self._policy to some other policy but maybe set an attribute like
        `driving_goal_state` but with a different name (or else not have that policy
        inherit from `InformedPolicy`).

        If there is no driving goal state, pick some other policy.
        """
        self._policy = self._select_policy()
        self._policy.set_experiment_mode(self._experiment_mode)
        if hasattr(self._policy, "set_driving_goal_state"):
            self._policy.set_driving_goal_state(self._driving_goal_state)
        self._policy.processed_observations = self._processed_observations

    def _select_policy(self) -> MotorPolicy:
        """Selects a policy for the motor system.

        Returns:
            The policy to use.
        """
        if self._driving_goal_state:
            if self._driving_goal_state.sender_id == "view_finder":
                return self._look_at_policy

        return self._default_policy

    def _post_call(self, action: Action) -> None:
        """Post call hook."""
        if self.save_telemetry:
            self._telemetry.append(
                MotorSystemTelemetry(
                    state=self._state,
                    driving_goal_state=self._driving_goal_state,
                    experiment_mode=self._experiment_mode,
                    processed_observations=self._processed_observations,
                    action=action,
                )
            )

        # Need to keep this in sync with the policy's driving goal state since
        # derive_habitat_goal_state() consumes the goal state.
        self._driving_goal_state = getattr(self._policy, "driving_goal_state", None)
        self._last_action = self._policy.last_action
        self._n_steps += 1

    def __call__(self) -> Action:
        """Defines the structure for __call__.

        Delegates to the motor policy.

        Returns:
            The action to take.
        """
        # TODO: ?Mark a goal state being attempted as the one being attempted so
        # it can be checked by a GSG.?
        action = self._policy(self._state)
        self._post_call(action)
        return action

"""
---------------------------------
 - LookAtPolicy implementation

This isn't meant to live in this file long-term, but `motor_policies.py`
is already > 2k lines.
"""

class LookAtPolicy(BasePolicy):
    """A policy that looks at a target."""

    def __init__(self, rng):
        action_sampler_class = ConstantSampler
        action_sampler_args = dict(
            actions=generate_action_list("distant_agent_no_translation"),
            rotation_degrees=5.0,
        )
        agent_id = "agent_id_0"
        switch_frequency = 1.0

        super().__init__(
            rng=rng,
            action_sampler_args=action_sampler_args,
            action_sampler_class=action_sampler_class,
            agent_id=agent_id,
            switch_frequency=switch_frequency,
        )
        self.driving_goal_state = None
        self.processed_observations = None

    def set_driving_goal_state(self, goal_state: GoalState | None) -> None:
        self.driving_goal_state = goal_state

    def dynamic_call(self, state: MotorSystemState) -> Action:
        # Clean up habitat state.
        state = clean_motor_system_state(state)

        # Find target location relative to sensor.
        target_loc_rel_world = self.driving_goal_state.location

        # Construct transform chain that maps between world and sensor coordinates.
        agent_rot = as_rotation_matrix(state["agent_id_0"]["rotation"])
        agent_pos = state["agent_id_0"]["position"]
        sensor_rot = as_rotation_matrix(
            state["agent_id_0"]["sensors"]["view_finder"]["rotation"]
        )
        sensor_pos = state["agent_id_0"]["sensors"]["view_finder"]["position"]
        agent_transform = RigidTransform(agent_pos, agent_rot)
        sensor_transform = RigidTransform(sensor_pos, sensor_rot)
        chain = TransformChain([agent_transform, sensor_transform])

        # Map goal from world to sensor coordinates.
        target_rel_sensor = chain.inv()(self.driving_goal_state.location)

        # Convert from cartesion sensor coordinates to degrees.
        x_rot, y_rot, z_rot = target_rel_sensor
        left_amount = -np.degrees(np.arctan2(x_rot, -z_rot))
        distance_horiz = np.sqrt(x_rot**2 + z_rot**2)
        down_amount = -np.degrees(np.arctan2(y_rot, distance_horiz))

        return [
            LookDown(agent_id=self.agent_id, rotation_degrees=down_amount),
            TurnLeft(agent_id=self.agent_id, rotation_degrees=left_amount),
        ]


def clean_motor_system_state(state: dict) -> dict:
    """Clean up a Habitat motor system state dictionaries.

    Function that cleans up Habitat's MotorSystemState to a more usable format.
    For example, a single RGBD camera normally has separate, redundant positions
    and rotations. For example, "patch.depth" and "patch.rgba" are both present
    and contain the same rotation and position data. This function consolidates
    these into a single position and rotation for the sensor. Positions are also
    converted to the more usable numpy arrays (as opposed to magnum.Vector3 objects).

    Args:
        state: The motor system state to clean.

    Returns:
        The cleaned motor system state.
    """
    clean = {}
    for agent_id, agent_state in state.items():
        pos = agent_state["position"]
        rot = agent_state["rotation"]
        clean[agent_id] = {
            "position": np.array([pos.x, pos.y, pos.z]),
            "rotation": rot,
            "sensors": {},
        }
        sensors_dict = agent_state["sensors"]
        all_keys = list(sensors_dict.keys())
        sensor_ids = {k.split(".")[0] for k in all_keys}
        for sm_id in sensor_ids:
            sm_key = [k for k in all_keys if k.startswith(sm_id + ".")][0]
            pos = sensors_dict[sm_key]["position"]
            rot = sensors_dict[sm_key]["rotation"]
            clean[agent_id]["sensors"][sm_id] = {
                "position": np.array([pos.x, pos.y, pos.z]),
                "rotation": rot,
            }
    return clean


def as_rotation_matrix(
    obj: quaternion.quaternion | ArrayLike | SciPyRotation,
) -> np.ndarray:
    """Convert a rotation description to a rotation matrix.

    Helper function for `RigidTransform`.

    Args:
        obj: The rotation description to convert. This can be a quaternion, a
            scipy rotation, or a rotation matrix.

    Returns:
        The rotation matrix.
    """
    if isinstance(obj, SciPyRotation):
        scipy_rot = obj
    elif isinstance(obj, quaternion.quaternion):
        scipy_rot = SciPyRotation.from_quat(np.array([obj.x, obj.y, obj.z, obj.w]))
    else:
        scipy_rot = SciPyRotation.from_matrix(obj)
    return scipy_rot.as_matrix()


class RigidTransform:
    """A rigid transform (rotation + translation)."""

    def __init__(
        self, pos: ArrayLike, rot: quaternion.quaternion | ArrayLike | SciPyRotation
    ):
        self.pos = np.array(pos)
        self.rot = as_rotation_matrix(rot)

    def inv(self) -> RigidTransform:
        rot = self.rot.T
        pos = rot @ (-self.pos)
        return RigidTransform(pos, rot)

    def __call__(self, point: ArrayLike) -> np.ndarray:
        point = np.asarray(point)
        if point.ndim == 1:
            return self.rot @ point + self.pos
        else:
            return (self.rot @ point.T).T + self.pos

    def __repr__(self):
        return f"RigidTransform(pos={self.pos}, rot={self.rot})"


class TransformChain:
    """A chain of rigid transformations."""

    def __init__(self, transforms: Sequence[RigidTransform]):
        """Initialize the transform chain.

        The order of transforms is meant to resemble parent to child ordering in
        a graph. For example, the first transform in the chain would represent
        an agent's position and rotation relative to the world, and the second
        transform would represent a sensor, mounted on the agent, relative to the
        agent's position and rotation. In this example, the chain would transform
        data from the sensor's coordinate system to the world coordinate system.
        Going in the opposite direction (i.e., from world to sensor) can be done
        using the inverse of the chain.

        Args:
            transforms: The rigid transformations to chain.
        """
        self.transforms = list(transforms)

    def __call__(self, point: ArrayLike) -> np.ndarray:
        for transform in reversed(self.transforms):
            point = transform(point)
        return point

    def inv(self) -> TransformChain:
        tforms = list(reversed([t.inv() for t in self.transforms]))
        return TransformChain(tforms)

    def __repr__(self):
        return f"TransformChain({self.transforms})"
