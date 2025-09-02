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
from typing import Literal

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
            policy: The motor policy to use.
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
                if self._n_steps < 100:
                    return self._look_at_policy
                self._driving_goal_state = None

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
        state = clean_motor_system_state(state)
        poses = walk_poses(state, ("agent_id_0", "sensors", "view_finder"))

        # Find target location relative to sensor.
        target_loc_rel_world = self.driving_goal_state.location

        """"
        p[A] = B.rot[A] * p[B] + B.origin[A]

        invert this

        B.rot[A].inv * (p[A] - B.origin[A]) = p[B]  <--- need to do this
        """
        target_loc_rel_sensor = target_loc_rel_world
        for node in reversed(poses):
            target_loc_rel_sensor = target_loc_rel_sensor - node["position"]

        # Find sensor rotation relative to world.
        sensor_rot_rel_world = quaternion.quaternion(1, 0, 0, 0)
        for node in poses:
            sensor_rot_rel_world = sensor_rot_rel_world * node["rotation"]

        # Rotate target location relative to sensor to world coordinates.
        w, x, y, z = sensor_rot_rel_world.components
        sensor_rot_rel_world = Rotation.from_quat([x, y, z, w])
        rotated_location = sensor_rot_rel_world.inv().apply(target_loc_rel_sensor)

        # Calculate the necessary rotation amounts and convert them to degrees.
        x_rot, y_rot, z_rot = rotated_location
        left_amount = -np.degrees(np.arctan2(x_rot, -z_rot))
        distance_horiz = np.sqrt(x_rot**2 + z_rot**2)
        down_amount = -np.degrees(np.arctan2(y_rot, distance_horiz))

        # ---------------------------------
        _agent_rot = as_rotation_matrix(state["agent_id_0"]["rotation"])
        _agent_pos = state["agent_id_0"]["position"]
        _sensor_rot = as_rotation_matrix(
            state["agent_id_0"]["sensors"]["view_finder"]["rotation"]
        )
        _sensor_pos = state["agent_id_0"]["sensors"]["view_finder"]["position"]
        _agent_transform = RigidTransform(_agent_pos, _agent_rot)
        _sensor_transform = RigidTransform(_sensor_pos, _sensor_rot)
        _chain = TransformChain([_agent_transform, _sensor_transform])
        _chain_inv = _chain.inv()

        _target_loc_rel_sensor = _chain_inv(target_loc_rel_world)
        _x_rot, _y_rot, _z_rot = _target_loc_rel_sensor
        _left_amount = -np.degrees(np.arctan2(_x_rot, -_z_rot))
        _distance_horiz = np.sqrt(_x_rot**2 + _z_rot**2)
        _down_amount = -np.degrees(np.arctan2(_y_rot, _distance_horiz))
        _ar = SciPyRotation.from_matrix(_agent_rot)
        _sr = SciPyRotation.from_matrix(_sensor_rot)
        _sw = _ar * _sr
        print(f"ar: {_ar.as_euler('xyz', degrees=True)}")
        print(f"sr: {_sr.as_euler('xyz', degrees=True)}")
        print(f"sw: {_sw.as_euler('xyz', degrees=True)}")
        print(f"left_amount: {left_amount}, down_amount: {down_amount}")
        print(f"_left_amount: {_left_amount}, _down_amount: {_down_amount}")
        print(target_loc_rel_world)

        # ---------------------------------
        return [
            LookDown(agent_id=self.agent_id, rotation_degrees=down_amount),
            TurnLeft(agent_id=self.agent_id, rotation_degrees=left_amount),
        ]


def clean_motor_system_state(state: dict) -> dict:
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


def walk_poses(state: dict, path: tuple[str]) -> list[dict]:
    """Iterate through a path of sensor ids in a state."""
    poses = []
    dct = state
    for part in path:
        dct = dct[part]
        if "position" in dct and "rotation" in dct:
            poses.append(
                {
                    "position": dct["position"],
                    "rotation": dct["rotation"],
                }
            )
    return poses


def as_scipy_rotation(
    obj: quaternion.quaternion | ArrayLike | SciPyRotation,
) -> SciPyRotation:
    if isinstance(obj, SciPyRotation):
        rot = obj
    elif isinstance(obj, quaternion.quaternion):
        rot = SciPyRotation.from_quat(np.array([obj.x, obj.y, obj.z, obj.w]))
    else:
        rot = SciPyRotation.from_matrix(obj)
    return rot


def as_rotation_matrix(
    obj: quaternion.quaternion | ArrayLike | SciPyRotation,
) -> np.ndarray:
    rot = as_scipy_rotation(obj)
    return rot.as_matrix()


def repr_vec(vec):
    return f"[{vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f}]"


class RigidTransform:
    def __init__(self, pos: np.ndarray, rot: np.ndarray):
        self.pos = np.array(pos)
        self.rot = as_rotation_matrix(rot)

    def inv(self) -> RigidTransform:
        rot = self.rot.T
        pos = rot @ (-self.pos)
        return RigidTransform(pos, rot)

    def __call__(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point)
        if point.ndim == 1:
            return self.rot @ point + self.pos
        else:
            return (self.rot @ point.T).T + self.pos

    def __repr__(self):
        return f"RigidTransform(pos={self.pos}, rot={self.rot})"


class TransformChain:
    def __init__(self, transforms: list[RigidTransform]):
        self.transforms = transforms

    def __call__(self, point: np.ndarray) -> np.ndarray:
        for transform in reversed(self.transforms):
            point = transform(point)
        return point

    def inv(self) -> TransformChain:
        tforms = list(reversed([t.inv() for t in self.transforms]))
        return TransformChain(tforms)

    def __repr__(self):
        return f"TransformChain({self.transforms})"
