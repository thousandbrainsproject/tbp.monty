# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Expose ARC-AGI frames as frozen sensor environments."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import quaternion as qt

from tbp.monty.frameworks.actions.actions import Action, SetSensorPose
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.environment import SimulatedEnvironment
from tbp.monty.frameworks.models.abstract_monty_classes import (
    AgentObservations,
    Observations,
    SensorObservation,
)
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    ProprioceptiveState,
    SensorState,
)
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.math import IDENTITY_QUATERNION, ZERO_VECTOR, VectorXYZ

__all__ = ["ArcFrameEnvironment"]

ARC_PALETTE_SIZE = 16


class ArcFrameEnvironment(SimulatedEnvironment):
    """Expose one frozen ARC frame through a movable square patch sensor."""

    def __init__(
        self,
        agent_id: AgentID | str = "agent_id_0",
        sensor_id: SensorID | str = "patch",
        frame_size: int = 64,
        patch_size: int = 8,
    ) -> None:
        for name, value in (("frame_size", frame_size), ("patch_size", patch_size)):
            if not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if patch_size > frame_size:
            raise ValueError("patch_size must not exceed frame_size")

        self.agent_id = AgentID(agent_id)
        self.sensor_id = SensorID(sensor_id)
        self.frame_size = frame_size
        self.patch_size = patch_size
        self._frame: np.ndarray | None = None
        self._sensor_location = self._initial_location()

    def _set_frames(self, frames: np.ndarray) -> None:
        frames = np.asarray(frames)
        expected_shape = (self.frame_size, self.frame_size)
        if (
            frames.ndim != 3
            or frames.shape[0] == 0
            or frames.shape[1:] != expected_shape
        ):
            raise ValueError(
                "Expected ARC frame shape "
                f"(n, {self.frame_size}, {self.frame_size}), received {frames.shape}"
            )
        if not np.issubdtype(frames.dtype, np.integer):
            raise ValueError(
                f"Expected integer ARC palette values, received {frames.dtype}"
            )
        if frames.min() < 0 or frames.max() >= ARC_PALETTE_SIZE:
            raise ValueError("ARC palette values must be between 0 and 15")

        frame = frames[-1].astype(np.uint8, copy=True)
        frame.flags.writeable = False
        self._frame = frame
        self._sensor_location = self._initial_location()

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        """Return the frozen frame's first patch and reset sensor state."""
        self._require_frame()
        self._sensor_location = self._initial_location()
        return self._observations(), self._state()

    def step(
        self, actions: Sequence[Action]
    ) -> tuple[Observations, ProprioceptiveState]:
        """Apply patch-sensor moves without changing the frozen ARC frame.

        Returns:
            The patch observation and updated proprioceptive state.
        """
        self._require_frame()
        for action in actions:
            self._set_sensor_pose(action)
        return self._observations(), self._state()

    def close(self) -> None:
        self._frame = None

    def _initial_location(self) -> VectorXYZ:
        center = float(self.patch_size // 2)
        return (center, center, 0.0)

    def _require_frame(self) -> np.ndarray:
        if self._frame is None:
            raise RuntimeError("An ARC frame must be loaded before reset or step")
        return self._frame

    def _set_sensor_pose(self, action: Action) -> None:
        if not isinstance(action, SetSensorPose):
            raise TypeError("ArcFrameEnvironment only accepts SetSensorPose")
        if action.agent_id != self.agent_id:
            raise ValueError(
                f"Action agent {action.agent_id} does not match {self.agent_id}"
            )

        rotation = np.asarray(action.rotation_quat, dtype=float)
        if rotation.shape != (4,) or not np.allclose(rotation, IDENTITY_QUATERNION):
            raise ValueError("ARC patch rotation must be the identity quaternion")

        location = np.asarray(action.location, dtype=float)
        if (
            location.shape != (3,)
            or not np.isfinite(location).all()
            or not np.equal(location, np.round(location)).all()
            or location[2] != 0
        ):
            raise ValueError("ARC patch location must be finite integer (x, y, 0)")

        x, y, z = (int(value) for value in location)
        half_patch = self.patch_size // 2
        x_start = x - half_patch
        y_start = y - half_patch
        if (
            x_start < 0
            or y_start < 0
            or x_start + self.patch_size > self.frame_size
            or y_start + self.patch_size > self.frame_size
        ):
            raise ValueError(f"ARC patch centered at {(x, y, z)} is outside the frame")

        self._sensor_location = (float(x), float(y), float(z))

    def _observations(self) -> Observations:
        frame = self._require_frame()
        x, y, _ = (int(value) for value in self._sensor_location)
        half_patch = self.patch_size // 2
        x_start = x - half_patch
        y_start = y - half_patch
        patch = frame[
            y_start : y_start + self.patch_size,
            x_start : x_start + self.patch_size,
        ].copy()
        return Observations(
            {
                self.agent_id: AgentObservations(
                    {self.sensor_id: SensorObservation({"raw": patch})}
                )
            }
        )

    def _state(self) -> ProprioceptiveState:
        return ProprioceptiveState(
            {
                self.agent_id: AgentState(
                    sensors={
                        self.sensor_id: SensorState(
                            position=self._sensor_location,
                            rotation=qt.one,
                        )
                    },
                    position=ZERO_VECTOR,
                    rotation=qt.one,
                )
            }
        )
