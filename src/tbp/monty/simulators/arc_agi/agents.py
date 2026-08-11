# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import TYPE_CHECKING

import quaternion as qt

from tbp.monty.frameworks.actions.actions import SetSensorPose
from tbp.monty.frameworks.models.abstract_monty_classes import (
    AgentObservations,
    SensorObservation,
)
from tbp.monty.frameworks.models.motor_system_state import AgentState, SensorState
from tbp.monty.frameworks.sensors import Resolution2D, SensorID
from tbp.monty.math import IDENTITY_QUATERNION, ZERO_VECTOR
from tbp.monty.simulators.arc_agi.actions import (
    GameClick,
    GameDown,
    GameLeft,
    GameReset,
    GameRight,
    GameUndo,
    GameUp,
    GameUse,
)

if TYPE_CHECKING:
    from tbp.monty.simulators.arc_agi import ArcAgiSimulator


class ArcAgent:
    VIEWPORT = SensorID("view_finder")
    PATCH = SensorID("patch")

    def __init__(
        self,
        simulator: ArcAgiSimulator,
        agent_id: str,
        patch_resolution: Resolution2D,
        sensor_configs,
    ):
        self.id = agent_id
        self._sim = simulator
        self._sensor_position = ZERO_VECTOR
        self._patch_res = patch_resolution

    @property
    def state(self) -> AgentState:
        return AgentState(
            sensors={
                self.VIEWPORT: SensorState(
                    position=ZERO_VECTOR,
                    rotation=qt.quaternion(*IDENTITY_QUATERNION),
                ),
                self.PATCH: SensorState(
                    position=self._sensor_position,
                    rotation=qt.quaternion(*IDENTITY_QUATERNION),
                ),
            },
            position=ZERO_VECTOR,
            rotation=qt.quaternion(*IDENTITY_QUATERNION),
        )

    @property
    def observations(self) -> AgentObservations:
        obs = AgentObservations()
        frame_raw = self._sim.env.observation_space.frame[-1]
        obs[self.VIEWPORT] = SensorObservation(raw=frame_raw)
        x, y, _ = self._sensor_position
        x = int(x)
        y = int(y)
        patch_raw = frame_raw[
            y : y + self._patch_res.height, x : x + self._patch_res.width
        ]
        obs[self.PATCH] = SensorObservation(raw=patch_raw)
        return obs

    def reset(self):
        self._sensor_position = ZERO_VECTOR

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}"

    def actuate_set_sensor_pose(self, action: SetSensorPose) -> None:
        self._sensor_position = action.location

    def actuate_game_reset(self, action: GameReset) -> None:
        self._sim.env.step(action=action.arc_action)

    def actuate_game_up(self, action: GameUp) -> None:
        self._sim.env.step(action=action.arc_action)

    def actuate_game_down(self, action: GameDown) -> None:
        self._sim.env.step(action=action.arc_action)

    def actuate_game_left(self, action: GameLeft) -> None:
        self._sim.env.step(action=action.arc_action)

    def actuate_game_right(self, action: GameRight) -> None:
        self._sim.env.step(action=action.arc_action)

    def actuate_game_use(self, action: GameUse) -> None:
        self._sim.env.step(action=action.arc_action)

    def actuate_game_click(self, action: GameClick) -> None:
        data = {"x": action.x, "y": action.y}
        self._sim.env.step(action=action.arc_action, data=data)

    def actuate_game_undo(self, action: GameUndo) -> None:
        self._sim.env.step(action=action.arc_action)
