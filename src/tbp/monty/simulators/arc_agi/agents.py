# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import numpy as np
import quaternion as qt
from arc_agi.rendering import COLOR_MAP

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
from tbp.monty.simulators.arc_agi.region_scan import SetArcRegionPose

if TYPE_CHECKING:
    from tbp.monty.simulators.arc_agi import ArcAgiSimulator


_ARC_RGBA_PALETTE = np.array(
    [tuple(bytes.fromhex(COLOR_MAP[index][1:])) for index in range(len(COLOR_MAP))],
    dtype=np.uint8,
)


class ArcAgent:
    def __init__(
        self,
        simulator: ArcAgiSimulator,
        agent_id: str,
        patch_resolution: Resolution2D,
        sensor_configs: Mapping[str, object],
        viewport_sensor_id: str = "view_finder",
    ):
        self.id = agent_id
        self._sim = simulator
        self._sensor_position = ZERO_VECTOR  # top-left corner of the patch crop
        self._display_sensor_position = ZERO_VECTOR
        self._active_region_id: str | None = None
        self._region_phase: str | None = None
        self._patch_res = patch_resolution
        self.viewport_sensor_id = SensorID(viewport_sensor_id)
        configured_sensor_ids = tuple(
            SensorID(sensor_id) for sensor_id in sensor_configs
        )
        if self.viewport_sensor_id not in configured_sensor_ids:
            raise ValueError(
                f"viewport_sensor_id {self.viewport_sensor_id!r} must be in "
                f"sensor_configs ({configured_sensor_ids!r})"
            )
        self._patch_ids = tuple(
            SensorID(sensor_id)
            for sensor_id in configured_sensor_ids
            if sensor_id != self.viewport_sensor_id
        )
        if not self._patch_ids:
            raise ValueError("sensor_configs must include at least one patch sensor")

    @property
    def state(self) -> AgentState:
        return AgentState(
            sensors={
                self.viewport_sensor_id: SensorState(
                    position=ZERO_VECTOR,
                    rotation=qt.quaternion(*IDENTITY_QUATERNION),
                ),
                **{
                    sensor_id: SensorState(
                        position=self._sensor_position,
                        rotation=qt.quaternion(*IDENTITY_QUATERNION),
                    )
                    for sensor_id in self._patch_ids
                },
            },
            position=ZERO_VECTOR,
            rotation=qt.quaternion(*IDENTITY_QUATERNION),
        )

    @property
    def observations(self) -> AgentObservations:
        obs = AgentObservations()
        frame_raw = np.asarray(self._sim.env.observation_space.frame[-1])
        frame_rgba = _ARC_RGBA_PALETTE[frame_raw]
        obs[self.viewport_sensor_id] = SensorObservation(
            raw=frame_raw,
            rgba=frame_rgba,
            oracle_regions=self._sim.oracle_regions,
        )
        x, y, _ = self._display_sensor_position
        x = int(x)
        y = int(y)
        patch_raw = frame_raw[
            y : y + self._patch_res.height, x : x + self._patch_res.width
        ]
        patch_rgba = frame_rgba[
            y : y + self._patch_res.height, x : x + self._patch_res.width
        ]
        for sensor_id in self._patch_ids:
            patch = SensorObservation(
                raw=patch_raw,
                rgba=patch_rgba,
                region_active=(
                    self._active_region_id is not None and self._region_phase == "scan"
                ),
            )
            if self._active_region_id is not None:
                region = self._sim.get_oracle_region(self._active_region_id)
                patch["region_id"] = region.region_id
                patch["object_label"] = region.object_label
                patch["region_phase"] = self._region_phase
                patch["region_anchor"] = region.global_anchor
                patch["region_location"] = self._sensor_position
            obs[sensor_id] = patch
        return obs

    def reset(self) -> None:
        self._sensor_position = ZERO_VECTOR
        self._display_sensor_position = ZERO_VECTOR
        self._active_region_id = None
        self._region_phase = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}"

    def actuate_set_sensor_pose(self, action: SetSensorPose) -> None:
        self._sensor_position = action.location
        self._display_sensor_position = action.location
        self._active_region_id = None
        self._region_phase = None

    def actuate_set_arc_region_pose(self, action: SetArcRegionPose) -> None:
        region = self._sim.get_oracle_region(action.region_id)
        x, y, z = action.display_location
        display_position = (int(x), int(y))
        if (
            z != 0
            or x != display_position[0]
            or y != display_position[1]
            or display_position not in region.display_positions
        ):
            raise ValueError(
                f"Display location {action.display_location!r} is not a visible "
                f"pixel in ARC oracle region {action.region_id!r}"
            )
        self._display_sensor_position = (float(x), float(y), 0.0)
        if action.phase == "scan":
            local_x = display_position[0] - region.display_origin[0]
            local_y = display_position[1] - region.display_origin[1]
            self._sensor_position = (float(local_x), float(local_y), 0.0)
        else:
            self._sensor_position = self._display_sensor_position
        self._active_region_id = region.region_id
        self._region_phase = action.phase

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
