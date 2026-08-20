# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Runtime environment for variable-shaped ARC sprite arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt
import quaternion as qt

from tbp.monty.frameworks.actions.actions import Action, SetSensorPose
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.environment import (
    ObjectID,
    ObjectInfo,
    SemanticID,
    SimulatedObjectEnvironment,
)
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
from tbp.monty.math import (
    IDENTITY_QUATERNION,
    ZERO_VECTOR,
    QuaternionWXYZ,
    VectorXYZ,
)

__all__ = ["ArcSpriteDatasetEnvironment"]


_ARC_RGBA_PALETTE = np.array(
    [
        [255, 255, 255, 255],
        [204, 204, 204, 255],
        [153, 153, 153, 255],
        [102, 102, 102, 255],
        [51, 51, 51, 255],
        [0, 0, 0, 255],
        [229, 58, 163, 255],
        [255, 123, 204, 255],
        [249, 60, 49, 255],
        [30, 147, 255, 255],
        [136, 216, 241, 255],
        [255, 220, 0, 255],
        [255, 133, 27, 255],
        [146, 18, 49, 255],
        [79, 204, 48, 255],
        [163, 86, 214, 255],
    ],
    dtype=np.uint8,
)


class ArcSpriteDatasetEnvironment(SimulatedObjectEnvironment):
    def __init__(
        self,
        data_path: str | Path,
        agent_id: str = "agent_id_0",
    ) -> None:
        self.data_path = Path(data_path).expanduser()
        self.agent_id = AgentID(agent_id)
        self._sprite_paths = {
            path.stem: path
            for path in sorted((self.data_path / "sprites").glob("*.npy"))
        }
        self.add_object(next(iter(self._sprite_paths)))

    @property
    def current_array(self) -> npt.NDArray[np.integer]:
        """Return the current sprite at its native effective shape."""
        return self._current_array.copy()

    def add_object(
        self,
        name: str,
        position: VectorXYZ = (0.0, 0.0, 0.0),  # noqa: ARG002
        rotation: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0),  # noqa: ARG002
        scale: VectorXYZ = (1.0, 1.0, 1.0),  # noqa: ARG002
        semantic_id: SemanticID | None = None,
        primary_target_object: ObjectID | None = None,  # noqa: ARG002
    ) -> ObjectInfo:
        """Load one native-shape sprite as the current object.

        Returns:
            Identifying information for the loaded sprite.
        """
        self._current_array = np.load(self._sprite_paths[name], allow_pickle=False)
        self._sensor_position = np.array(ZERO_VECTOR, dtype=float)
        return ObjectInfo(object_id=ObjectID(0), semantic_id=semantic_id)

    def remove_all_objects(self) -> None:
        """Clear the current object before the interface loads the next one."""

    def close(self) -> None:
        """Close the in-memory environment."""

    def step(
        self, actions: Sequence[Action]
    ) -> tuple[Observations, ProprioceptiveState]:
        """Apply sensor-pose actions and return current sprite observations.

        Returns:
            The current observations and proprioceptive state.
        """
        for action in actions:
            action.act(self)

        return self.observations, self.states

    def actuate_set_sensor_pose(self, action: SetSensorPose) -> None:
        """Move the pixel sensor to the requested effective coordinate."""
        self._sensor_position = np.asarray(action.location, dtype=float)

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        """Reset the pixel sensor to the top-left cell of the current sprite.

        Returns:
            The current observations and proprioceptive state.
        """
        self._sensor_position = np.array(ZERO_VECTOR, dtype=float)
        return self.observations, self.states

    @property
    def observations(self) -> Observations:
        """Return native-shape raw/RGBA viewport and a 1x1 patch."""
        frame_raw = self._current_array.copy()
        frame_rgba = self._to_rgba(frame_raw)
        x, y = self._pixel_position
        patch_raw = frame_raw[y : y + 1, x : x + 1].copy()
        patch_rgba = frame_rgba[y : y + 1, x : x + 1].copy()

        return Observations(
            {
                self.agent_id: AgentObservations(
                    {
                        SensorID("view_finder"): SensorObservation(
                            raw=frame_raw,
                            rgba=frame_rgba,
                        ),
                        SensorID("patch_0"): SensorObservation(
                            raw=patch_raw,
                            rgba=patch_rgba,
                        ),
                        SensorID("patch_1"): SensorObservation(
                            raw=patch_raw.copy(),
                            rgba=patch_rgba.copy(),
                        ),
                    }
                )
            }
        )

    @property
    def states(self) -> ProprioceptiveState:
        """Return sensor state whose positions are effective pixel coordinates."""
        rotation = qt.quaternion(*IDENTITY_QUATERNION)
        sensor_state = SensorState(
            position=self._sensor_position.copy(),
            rotation=rotation,
        )
        return ProprioceptiveState(
            {
                self.agent_id: AgentState(
                    sensors={
                        SensorID("view_finder"): sensor_state,
                        SensorID("patch_0"): SensorState(
                            position=self._sensor_position.copy(),
                            rotation=rotation,
                        ),
                        SensorID("patch_1"): SensorState(
                            position=self._sensor_position.copy(),
                            rotation=rotation,
                        ),
                    },
                    position=np.array(ZERO_VECTOR, dtype=float),
                    rotation=rotation,
                )
            }
        )

    @property
    def _pixel_position(self) -> tuple[int, int]:
        return int(self._sensor_position[0]), int(self._sensor_position[1])

    @staticmethod
    def _to_rgba(raw: npt.NDArray[np.integer]) -> npt.NDArray[np.uint8]:
        rgba = np.zeros((*raw.shape, 4), dtype=np.uint8)
        visible = raw >= 0
        rgba[visible] = _ARC_RGBA_PALETTE[raw[visible]]
        return rgba
