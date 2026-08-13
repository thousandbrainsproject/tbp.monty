# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Sensor processing for ARC-AGI palette-index patches."""

from __future__ import annotations

import numpy as np
import scipy

from tbp.monty.cmp import Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.abstract_monty_classes import (
    SensorModule,
    SensorObservation,
)
from tbp.monty.frameworks.models.motor_system_state import AgentState, SensorState
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.memento import Memento

__all__ = ["ArcPatchSensorModule"]

ARC_PALETTE_SIZE = 16
# Color mapping for frame values (0-15)
ARC_PALETTE = np.array(
    (
        (255, 255, 255, 255),  # 0: White "#FFFFFFFF"
        (204, 204, 204, 255),  # 1: Off-white "#CCCCCCFF"
        (153, 153, 153, 255),  # 2: Neutral Light "#999999FF"
        (102, 102, 102, 255),  # 3: Neutral "#666666FF"
        (51, 51, 51, 255),  # 4: Off Black "#333333FF"
        (0, 0, 0, 255),  # 5: Black "#000000FF"
        (229, 58, 163, 255),  # 6: Magenta "#E53AA3FF"
        (255, 123, 204, 255),  # 7: Magenta Light "#FF7BCCFF"
        (249, 60, 49, 255),  # 8: Red "#F93C31FF"
        (30, 147, 255, 255),  # 9: Blue "#1E93FFFF"
        (136, 216, 241, 255),  # 10: Blue Light "#88D8F1FF"
        (255, 220, 0, 255),  # 11: Yellow "#FFDC00FF"
        (255, 133, 27, 255),  # 12: Orange "#FF851BFF"
        (146, 18, 49, 255),  # 13: Maroon "#921231FF"
        (79, 204, 48, 255),  # 14: Green "#4FCC30FF"
        (163, 86, 214, 255),  # 15: Purple "#A356D6FF"
    ),
    dtype=np.uint8,
)
ARC_FRAME_POSE = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)


class ArcPatchSensorModule(SensorModule):
    """Convert one square ARC palette-index crop into a CMP message.

    The simulator supplies the crop as ``observation["raw"]`` and exposes its
    top-left corner as the patch sensor's pixel-space position. Transition density is
    ordered as changes across columns followed by changes across rows.
    """

    def __init__(self, sensor_module_id: str, patch_size: int = 8) -> None:
        if not isinstance(patch_size, int):
            raise TypeError("patch_size must be an integer")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")

        self.sensor_module_id = sensor_module_id
        self.patch_size = patch_size
        self.is_exploring = False
        self.state: SensorState | None = None

    def update_state(self, agent: AgentState) -> None:
        sensor = agent.sensors[SensorID(self.sensor_module_id)]
        self.state = SensorState(
            position=(
                float(sensor.position[0]),
                float(sensor.position[1]),
                float(sensor.position[2]),
            ),
            rotation=sensor.rotation,
        )

    def step(
        self,
        ctx: RuntimeContext,  # noqa: ARG002
        observation: SensorObservation,
        motor_only_step: bool = False,
    ) -> Message:
        if self.state is None:
            raise RuntimeError("update_state must be called before step")

        raw = observation.get("raw")
        if raw is None:
            raise ValueError("ARC patch observation must contain 'raw'")
        patch = np.asarray(raw)
        expected_shape = (self.patch_size, self.patch_size)
        if patch.shape != expected_shape:
            raise ValueError(
                f"Expected ARC patch shape {expected_shape}, received {patch.shape}"
            )
        if not np.issubdtype(patch.dtype, np.integer):
            raise ValueError(
                f"Expected integer ARC palette values, received {patch.dtype}"
            )
        if patch.min() < 0 or patch.max() >= ARC_PALETTE_SIZE:
            raise ValueError("ARC palette values must be between 0 and 15")

        histogram = (
            np.bincount(patch.ravel(), minlength=ARC_PALETTE_SIZE).astype(float)
            / patch.size
        )
        if self.patch_size == 1:
            transitions = np.zeros(2)
        else:
            transitions = np.array(
                [
                    np.mean(patch[:, 1:] != patch[:, :-1]),
                    np.mean(patch[1:, :] != patch[:-1, :]),
                ]
            )

        rgba = ARC_PALETTE[scipy.stats.mode(patch)[0]]
        return Message(
            location=np.asarray(self.state.position, dtype=float).copy(),
            morphological_features={
                "pose_vectors": ARC_FRAME_POSE.copy(),
                "pose_fully_defined": True,
            },
            non_morphological_features={
                "palette_histogram": histogram,
                "transition_density": transitions,
                "rgba": rgba,
            },
            confidence=1.0,
            use_state=not motor_only_step,
            sender_id=self.sensor_module_id,
            sender_type="SM",
        )

    def reset(self) -> None:
        self.is_exploring = False

    def state_dict(self) -> Memento:
        return {}
