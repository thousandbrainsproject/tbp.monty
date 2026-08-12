# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import numpy as np

from tbp.monty.cmp import Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.abstract_monty_classes import (
    SensorModule,
    SensorObservation,
)
from tbp.monty.frameworks.models.arc_sensor_module import ARC_FRAME_POSE
from tbp.monty.frameworks.models.motor_system_state import AgentState
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.memento import Memento

__all__ = ["ArcSpritePixelSensorModule"]


class ArcSpritePixelSensorModule(SensorModule):
    """Convert one ARC palette-index pixel into a CMP message."""

    def __init__(self, sensor_module_id: str) -> None:
        self.sensor_module_id = sensor_module_id
        self.is_exploring = False
        self.location = np.zeros(3)

    def update_state(self, agent: AgentState) -> None:
        self.location = np.asarray(
            agent.sensors[SensorID(self.sensor_module_id)].position,
            dtype=float,
        )

    def step(
        self,
        ctx: RuntimeContext,  # noqa: ARG002
        observation: SensorObservation,
        motor_only_step: bool = False,
    ) -> Message:
        palette_index = int(observation["raw"][0, 0])
        visible = palette_index >= 0
        return Message(
            location=self.location.copy(),
            morphological_features={
                "pose_vectors": ARC_FRAME_POSE.copy(),
                "pose_fully_defined": True,
            },
            non_morphological_features=(
                {"palette_index": np.array([palette_index], dtype=float)}
                if visible
                else {}
            ),
            confidence=1.0,
            use_state=visible and not motor_only_step,
            sender_id=self.sensor_module_id,
            sender_type="SM",
        )

    def reset(self) -> None:
        self.is_exploring = False

    def state_dict(self) -> Memento:
        return {}
