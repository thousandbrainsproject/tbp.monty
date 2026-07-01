# Copyright 2025-2026 Thousand Brains Project
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
)
from tbp.monty.frameworks.models.motor_system_state import AgentState
from tbp.monty.memento import Memento
from tbp.monty.observations import SensorObservation

__all__ = ["FakeSensorModule"]


class FakeSensorModule(SensorModule):
    """Dummy placeholder class used only for tests."""

    def __init__(self, sensor_module_id: str):
        super().__init__()
        self.sensor_module_id = sensor_module_id

    def state_dict(self) -> Memento:
        return {}

    def reset(self) -> None:
        pass

    def update_state(self, agent: AgentState) -> None:
        pass

    def step(
        self,
        ctx: RuntimeContext,  # noqa: ARG002
        observation: SensorObservation,
        motor_only_step: bool = False,  # noqa: ARG002
    ) -> Message:
        """Returns a dummy/placeholder message."""
        return Message(
            location=np.zeros(3),
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": True,
            },
            non_morphological_features=observation,
            confidence=1,
            use_state=True,
            sender_id=self.sensor_module_id,
            sender_type="SM",
        )
