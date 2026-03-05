# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from typing import Sequence

import numpy as np
import quaternion as qt

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.abstract_monty_classes import (
    AgentObservations,
    SensorObservation,
)
from tbp.monty.frameworks.models.motor_system_state import AgentState, SensorState
from tbp.monty.frameworks.sensors import SensorID

# TODO: Create a base Agent class?


class MultiSensorAgent:
    def __init__(
        self,
        agent_id: AgentID,
        sensor_ids: Sequence[SensorID],
        **kwargs,  # noqa: ARG002
    ):
        self.id = agent_id

        self._sensor_ids = sensor_ids

    @property
    def observations(self) -> AgentObservations:
        obs = AgentObservations()
        for sensor_id in self._sensor_ids:
            obs[sensor_id] = SensorObservation(
                depth=np.zeros((64, 64)),
                rgba=np.zeros((64, 64, 4)),
            )

        return obs

    @property
    def state(self) -> AgentState:
        sensor_states = {}
        for sensor_id in self._sensor_ids:
            sensor_states[sensor_id] = SensorState(
                position=(0, 0, 0),
                rotation=qt.quaternion(1, 0, 0, 0),
            )
        return AgentState(
            position=(0, 0, 0),
            rotation=qt.quaternion(1, 0, 0, 0),
            sensors=sensor_states,
        )
