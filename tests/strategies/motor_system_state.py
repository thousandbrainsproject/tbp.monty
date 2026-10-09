# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import hypothesis.strategies as st

from tbp.monty.frameworks.models.motor_system_state import AgentState, SensorState
from tbp.monty.frameworks.sensors import SensorID
from tests.strategies import sensors
from tests.strategies.geometry import position, rotation_non_zero


@st.composite
def agent_state(draw: st.DrawFn) -> AgentState:
    return AgentState(
        sensors=draw(st.dictionaries(sensors.sensor_id(), sensor_state(), max_size=5)),
        position=draw(position()),
        rotation=draw(rotation_non_zero()),
    )


@st.composite
def agent_state_with_sensor(draw: st.DrawFn, sensor_id: SensorID) -> AgentState:
    sensor_state_value = draw(sensor_state())
    return AgentState(
        sensors={
            **draw(st.dictionaries(sensors.sensor_id(), sensor_state(), max_size=5)),
            sensor_id: sensor_state_value,
        },
        position=draw(position()),
        rotation=draw(rotation_non_zero()),
    )


@st.composite
def sensor_state(draw: st.DrawFn) -> SensorState:
    return SensorState(
        position=draw(position()),
        rotation=draw(rotation_non_zero()),
    )
