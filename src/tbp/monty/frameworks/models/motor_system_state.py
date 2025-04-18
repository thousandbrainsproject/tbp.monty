# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from typing import Any, Dict, TypedDict


class SensorState(TypedDict):
    """The proprioceptive state of a sensor."""

    position: Any  # TODO: Stop using magnum.Vector3 and decide on Monty standard
    """The sensor's position relative to the agent."""
    rotation: Any  # TODO: Stop using quaternion.quaternion and decide on Monty standard
    """The sensor's rotation relative to the agent."""


class AgentState(TypedDict):
    """The proprioceptive state of an agent."""

    sensors: Dict[str, SensorState]
    """The proprioceptive state of the agent's sensors."""
    position: Any  # TODO: Stop using magnum.Vector3 and decide on Monty standard
    """The agent's position relative to some global reference frame."""
    rotation: Any  # TODO: Stop using quaternion.quaternion and decide on Monty standard
    """The agent's rotation relative to some global reference frame."""


MotorSystemState = Dict[str, AgentState]
"""The proprioceptive state of the motor system."""
