# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.sensors import SensorID


@dataclass
class SensorState:
    """The proprioceptive state of a sensor."""

    position: Any  # TODO: Stop using magnum.Vector3 and decide on Monty standard
    """The sensor's position."""
    rotation: Any  # TODO: Stop using quaternion.quaternion and decide on Monty standard
    """The sensor's rotation."""


@dataclass
class AgentState:
    """The proprioceptive state of an agent."""

    sensors: dict[SensorID, SensorState]
    """The proprioceptive state of the agent's sensors.

    When part of an AgentState, the SensorState's position and rotation are relative to
    the agent's position and rotation.
    """
    position: Any  # TODO: Stop using magnum.Vector3 and decide on Monty standard
    """The agent's position."""
    rotation: Any  # TODO: Stop using quaternion.quaternion and decide on Monty standard
    """The agent's rotation."""
    motor_only_step: bool = False
    """Control flow parameter. Processing will bypass the learning module if True.

    TODO: Remove once we refactor Monty main processing loop to no longer need
    control flow parameters in motor system state.
    """


class ProprioceptiveState(Dict[AgentID, AgentState]):
    """The proprioceptive state of the motor system.

    When part of a ProprioceptiveState, the AgentState's position and rotation are
    relative to some global reference frame.
    """


class MotorSystemState(Dict[AgentID, AgentState]):
    """The state of the motor system.

    TODO: Currently, ProprioceptiveState can be cast to MotorSystemState since
          MotorSystemState is a generic dictionary. In the future, make
          ProprioceptiveState a param on MotorSystemState to more clearly distinguish
          between the two. These are separate from each other because
          ProprioceptiveState is the information returned from the environment, while
          MotorSystemState is that, as well as any state that the motor system
          needs for operation.
    """
