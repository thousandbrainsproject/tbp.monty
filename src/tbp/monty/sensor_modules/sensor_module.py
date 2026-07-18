# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Protocol, Sequence

import numpy as np
import quaternion as qt
from typing_extensions import Self

from tbp.monty.cmp import Goal, Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.motor_system_state import AgentState, SensorState
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.observations import SensorObservation


@dataclass
class Payload:
    observation: SensorObservation
    percept: Message | None
    goals: list[Goal]


@dataclass
class TransformContext:
    rng: np.random.RandomState
    agent_state: AgentState
    sensor_state: SensorState
    motor_only_step: bool = False


class Transform(Protocol):
    """A transform that can be applied to a payload."""

    def __call__(
        self: Self,
        ctx: TransformContext,
        payload: Payload,
    ) -> Payload:
        """Apply the transform to the payload.

        Args:
            ctx: The transform context.
            payload: The payload to transform.

        Returns:
            A payload with the transformed observation, percept, and goals.
        """
        ...


class RuntimeSensorModule(Protocol):
    """Monty runtime interface to a Sensor Module."""

    def update_state(self, agent: AgentState) -> None:
        """Update the proprioceptive state for this Sensor Module.

        Args:
            agent: The proprioceptive state of this sensor module's Agent.
        """
        ...

    def step(
        self,
        ctx: RuntimeContext,
        observation: SensorObservation,
        motor_only_step: bool = False,
    ) -> Message | None:
        """Execute a time-step for the Sensor Module.

        Args:
            ctx: The runtime context.
            observation: Sensor observation.
            motor_only_step: Whether the current step is a motor-only step.

        Returns:
            An optional percept with features and morphological features.
        """
        ...

    def propose_goals(self) -> Collection[Goal]:
        """Return the goals proposed by this Sensor Module.

        Returns:
            A collection of proposed Goals.
        """
        ...


class SensorModule(RuntimeSensorModule):
    _agent_state: AgentState
    _goals: list[Goal]
    _sensor_id: SensorID
    _sensor_module_id: str
    _sensor_state: SensorState
    _transforms: Sequence[Transform]

    def __init__(
        self: Self,
        sensor_module_id: str,
        sensor_id: SensorID,
        transforms: Sequence[Transform],
    ) -> None:
        self._sensor_module_id = sensor_module_id
        self._sensor_id = sensor_id
        self._transforms = transforms

    @property
    def sensor_module_id(self) -> str:
        return self._sensor_module_id

    def propose_goals(self) -> Collection[Goal]:
        return self._goals

    def step(
        self: Self,
        ctx: RuntimeContext,
        observation: SensorObservation,
        motor_only_step: bool = False,
    ) -> Message | None:
        """Process an observation into a percept and goals.

        Args:
            ctx: The runtime context.
            observation: Sensor observation.
            motor_only_step: Whether the current step is a motor-only step.

        Returns:
            An optional percept with features and morphological features.
        """
        transform_ctx = TransformContext(
            rng=ctx.rng,
            agent_state=self._agent_state,
            sensor_state=self._sensor_state,
            motor_only_step=motor_only_step,
        )
        payload = Payload(observation=observation, percept=None, goals=[])
        for transform in self._transforms:
            payload = transform(transform_ctx, payload)
        self._goals = payload.goals
        return payload.percept

    def update_state(self: Self, agent: AgentState) -> None:
        self._agent_state = agent
        sensor = agent.sensors[self._sensor_id]
        self._sensor_state = SensorState(
            position=agent.position
            + qt.rotate_vectors(agent.rotation, sensor.position),
            rotation=agent.rotation * sensor.rotation,
        )
