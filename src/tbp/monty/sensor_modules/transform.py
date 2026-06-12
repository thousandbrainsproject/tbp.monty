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
from typing import Callable, Protocol, Sequence

import numpy as np
from typing_extensions import Self

from tbp.monty.cmp import Goal, Message
from tbp.monty.frameworks.models.abstract_monty_classes import SensorObservation
from tbp.monty.frameworks.models.motor_system_state import AgentState

__all__ = [
    "Transform",
    "TransformContext",
    "TransformMiddleware",
    "TransformPipeline",
    "identity_transform",
]

@dataclass
class TransformContext:
    rng: np.random.RandomState
    state: AgentState | None = None
    motor_only_step: bool = False

class Transform(Protocol):
    def __call__(
        self,
        ctx: TransformContext,
        observation: SensorObservation,
        percept: Message | None,
        goals: list[Goal],
    ) -> tuple[SensorObservation, Message | None, list[Goal]]:
        """Apply the transform to the observation.

        Args:
            ctx: The transform context.
            observation: The observation to transform.
            percept: The percept to transform.
            goals: The goals to transform.

        Returns:
            A tuple containing the transformed observation, percept, and goals.
        """
        ...

def identity_transform(
    ctx: TransformContext,  # noqa: ARG001
    observation: SensorObservation,
    percept: Message | None,
    goals: list[Goal],
) -> tuple[SensorObservation, Message | None, list[Goal]]:
    """Identity transform the returns the observation, percept, and goals unchanged.

    The main purpose of this transform is to be the last transform in the sequence
    when assembling a transform pipeline.

    Args:
        ctx: The transform context.
        observation: The observation to return.
        percept: The percept to return.
        goals: The goals to return.

    Returns:
        A tuple containing the observation, percept, and goals unchanged.
    """
    return observation, percept, goals

TransformMiddleware = Callable[[Transform], Transform]

class TransformPipeline(Transform):

    _transform: Transform

    def __init__(self: Self, transforms: Sequence[TransformMiddleware]) -> None:
        transform = identity_transform
        for next_transform in reversed(transforms):
            transform = next_transform(transform)
        self._transform = transform

    def __call__(
        self,
        ctx: TransformContext,
        observation: SensorObservation,
        percept: Message | None,
        goals: list[Goal],
    ) -> tuple[SensorObservation, Message | None, list[Goal]]:
        return self._transform(ctx, observation, percept, goals)
