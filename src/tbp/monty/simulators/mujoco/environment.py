# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence

import quaternion as qt

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.agents import Agent
from tbp.monty.frameworks.environments.environment import (
    ObjectID,
    SemanticID,
    SimulatedObjectEnvironment,
)
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_system_state import ProprioceptiveState
from tbp.monty.math import IDENTITY_QUATERNION, ZERO_VECTOR, QuaternionWXYZ, VectorXYZ
from tbp.monty.simulators.mujoco.simulator import MuJoCoSimulator

if TYPE_CHECKING:
    from pathlib import Path

AgentPartial = Callable[[MuJoCoSimulator], Agent]


class MuJoCoEnvironment(SimulatedObjectEnvironment):
    def __init__(
        self,
        agents: Sequence[AgentPartial],
        data_path: str | Path | None,
    ):
        self._sim = MuJoCoSimulator(agents, data_path)

    def step(
        self,
        actions: Sequence[Action],
    ) -> tuple[Observations, ProprioceptiveState]:
        return self._sim.step(actions)

    def close(self) -> None:
        self._sim.close()

    def add_object(
        self,
        name: str,
        position: VectorXYZ = ZERO_VECTOR,
        rotation: QuaternionWXYZ = IDENTITY_QUATERNION,
        scale: VectorXYZ = (1.0, 1.0, 1.0),
        semantic_id: SemanticID | None = None,
        primary_target_object: ObjectID | None = None,
    ) -> ObjectID:
        # TODO: move this up the call chain since this method's argument types are lying
        if isinstance(rotation, qt.quaternion):
            rotation = (
                rotation.w,
                rotation.x,
                rotation.y,
                rotation.z,
            )
        return self._sim.add_object(
            name, position, rotation, scale, semantic_id, primary_target_object
        ).object_id

    def remove_all_objects(self) -> None:
        self._sim.remove_all_objects()

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        return self._sim.reset()
