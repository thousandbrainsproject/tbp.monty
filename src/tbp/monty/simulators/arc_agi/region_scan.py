# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Deterministic motor acquisition for one frozen ARC oracle region."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from tbp.monty.cmp import Goal, Message
from tbp.monty.context import RuntimeContext
from tbp.monty.experiment.motor_system import ExperimentMotorSystem
from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_policies import MotorPolicy, MotorPolicyResult
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.math import VectorXYZ
from tbp.monty.memento import Memento

if TYPE_CHECKING:
    from tbp.monty.simulators.arc_agi.simulator import ArcOracleRegion

__all__ = ["ArcRegionScanPolicy", "SetArcRegionPose"]


class SetArcRegionPose(Action):
    """Move an ARC patch sensor to one pixel in a frozen oracle region."""

    def __init__(
        self,
        agent_id: AgentID,
        region_id: str,
        display_location: VectorXYZ,
    ) -> None:
        super().__init__(agent_id)
        self.region_id = region_id
        self.display_location = display_location

    def act(self, actor) -> None:
        actor.actuate_set_arc_region_pose(self)


class ArcRegionScanPolicy(MotorPolicy):
    """Visit every visible pixel in one oracle region in sparse snake order."""

    def __init__(
        self,
        agent_id: AgentID,
        sensor_id: SensorID | str = "patch_0",
        frame_sensor_id: SensorID | str = "view_finder",
        region_index: int = 0,
    ) -> None:
        if not isinstance(region_index, int):
            raise TypeError("region_index must be an integer")
        if region_index < 0:
            raise ValueError("region_index must be non-negative")
        self.agent_id = AgentID(agent_id)
        self.sensor_id = SensorID(sensor_id)
        self.frame_sensor_id = SensorID(frame_sensor_id)
        self.region_index = region_index
        self.reset()

    @staticmethod
    def _snake_positions(region: ArcOracleRegion) -> tuple[tuple[int, int], ...]:
        rows: dict[int, list[int]] = defaultdict(list)
        for x, y in region.local_positions:
            rows[y].append(x)
        return tuple(
            (x, y)
            for row_index, (y, xs) in enumerate(sorted(rows.items()))
            for x in sorted(xs, reverse=bool(row_index % 2))
        )

    def _select_region(self, observations: Observations) -> ArcOracleRegion:
        regions = observations[self.agent_id][self.frame_sensor_id].get(
            "oracle_regions", ()
        )
        if not regions:
            raise RuntimeError("ARC region scan requires non-empty oracle_regions")
        if self._region_id is None:
            if self.region_index >= len(regions):
                raise IndexError(
                    f"region_index {self.region_index} is out of range for "
                    f"{len(regions)} ARC oracle regions"
                )
            region = regions[self.region_index]
            self._region_id = region.region_id
            return region
        for region in regions:
            if region.region_id == self._region_id:
                return region
        raise RuntimeError(f"Selected ARC oracle region {self._region_id!r} is stale")

    def __call__(
        self,
        ctx: RuntimeContext,  # noqa: ARG002
        observations: Observations,
        state: MotorSystemState,
        percept: Message,  # noqa: ARG002
        goal: Goal | None,  # noqa: ARG002
    ) -> MotorPolicyResult:
        region = self._select_region(observations)
        positions = self._snake_positions(region)
        if self._next_position_index:
            expected = (*positions[self._next_position_index - 1], 0.0)
            actual = state[self.agent_id].sensors[self.sensor_id].position
            if not np.array_equal(actual, expected):
                raise RuntimeError(
                    "ARC region scan coordinate mismatch: expected local sensor "
                    f"position {expected}, received {tuple(actual)}"
                )
        if self._next_position_index == len(positions):
            raise StopIteration

        local_x, local_y = positions[self._next_position_index]
        self._next_position_index += 1
        origin_x, origin_y = region.display_origin
        return MotorPolicyResult(
            [
                SetArcRegionPose(
                    agent_id=self.agent_id,
                    region_id=region.region_id,
                    display_location=(
                        float(origin_x + local_x),
                        float(origin_y + local_y),
                        0.0,
                    ),
                )
            ]
        )

    def fixme_provide_motor_system(self, motor_system: ExperimentMotorSystem) -> None:
        pass

    def reset(self) -> None:
        self._region_id: str | None = None
        self._next_position_index = 0

    def state_dict(self) -> Memento:
        return {
            "region_id": self._region_id,
            "next_position_index": self._next_position_index,
        }

    def load_state_dict(self, memento: Memento) -> None:
        self._region_id = memento["region_id"]
        self._next_position_index = memento["next_position_index"]
