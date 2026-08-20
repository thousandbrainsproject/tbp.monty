# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Deterministic motor acquisition for frozen ARC oracle regions."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Literal

import numpy as np

from tbp.monty.cmp import Goal, Message
from tbp.monty.context import RuntimeContext
from tbp.monty.experiment.motor_system import ExperimentMotorSystem
from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_policies import (
    MotorPolicy,
    MotorPolicyResult,
    SnakeScanPolicy,
)
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.math import VectorXYZ
from tbp.monty.memento import Memento

if TYPE_CHECKING:
    from tbp.monty.simulators.arc_agi.simulator import ArcOracleRegion

__all__ = ["ArcCompositionalScanPolicy", "ArcRegionScanPolicy", "SetArcRegionPose"]

RegionPhase = Literal["scan", "emit", "skip"]


class SetArcRegionPose(Action):
    """Move an ARC patch sensor within one frozen oracle region."""

    def __init__(
        self,
        agent_id: AgentID,
        region_id: str,
        display_location: VectorXYZ,
        phase: RegionPhase = "scan",
    ) -> None:
        super().__init__(agent_id)
        if phase not in {"scan", "emit", "skip"}:
            raise ValueError(f"Unknown ARC region phase {phase!r}")
        self.region_id = region_id
        self.display_location = display_location
        self.phase = phase

    def act(self, actor) -> None:
        actor.actuate_set_arc_region_pose(self)


class ArcCompositionalScanPolicy(MotorPolicy):
    """Scan the whole display, then recognize and emit oracle regions."""

    def __init__(
        self,
        agent_id: AgentID,
        child_sensor_id: SensorID | str = "patch_0",
        map_sensor_id: SensorID | str = "patch_1",
        frame_sensor_id: SensorID | str = "view_finder",
        frame_size: int | None = None,
        region_index: int = 0,
        recognition_passes: int = 2,
    ) -> None:
        self._map_scan = SnakeScanPolicy(
            agent_id=agent_id,
            sensor_id=map_sensor_id,
            frame_sensor_id=frame_sensor_id,
            frame_size=frame_size,
            patch_size=1,
            stride=1,
        )
        self._region_scan = ArcRegionScanPolicy(
            agent_id=agent_id,
            sensor_id=child_sensor_id,
            frame_sensor_id=frame_sensor_id,
            region_index=region_index,
            recognition_passes=recognition_passes,
        )
        self.reset()

    def __call__(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        state: MotorSystemState,
        percept: Message,
        goal: Goal | None,
    ) -> MotorPolicyResult:
        if self._phase == "map":
            try:
                return self._map_scan(ctx, observations, state, percept, goal)
            except StopIteration:
                self._phase = "regions"
        return self._region_scan(ctx, observations, state, percept, goal)

    def fixme_provide_motor_system(self, motor_system: ExperimentMotorSystem) -> None:
        self._map_scan.fixme_provide_motor_system(motor_system)
        self._region_scan.fixme_provide_motor_system(motor_system)

    def reset(self) -> None:
        self._phase = "map"
        self._map_scan.reset()
        self._region_scan.reset()

    def state_dict(self) -> Memento:
        return {
            "phase": self._phase,
            "map_scan": self._map_scan.state_dict(),
            "region_scan": self._region_scan.state_dict(),
        }

    def load_state_dict(self, memento: Memento) -> None:
        self._phase = memento["phase"]
        self._map_scan.load_state_dict(memento["map_scan"])
        self._region_scan.load_state_dict(memento["region_scan"])


class ArcRegionScanPolicy(MotorPolicy):
    """Recognize each oracle region, then emit its identity at every pixel.

    Recognition traverses the region in sparse snake order multiple times. The
    first emission action is initially marked ``skip``; the ARC Monty lifecycle
    promotes it to ``emit`` only when a child identity is available. Subsequent
    emission actions preserve that decision.
    """

    def __init__(
        self,
        agent_id: AgentID,
        sensor_id: SensorID | str = "patch_0",
        frame_sensor_id: SensorID | str = "view_finder",
        region_index: int = 0,
        recognition_passes: int = 2,
    ) -> None:
        if not isinstance(region_index, int):
            raise TypeError("region_index must be an integer")
        if region_index < 0:
            raise ValueError("region_index must be non-negative")
        if not isinstance(recognition_passes, int):
            raise TypeError("recognition_passes must be an integer")
        if recognition_passes < 1:
            raise ValueError("recognition_passes must be positive")
        self.agent_id = AgentID(agent_id)
        self.sensor_id = SensorID(sensor_id)
        self.frame_sensor_id = SensorID(frame_sensor_id)
        self.region_index = region_index
        self.recognition_passes = recognition_passes
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

    @staticmethod
    def _current_phase(observations: Observations, agent_id, sensor_id) -> str | None:
        return observations[agent_id].get(sensor_id, {}).get("region_phase")

    def _regions(self, observations: Observations) -> tuple[ArcOracleRegion, ...]:
        regions = observations[self.agent_id][self.frame_sensor_id].get(
            "oracle_regions", ()
        )
        if not regions:
            raise RuntimeError("ARC region scan requires non-empty oracle_regions")
        return regions

    def _region(self, regions: tuple[ArcOracleRegion, ...]) -> ArcOracleRegion:
        if self._region_index >= len(regions):
            if self._region_id is None:
                raise IndexError(
                    f"region_index {self._region_index} is out of range for "
                    f"{len(regions)} ARC oracle regions"
                )
            raise StopIteration
        region = regions[self._region_index]
        if self._region_id is None:
            self._region_id = region.region_id
        elif region.region_id != self._region_id:
            raise RuntimeError(
                f"Selected ARC oracle region {self._region_id!r} is stale"
            )
        return region

    def _validate_previous_location(
        self, state: MotorSystemState, phase: str | None
    ) -> None:
        if self._requested_location is None:
            return
        actual = state[self.agent_id].sensors[self.sensor_id].position
        if not np.array_equal(actual, self._requested_location):
            coordinate_space = "local" if phase == "scan" else "display"
            raise RuntimeError(
                "ARC region scan coordinate mismatch: expected "
                f"{coordinate_space} sensor position {self._requested_location}, "
                f"received {tuple(actual)}"
            )

    def _result(
        self,
        region: ArcOracleRegion,
        local_position: tuple[int, int],
        phase: RegionPhase,
    ) -> MotorPolicyResult:
        local_x, local_y = local_position
        origin_x, origin_y = region.display_origin
        display_location = (
            float(origin_x + local_x),
            float(origin_y + local_y),
            0.0,
        )
        self._requested_location = (
            (*local_position, 0.0) if phase == "scan" else display_location
        )
        return MotorPolicyResult(
            [
                SetArcRegionPose(
                    agent_id=self.agent_id,
                    region_id=region.region_id,
                    display_location=display_location,
                    phase=phase,
                )
            ]
        )

    def _start_next_region(
        self, regions: tuple[ArcOracleRegion, ...]
    ) -> MotorPolicyResult:
        self._region_index += 1
        if self._region_index >= len(regions):
            self._stage = "done"
            raise StopIteration
        self._region_id = regions[self._region_index].region_id
        self._recognition_pass_index = 0
        self._next_position_index = 1
        self._next_emission_index = 0
        self._stage = "scan"
        return self._result(
            regions[self._region_index],
            self._snake_positions(regions[self._region_index])[0],
            "scan",
        )

    def __call__(
        self,
        ctx: RuntimeContext,  # noqa: ARG002
        observations: Observations,
        state: MotorSystemState,
        percept: Message,  # noqa: ARG002
        goal: Goal | None,  # noqa: ARG002
    ) -> MotorPolicyResult:
        regions = self._regions(observations)
        region = self._region(regions)
        phase = self._current_phase(observations, self.agent_id, self.sensor_id)
        self._validate_previous_location(state, phase)
        positions = self._snake_positions(region)

        if self._stage == "scan":
            path = positions[:: (-1 if self._recognition_pass_index % 2 else 1)]
            if self._next_position_index < len(path):
                position = path[self._next_position_index]
                self._next_position_index += 1
                return self._result(region, position, "scan")
            if self._recognition_pass_index + 1 < self.recognition_passes:
                self._recognition_pass_index += 1
                self._next_position_index = 1
                path = positions[::-1]
                if self._recognition_pass_index % 2 == 0:
                    path = positions
                return self._result(region, path[0], "scan")
            self._stage = "emission"
            self._next_emission_index = 1
            return self._result(region, positions[0], "skip")

        if self._stage == "emission":
            if phase == "emit" and self._next_emission_index < len(positions):
                position = positions[self._next_emission_index]
                self._next_emission_index += 1
                return self._result(region, position, "emit")
            return self._start_next_region(regions)

        raise StopIteration

    def fixme_provide_motor_system(self, motor_system: ExperimentMotorSystem) -> None:
        pass

    def reset(self) -> None:
        self._region_index = self.region_index
        self._region_id: str | None = None
        self._recognition_pass_index = 0
        self._next_position_index = 0
        self._next_emission_index = 0
        self._stage = "scan"
        self._requested_location: VectorXYZ | None = None

    def state_dict(self) -> Memento:
        return {
            "region_index": self._region_index,
            "region_id": self._region_id,
            "recognition_pass_index": self._recognition_pass_index,
            "next_position_index": self._next_position_index,
            "next_emission_index": self._next_emission_index,
            "stage": self._stage,
            "requested_location": self._requested_location,
        }

    def load_state_dict(self, memento: Memento) -> None:
        self._region_index = memento["region_index"]
        self._region_id = memento["region_id"]
        self._recognition_pass_index = memento["recognition_pass_index"]
        self._next_position_index = memento["next_position_index"]
        self._next_emission_index = memento["next_emission_index"]
        self._stage = memento["stage"]
        self._requested_location = memento["requested_location"]
