# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import json
import unittest

from tbp.monty.attention.attention_system import (
    DEFAULT_VOXEL_SIZE,
    AttentionSystem,
)
from tbp.monty.attention.telemetry import (
    AttentionSystemTelemetry,
    NoopAttentionSystemTelemetry,
)
from tbp.monty.frameworks.models.buffer import BufferEncoder

from .attention_system_test import goal_at, point_in, region

NEAR_VOXEL = (0, 0, 0)
FAR_VOXEL = (50, 0, 0)
NEAR_POINT = point_in(NEAR_VOXEL)
FAR_POINT = point_in(FAR_VOXEL)


class AttentionSystemTelemetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.telemetry = AttentionSystemTelemetry()
        self.system = AttentionSystem(telemetry=self.telemetry)

    def test_each_step_records_a_snapshot(self) -> None:
        self.system.step([], [region(NEAR_POINT)])
        self.system.step([], [region(FAR_POINT)])
        self.assertEqual(len(self.telemetry.voxel_grids), 2)

    def test_a_snapshot_is_unaffected_by_later_steps(self) -> None:
        self.system.step([], [region(NEAR_POINT)])
        self.system.step([], [region(FAR_POINT)])
        self.assertEqual(len(self.telemetry.voxel_grids[0]), 1)
        self.assertEqual(len(self.telemetry.voxel_grids[1]), 2)

    def test_reset_discards_the_snapshots(self) -> None:
        self.system.step([], [region(NEAR_POINT)])
        self.system.reset()
        self.assertEqual(len(self.telemetry.voxel_grids), 0)

    def test_snapshots_encode_into_arrays(self) -> None:
        self.system.step([], [region(NEAR_POINT, weight=2)])
        self.system.step([], [region(NEAR_POINT, FAR_POINT, weight=2)])
        encoded = json.loads(json.dumps(self.system.state_dict(), cls=BufferEncoder))
        snapshot = encoded["voxel_grids"][1]
        self.assertEqual(snapshot["voxels"], [list(NEAR_VOXEL), list(FAR_VOXEL)])
        # Both voxels take this step's freshly proposed weight outright.
        self.assertEqual(snapshot["weight"], [2, 2])

    def test_an_empty_grid_snapshot_is_exported_empty(self) -> None:
        self.system.step([], [region(NEAR_POINT, weight=0.15)])
        # The first empty step decays 0.15 to within the rate of zero,
        # clamping it and expiring the voxel.
        self.system.step([], [])
        self.system.step([], [])
        snapshot = self.system.state_dict()["voxel_grids"][2]
        self.assertEqual(len(snapshot), 0)

    def test_state_dict_is_json_encodable(self) -> None:
        self.system.step([], [region(NEAR_POINT)])
        self.system.step([], [])
        encoded = json.loads(json.dumps(self.system.state_dict(), cls=BufferEncoder))
        self.assertEqual(encoded["voxel_grids"][0]["voxels"], [list(NEAR_VOXEL)])

    def test_a_default_telemetry_is_created_when_none_is_supplied(self) -> None:
        system = AttentionSystem()
        system.step([], [region(NEAR_POINT)])
        self.assertEqual(len(system.state_dict()["voxel_grids"]), 1)

    def test_state_dict_carries_the_grid_geometry(self) -> None:
        state = self.system.state_dict()
        self.assertEqual(state["voxel_size"], DEFAULT_VOXEL_SIZE)


class NoopAttentionSystemTelemetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.telemetry = NoopAttentionSystemTelemetry()
        self.system = AttentionSystem(telemetry=self.telemetry)

    def test_steps_record_nothing(self) -> None:
        self.system.step([], [region(NEAR_POINT)])
        self.assertEqual(self.system.state_dict()["voxel_grids"], [])


class AttentionSystemGoalFilteringTelemetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.telemetry = AttentionSystemTelemetry()
        self.system = AttentionSystem(telemetry=self.telemetry)

    def test_each_step_records_pre_and_post_goals(self) -> None:
        inside = goal_at(NEAR_POINT)
        outside = goal_at([9.0, 9, 9])
        self.system.step([inside, outside], [region(NEAR_POINT)])
        state = self.system.state_dict()
        self.assertEqual(state["pre_filter_goals"], [[inside, outside]])
        self.assertEqual(state["post_filter_goals"], [[inside]])

    def test_pass_through_steps_record_identical_pre_and_post(self) -> None:
        goal = goal_at(NEAR_POINT)
        self.system.step([goal], [])
        state = self.system.state_dict()
        self.assertEqual(state["pre_filter_goals"], state["post_filter_goals"])

    def test_reset_discards_the_goal_records(self) -> None:
        self.system.step([goal_at(NEAR_POINT)], [region(NEAR_POINT)])
        self.system.reset()
        state = self.system.state_dict()
        self.assertEqual(state["pre_filter_goals"], [])
        self.assertEqual(state["post_filter_goals"], [])
