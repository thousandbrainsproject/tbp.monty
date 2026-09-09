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

import numpy as np

from tbp.monty.attention.attention_system import (
    DEFAULT_VOXEL_SIZE,
    AttentionSystem,
)
from tbp.monty.attention.telemetry import (
    AttentionSystemTelemetry,
    NoopAttentionSystemTelemetry,
)
from tbp.monty.cmp import AttentionRegion
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
        self.assertEqual(len(self.system.state_dict()["grids"]), 2)

    def test_a_snapshot_is_unaffected_by_later_steps(self) -> None:
        self.system.step([], [region(NEAR_POINT)])
        first = self.system.state_dict()["grids"][0]
        weight_then = first["weight"].to_numpy().copy()

        # The next step decays the live grid in place and grows it.
        self.system.step([], [region(FAR_POINT)])

        grids = self.system.state_dict()["grids"]
        np.testing.assert_array_equal(grids[0]["weight"].to_numpy(), weight_then)
        self.assertEqual(len(grids[0]), 1)
        self.assertEqual(len(grids[1]), 2)

    def test_reset_discards_the_snapshots(self) -> None:
        self.system.step([], [region(NEAR_POINT)])
        self.system.reset()
        self.assertEqual(self.system.state_dict()["grids"], [])

    def test_the_proposed_grid_holds_only_this_steps_regions(self) -> None:
        self.system.step([], [region(NEAR_POINT)])
        self.system.step([], [region(FAR_POINT)])

        state = self.system.state_dict()
        self.assertEqual(len(state["proposed_grids"][1]), 1)
        self.assertEqual(len(state["grids"][1]), 2)

    def test_snapshots_encode_into_arrays(self) -> None:
        self.system.step([], [region(NEAR_POINT, weight=2)])
        self.system.step([], [region(NEAR_POINT, FAR_POINT, weight=2)])
        encoded = json.loads(json.dumps(self.system.state_dict(), cls=BufferEncoder))
        snapshot = encoded["grids"][1]
        self.assertEqual(snapshot["voxels"], [list(NEAR_VOXEL), list(FAR_VOXEL)])
        # Both voxels take this step's freshly proposed weight outright.
        self.assertEqual(snapshot["weight"], [2, 2])

    def test_a_proposed_grid_encodes_its_inhibit_all_signal(self) -> None:
        self.system.step([], [region(NEAR_POINT)])
        self.system.step([], [AttentionRegion.empty(inhibit_all=True)])
        encoded = json.loads(json.dumps(self.system.state_dict(), cls=BufferEncoder))
        self.assertEqual(
            [grid["inhibit_all"] for grid in encoded["proposed_grids"]], [False, True]
        )
        self.assertEqual(
            [grid["inhibit_all"] for grid in encoded["grids"]], [False, False]
        )

    def test_an_empty_grid_snapshot_is_exported_empty(self) -> None:
        self.system.step([], [region(NEAR_POINT, weight=0.15)])
        # The first empty step decays 0.15 to within the rate of zero,
        # clamping it and expiring the voxel.
        self.system.step([], [])
        self.system.step([], [])
        snapshot = self.system.state_dict()["grids"][2]
        self.assertEqual(len(snapshot), 0)

    def test_state_dict_is_json_encodable(self) -> None:
        self.system.step([], [region(NEAR_POINT)])
        self.system.step([], [])
        encoded = json.loads(json.dumps(self.system.state_dict(), cls=BufferEncoder))
        self.assertEqual(encoded["grids"][0]["voxels"], [list(NEAR_VOXEL)])

    def test_a_default_telemetry_is_created_when_none_is_supplied(self) -> None:
        system = AttentionSystem()
        system.step([], [region(NEAR_POINT)])
        self.assertEqual(len(system.state_dict()["grids"]), 1)

    def test_state_dict_carries_the_grid_geometry(self) -> None:
        state = self.system.state_dict()
        self.assertEqual(state["voxel_size"], DEFAULT_VOXEL_SIZE)


class NoopAttentionSystemTelemetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.telemetry = NoopAttentionSystemTelemetry()
        self.system = AttentionSystem(telemetry=self.telemetry)

    def test_steps_record_nothing(self) -> None:
        self.system.step([goal_at(NEAR_POINT)], [region(NEAR_POINT)])

        state = self.system.state_dict()
        self.assertEqual(
            {k: state[k] for k in ("grids", "proposed_grids")},
            {"grids": [], "proposed_grids": []},
        )
