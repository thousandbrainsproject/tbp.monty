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

from tbp.monty.attention.attention_system import AttentionSystem
from tbp.monty.attention.telemetry import AttentionSystemTelemetry
from tbp.monty.frameworks.models.buffer import BufferEncoder

from .attention_system_test import region

NEAR_POINT = [0.0, 0, 0]
FAR_POINT = [0.5, 0, 0]


class AttentionSystemTelemetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.telemetry = AttentionSystemTelemetry()
        self.system = AttentionSystem(
            voxel_size=0.01, voxel_lifetime=2, telemetry=self.telemetry
        )

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

    def test_state_dict_flattens_each_snapshot_into_arrays(self) -> None:
        self.system.step([], [region(NEAR_POINT)])
        self.system.step([], [region(NEAR_POINT, FAR_POINT)])
        snapshot = self.system.state_dict()["voxel_grids"][1]
        np.testing.assert_array_equal(
            snapshot["voxels"], [[0, 0, 0], [50, 0, 0]]
        )
        np.testing.assert_array_equal(snapshot["age"], [2, 2])
        np.testing.assert_array_equal(snapshot["count"], [2, 1])

    def test_an_empty_grid_snapshot_is_exported_empty(self) -> None:
        self.system.step([], [region(NEAR_POINT)])
        # Two empty steps age the voxel past its lifetime of 2.
        self.system.step([], [])
        self.system.step([], [])
        snapshot = self.system.state_dict()["voxel_grids"][2]
        self.assertEqual(len(snapshot["voxels"]), 0)
        self.assertEqual(len(snapshot["age"]), 0)

    def test_state_dict_is_json_encodable(self) -> None:
        self.system.step([], [region(NEAR_POINT)])
        self.system.step([], [])
        encoded = json.loads(
            json.dumps(self.system.state_dict(), cls=BufferEncoder)
        )
        self.assertEqual(encoded["voxel_grids"][0]["voxels"], [[0, 0, 0]])

    def test_a_default_telemetry_is_created_when_none_is_supplied(self) -> None:
        system = AttentionSystem(voxel_size=0.01)
        system.step([], [region(NEAR_POINT)])
        self.assertEqual(len(system.state_dict()["voxel_grids"]), 1)
