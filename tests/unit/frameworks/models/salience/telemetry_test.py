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

from tbp.monty.cmp import Goal
from tbp.monty.frameworks.models.buffer import BufferEncoder
from tbp.monty.frameworks.models.salience.telemetry import SalienceSMTelemetry


def goal_at(location) -> Goal:
    """Build a goal at the given location.

    Returns:
        A goal whose only meaningful property here is its location.

    """
    return Goal(
        location=np.asarray(location, dtype=float),
        morphological_features=None,
        non_morphological_features=None,
        confidence=0.5,
        use_state=False,
        sender_id="SM_0",
        sender_type="SM",
        goal_tolerances=None,
    )


class SalienceSMTelemetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.telemetry = SalienceSMTelemetry(save_segmentation=True)
        self.mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        self.region = [goal_at([0.0, 0, 1]), goal_at([1.0, 1, 1])]

    def test_record_is_a_noop_by_default(self) -> None:
        telemetry = SalienceSMTelemetry()
        telemetry.record(self.mask, self.region)
        state = telemetry.state_dict()
        self.assertEqual(state["segmentation_maps"], [])
        self.assertEqual(state["regions"], [])

    def test_state_dict_includes_the_snapshot_telemetry(self) -> None:
        state = self.telemetry.state_dict()
        self.assertIn("raw_observations", state)
        self.assertIn("sm_properties", state)

    def test_records_one_entry_per_step(self) -> None:
        self.telemetry.record(self.mask, self.region)
        self.telemetry.record(None, [])
        state = self.telemetry.state_dict()
        self.assertEqual(len(state["segmentation_maps"]), 2)
        self.assertEqual(len(state["regions"]), 2)

    def test_state_dict_holds_what_was_recorded(self) -> None:
        self.telemetry.record(self.mask, self.region)
        state = self.telemetry.state_dict()
        np.testing.assert_array_equal(state["segmentation_maps"][0], self.mask)
        self.assertEqual(state["regions"][0], self.region)

    def test_a_step_without_segmentation_records_none(self) -> None:
        self.telemetry.record(None, [])
        state = self.telemetry.state_dict()
        self.assertIsNone(state["segmentation_maps"][0])
        self.assertEqual(state["regions"][0], [])

    def test_the_recorded_region_is_detached_from_the_callers_list(self) -> None:
        region = list(self.region)
        self.telemetry.record(self.mask, region)
        region.clear()
        self.assertEqual(len(self.telemetry.state_dict()["regions"][0]), 2)

    def test_reset_discards_the_recordings(self) -> None:
        self.telemetry.record(self.mask, self.region)
        self.telemetry.reset()
        state = self.telemetry.state_dict()
        self.assertEqual(state["segmentation_maps"], [])
        self.assertEqual(state["regions"], [])

    def test_state_dict_is_json_encodable(self) -> None:
        self.telemetry.record(self.mask, self.region)
        self.telemetry.record(None, [])
        encoded = json.loads(
            json.dumps(self.telemetry.state_dict(), cls=BufferEncoder)
        )
        self.assertEqual(encoded["segmentation_maps"][0], [[1, 0], [0, 1]])
        self.assertIsNone(encoded["segmentation_maps"][1])
        self.assertEqual(len(encoded["regions"][0]), 2)
