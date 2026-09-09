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
import quaternion as qt

from tbp.monty.cmp import AttentionRegion, Goal
from tbp.monty.frameworks.models.buffer import BufferEncoder
from tbp.monty.frameworks.models.salience.telemetry import (
    NoopSalienceSMTelemetry,
    SalienceSMTelemetry,
)


class SalienceSMTelemetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.telemetry = SalienceSMTelemetry()
        self.mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)

    def test_state_dict_includes_the_snapshot_telemetry(self) -> None:
        state = self.telemetry.state_dict()
        self.assertIn("raw_observations", state)
        self.assertIn("sm_properties", state)

    def test_records_one_entry_per_step(self) -> None:
        self.telemetry.segmentation_map(self.mask)
        self.telemetry.segmentation_map(None)
        state = self.telemetry.state_dict()
        self.assertEqual(len(state["segmentation_maps"]), 2)

    def test_state_dict_holds_what_was_recorded(self) -> None:
        self.telemetry.segmentation_map(self.mask)
        state = self.telemetry.state_dict()
        np.testing.assert_array_equal(state["segmentation_maps"][0], self.mask)

    def test_a_step_without_segmentation_records_none(self) -> None:
        self.telemetry.segmentation_map(None)
        state = self.telemetry.state_dict()
        self.assertIsNone(state["segmentation_maps"][0])

    def test_reset_discards_the_recordings(self) -> None:
        self.telemetry.segmentation_map(self.mask)
        self.telemetry.reset()
        state = self.telemetry.state_dict()
        self.assertEqual(state["segmentation_maps"], [])

    def test_state_dict_is_json_encodable(self) -> None:
        self.telemetry.segmentation_map(self.mask)
        self.telemetry.segmentation_map(None)
        encoded = json.loads(json.dumps(self.telemetry.state_dict(), cls=BufferEncoder))
        self.assertEqual(encoded["segmentation_maps"][0], [[1, 0], [0, 1]])
        self.assertIsNone(encoded["segmentation_maps"][1])


class SalienceSMTelemetrySalienceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.telemetry = SalienceSMTelemetry()
        self.salience_map = np.array([[0.1, 0.9], [0.5, 0.0]])

    def test_each_call_records_a_map(self) -> None:
        self.telemetry.salience_map(self.salience_map)
        state = self.telemetry.state_dict()
        np.testing.assert_array_equal(state["salience_maps"][0], self.salience_map)

    def test_reset_discards_the_maps(self) -> None:
        self.telemetry.salience_map(self.salience_map)
        self.telemetry.reset()
        self.assertEqual(self.telemetry.state_dict()["salience_maps"], [])


def goal(x: float, confidence: float) -> Goal:
    return Goal(
        location=np.array([x, 0.0, 0.0]),
        morphological_features=None,
        non_morphological_features=None,
        confidence=confidence,
        pass_message=False,
        sender_id="view_finder",
        sender_type="SM",
        process_features_in_lm=False,
        goal_tolerances=None,
    )


class SalienceSMTelemetryGoalsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.telemetry = SalienceSMTelemetry()

    def test_each_step_records_its_goals_as_columns(self) -> None:
        self.telemetry.goals([goal(1.0, 0.2), goal(2.0, 0.8)])
        self.telemetry.goals([])

        steps = self.telemetry.state_dict()["goals"]

        self.assertEqual(len(steps), 2)
        np.testing.assert_array_equal(steps[0]["locations"], [[1.0, 0, 0], [2.0, 0, 0]])
        np.testing.assert_array_equal(steps[0]["confidences"], [0.2, 0.8])
        self.assertEqual(list(steps[0]["sender_ids"]), ["view_finder"] * 2)
        self.assertEqual(steps[1]["locations"].shape, (0, 3))

    def test_columns_are_json_encodable(self) -> None:
        self.telemetry.goals([goal(1.0, 0.5)])

        encoded = json.loads(json.dumps(self.telemetry.state_dict(), cls=BufferEncoder))

        self.assertEqual(encoded["goals"][0]["confidences"], [0.5])

    def test_reset_discards_the_goals(self) -> None:
        self.telemetry.goals([goal(1.0, 0.5)])
        self.telemetry.reset()
        self.assertEqual(self.telemetry.state_dict()["goals"], [])


class SalienceSMTelemetryRegionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.telemetry = SalienceSMTelemetry()
        self.region = AttentionRegion.uniform(
            [[1.0, 0, 0]], 1.0, sender_id="view_finder"
        )

    def test_each_step_records_its_region_or_none(self) -> None:
        self.telemetry.attention_region(self.region)
        self.telemetry.attention_region(None)

        regions = self.telemetry.state_dict()["attention_regions"]

        self.assertEqual(regions, [self.region, None])

    def test_regions_are_json_encodable(self) -> None:
        self.telemetry.attention_region(self.region)

        encoded = json.loads(json.dumps(self.telemetry.state_dict(), cls=BufferEncoder))

        self.assertEqual(encoded["attention_regions"][0]["sender_id"], "view_finder")

    def test_reset_discards_the_regions(self) -> None:
        self.telemetry.attention_region(self.region)
        self.telemetry.reset()
        self.assertEqual(self.telemetry.state_dict()["attention_regions"], [])


class NoopSalienceSMTelemetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.telemetry = NoopSalienceSMTelemetry()

    def test_records_nothing(self) -> None:
        self.telemetry.raw_observation(
            {"rgba": np.zeros((2, 2, 4))}, qt.quaternion(1, 0, 0, 0), np.zeros(3)
        )
        self.telemetry.salience_map(np.zeros((2, 2)))
        self.telemetry.segmentation_map(np.zeros((2, 2), dtype=np.uint8))
        self.telemetry.goals([goal(1.0, 0.5)])
        self.telemetry.attention_region(AttentionRegion.empty())

        self.assertEqual(
            self.telemetry.state_dict(),
            {
                "raw_observations": [],
                "sm_properties": [],
                "salience_maps": [],
                "segmentation_maps": [],
                "goals": [],
                "attention_regions": [],
            },
        )

    def test_exports_the_same_keys_as_the_recording_telemetry(self) -> None:
        self.assertEqual(
            set(self.telemetry.state_dict()), set(SalienceSMTelemetry().state_dict())
        )
