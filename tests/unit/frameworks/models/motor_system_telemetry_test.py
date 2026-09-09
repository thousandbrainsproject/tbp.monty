# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest

import numpy as np
import numpy.testing as nptest

from tbp.monty.cmp import Goal
from tbp.monty.frameworks.models.motor_system_telemetry import (
    MotorSystemTelemetry,
    NoopMotorSystemTelemetry,
)


def goal(x: float, confidence: float = 1.0) -> Goal:
    return Goal(
        location=np.array([x, 0.0, 0.0]),
        morphological_features=None,
        non_morphological_features=None,
        confidence=confidence,
        pass_message=True,
        sender_id="view_finder",
        sender_type="SM",
        process_features_in_lm=False,
        goal_tolerances=None,
    )


class MotorSystemTelemetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.telemetry = MotorSystemTelemetry()

    def test_starts_empty(self) -> None:
        self.assertEqual(self.telemetry.state_dict(), {"goals_in": []})

    def test_records_each_steps_goals_as_columns(self) -> None:
        self.telemetry.goals_in([goal(1.0, 0.2), goal(2.0, 0.8)])
        self.telemetry.goals_in([])

        steps = self.telemetry.state_dict()["goals_in"]

        self.assertEqual(len(steps), 2)
        nptest.assert_array_equal(steps[0]["locations"], [[1.0, 0, 0], [2.0, 0, 0]])
        nptest.assert_array_equal(steps[0]["confidences"], [0.2, 0.8])
        self.assertEqual(list(steps[0]["sender_ids"]), ["view_finder"] * 2)
        self.assertEqual(steps[1]["locations"].shape, (0, 3))

    def test_state_dict_is_a_snapshot(self) -> None:
        self.telemetry.goals_in([goal(1.0)])
        snapshot = self.telemetry.state_dict()["goals_in"]

        self.telemetry.goals_in([goal(2.0)])

        self.assertEqual(len(snapshot), 1)

    def test_reset_discards_the_goals(self) -> None:
        self.telemetry.goals_in([goal(1.0)])

        self.telemetry.reset()

        self.assertEqual(self.telemetry.state_dict(), {"goals_in": []})


class NoopMotorSystemTelemetryTest(unittest.TestCase):
    def test_records_nothing_and_keeps_the_schema(self) -> None:
        telemetry = NoopMotorSystemTelemetry()

        telemetry.goals_in([goal(1.0)])
        telemetry.reset()

        self.assertEqual(telemetry.state_dict(), {"goals_in": []})
