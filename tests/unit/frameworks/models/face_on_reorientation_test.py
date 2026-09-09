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
import quaternion as qt

from tbp.monty.cmp import Message
from tbp.monty.frameworks.models.face_on_reorientation import (
    FaceOnReorientation,
    view_angle,
)
from tbp.monty.frameworks.models.motor_system_state import SensorState

SENSOR_ID = "patch_0"
# A +Z surface point the sensor looks at.
SURFACE = np.array([0.1, 1.5, 0.05])


def percept(normal=(0.0, 0.0, 1.0), location=SURFACE, valid: bool = True) -> Message:
    """A camera SM percept: the surface normal is the first pose vector.

    Returns:
        The percept.
    """
    return Message(
        location=np.asarray(location, dtype=float),
        morphological_features={
            "pose_vectors": np.array([normal, [1.0, 0, 0], [0, 1.0, 0]], dtype=float),
            "pose_fully_defined": True,
            "on_object": 1.0,
        },
        non_morphological_features={},
        confidence=1.0,
        pass_message=True,
        sender_id=SENSOR_ID,
        sender_type="SM",
        process_features_in_lm=valid,
    )


def sensor_at(tilt_deg: float, distance: float = 0.08) -> SensorState:
    """A sensor ``distance`` from SURFACE looking at it, tilted about Y.

    At zero tilt it sits on the +Z normal looking down -Z; tilting rotates
    it about the surface point so the view angle equals the tilt.

    Returns:
        The sensor state.
    """
    rotation = qt.from_rotation_vector([0.0, np.radians(tilt_deg), 0.0])
    offset = qt.rotate_vectors(rotation, [0.0, 0.0, distance])
    return SensorState(position=SURFACE + offset, rotation=rotation)


def reorientation(**kwargs) -> FaceOnReorientation:
    return FaceOnReorientation(**{"max_view_angle": 45.0, "window": 1, **kwargs})


def run(gsg: FaceOnReorientation, tilts, **kwargs) -> list:
    """Step the generator through views at the tilts; the last step's goals.

    Returns:
        The goals the last step proposed.
    """
    goals = []
    for tilt in tilts:
        goals = gsg(percept(), sensor_at(tilt), **kwargs)
    return goals


class ViewAngleTest(unittest.TestCase):
    def test_is_zero_when_looking_along_the_negated_normal(self) -> None:
        self.assertAlmostEqual(view_angle(percept(), sensor_at(0.0)), 0.0)

    def test_matches_the_tilt_of_the_sensor(self) -> None:
        for tilt in (10.0, 45.0, 80.0):
            self.assertAlmostEqual(view_angle(percept(), sensor_at(tilt)), tilt)

    def test_is_none_without_a_valid_normal(self) -> None:
        self.assertIsNone(view_angle(percept(valid=False), sensor_at(0.0)))
        self.assertIsNone(view_angle(percept(normal=(0, 0, 0)), sensor_at(0.0)))


class TriggerTest(unittest.TestCase):
    def test_a_single_steep_view_does_not_trigger(self) -> None:
        gsg = reorientation(window=5)
        self.assertEqual(run(gsg, [10.0] * 4 + [80.0]), [])

    def test_a_persistently_steep_view_triggers_a_face_on_goal(self) -> None:
        gsg = reorientation(window=3)
        goals = run(gsg, [70.0] * 3)
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0].sender_type, "SM")
        self.assertEqual(goals[0].sender_id, SENSOR_ID)
        self.assertFalse(goals[0].pass_message)

    def test_does_not_trigger_when_this_step_is_face_on(self) -> None:
        # A steep history but a face-on current view: the jump would be
        # computed from a good view, so wait.
        gsg = reorientation(window=3)
        self.assertEqual(run(gsg, [80.0, 80.0, 5.0]), [])

    def test_does_not_trigger_off_the_object(self) -> None:
        gsg = reorientation()
        self.assertEqual(gsg(percept(valid=False), sensor_at(80.0)), [])

    def test_does_not_trigger_on_a_motor_only_step(self) -> None:
        gsg = reorientation()
        self.assertEqual(run(gsg, [80.0], motor_only_step=True), [])
        self.assertIsNone(gsg.view_angle)

    def test_nothing_is_proposed_before_the_window_is_full(self) -> None:
        gsg = reorientation(window=3)
        proposed = [len(gsg(percept(), sensor_at(70.0))) for _ in range(3)]
        self.assertEqual(proposed, [0, 0, 1])

    def test_the_goal_is_one_off(self) -> None:
        gsg = reorientation()
        self.assertEqual(len(run(gsg, [70.0])), 1)
        self.assertEqual(run(gsg, [70.0]), [])

    def test_waits_min_steps_between_goals(self) -> None:
        gsg = reorientation(min_steps_between_goals=3)
        proposed = [len(gsg(percept(), sensor_at(70.0))) for _ in range(6)]
        # Proposed on step 0, then not until more than 3 steps have passed.
        self.assertEqual(proposed, [1, 0, 0, 0, 1, 0])

    def test_reset_forgets_the_episode(self) -> None:
        gsg = reorientation(window=3, min_steps_between_goals=100)
        run(gsg, [70.0] * 3)
        gsg.reset()
        self.assertEqual(gsg.state_dict()["view_angle"], [])
        # One steep view after a reset fills only a third of the window.
        self.assertEqual(run(gsg, [70.0]), [])


class GoalGeometryTest(unittest.TestCase):
    def test_places_the_agent_on_the_normal_at_the_sensor_distance_looking_back(
        self,
    ) -> None:
        (goal,) = reorientation()(percept(), sensor_at(70.0, distance=0.08))
        nptest.assert_allclose(goal.location, SURFACE + [0, 0, 0.08], atol=1e-12)
        nptest.assert_allclose(
            goal.morphological_features["pose_vectors"][0], [0, 0, -1]
        )

    def test_a_configured_standoff_overrides_the_distance(self) -> None:
        (goal,) = reorientation(standoff=0.2)(percept(), sensor_at(70.0, distance=0.08))
        nptest.assert_allclose(goal.location, SURFACE + [0, 0, 0.2], atol=1e-12)

    def test_uses_a_unit_normal(self) -> None:
        (goal,) = reorientation(standoff=0.1)(
            percept(normal=(0, 0, 2.0)), sensor_at(70.0)
        )
        nptest.assert_allclose(goal.location, SURFACE + [0, 0, 0.1], atol=1e-12)


class TelemetryTest(unittest.TestCase):
    def test_records_one_angle_per_step(self) -> None:
        gsg = reorientation(window=2)
        gsg(percept(), sensor_at(30.0))
        gsg(percept(valid=False), sensor_at(30.0))
        gsg(percept(), sensor_at(60.0))
        state = gsg.state_dict()
        nptest.assert_allclose(state["view_angle"], [30.0, np.nan, 60.0])
        nptest.assert_allclose(state["smoothed_view_angle"], [30.0, 30.0, 45.0])

    def test_records_the_goal_and_whether_it_was_achieved(self) -> None:
        gsg = reorientation()
        gsg(percept(), sensor_at(70.0))
        (record,) = gsg.state_dict()["goals"]
        self.assertEqual(record["step"], 0)
        self.assertIsNone(record["achieved"])
        nptest.assert_allclose(record["surface_location"], SURFACE)
        nptest.assert_allclose(record["look_direction"], [0, 0, -1])
        self.assertAlmostEqual(record["view_angle"], 70.0)
        gsg(percept(), sensor_at(2.0))
        self.assertTrue(gsg.state_dict()["goals"][0]["achieved"])

    def test_a_still_steep_view_after_the_goal_counts_as_not_achieved(self) -> None:
        gsg = reorientation()
        run(gsg, [70.0, 60.0])
        self.assertFalse(gsg.state_dict()["goals"][0]["achieved"])
