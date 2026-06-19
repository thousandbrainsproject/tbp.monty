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
import numpy.typing as npt

from tbp.monty.cmp import Message
from tbp.monty.frameworks.models.sensor_modules import (
    FeatureChangeFilter,
)


def create_percept(
    location: npt.NDArray[np.float64],
    on_object: bool,
    use_state: bool,
) -> Message:
    """Create a percept for testing percept filters.

    Args:
        location: 3D location array.
        on_object: Whether the percept is on the object.
        use_state: Whether the observation processor marked the percept usable
            (on object and valid).

    Returns:
        A percept Message object
    """
    return Message(
        location=location,
        morphological_features={
            "pose_vectors": np.eye(3),
            "pose_fully_defined": True,
            "on_object": on_object,
        },
        non_morphological_features={},
        confidence=1.0,
        use_state=use_state,
        sender_id="SM_0",
        sender_type="SM",
    )


class FeatureChangeFilterTest(unittest.TestCase):
    def setUp(self):
        self.filter = FeatureChangeFilter(delta_thresholds={"distance": 0.5})

    def test_first_on_object_step_is_feature_step(self):
        percept = create_percept(
            location=np.array([0.0, 0.0, 0.0]), on_object=True, use_state=True
        )
        result = self.filter(percept)
        self.assertTrue(result.use_state)
        self.assertTrue(result.contains_features)

    def test_first_off_object_step_is_dropped(self):
        percept = create_percept(
            location=np.array([0.0, 0.0, 0.0]), on_object=False, use_state=False
        )
        result = self.filter(percept)
        self.assertFalse(result.use_state)

    def test_no_feature_change_on_object_is_location_only(self):
        first = create_percept(
            location=np.array([0.0, 0.0, 0.0]), on_object=True, use_state=True
        )
        self.filter(first)
        same = create_percept(
            location=np.array([0.0, 0.0, 0.0]), on_object=True, use_state=True
        )
        result = self.filter(same)
        self.assertTrue(result.use_state)
        self.assertFalse(result.contains_features)

    def test_significant_feature_change_on_object_is_feature_step(self):
        first = create_percept(
            location=np.array([0.0, 0.0, 0.0]), on_object=True, use_state=True
        )
        self.filter(first)
        moved = create_percept(
            location=np.array([10.0, 0.0, 0.0]), on_object=True, use_state=True
        )
        result = self.filter(moved)
        self.assertTrue(result.use_state)
        self.assertTrue(result.contains_features)

    def test_off_object_step_is_dropped(self):
        first = create_percept(
            location=np.array([0.0, 0.0, 0.0]), on_object=True, use_state=True
        )
        self.filter(first)
        off = create_percept(
            location=np.array([0.0, 0.0, 0.0]), on_object=False, use_state=False
        )
        result = self.filter(off)
        self.assertFalse(result.use_state)
        self.assertFalse(result.contains_features)

    def test_on_object_invalid_is_location_only(self):
        # On object but the observation processor flagged invalid (use_state=False).
        # It should be delivered as location-only so the agent location still updates.

        percept = create_percept(
            location=np.array([0.0, 0.0, 0.0]), on_object=True, use_state=False
        )
        result = self.filter(percept)
        self.assertTrue(result.use_state)
        self.assertFalse(result.contains_features)


if __name__ == "__main__":
    unittest.main()
