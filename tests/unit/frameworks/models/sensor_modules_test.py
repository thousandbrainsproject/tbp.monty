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
from hypothesis import given
from hypothesis import strategies as st

from tbp.monty.cmp import Message
from tbp.monty.frameworks.models.sensor_modules import (
    FeatureChangeFilter,
    PassthroughPerceptFilter,
)


def create_percept(
    location: npt.NDArray[np.float64],
    on_object: bool,
    process_features_in_lm: bool,
) -> Message:
    """Create a percept for testing percept filters.

    Args:
        location: 3D location array.
        on_object: Whether the percept is on the object.
        process_features_in_lm: Whether the observation processor marked the percept as
            carrying valid features (on object and valid).

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
        pass_message=True,
        process_features_in_lm=process_features_in_lm,
        sender_id="SM_0",
        sender_type="SM",
    )


class FeatureChangeFilterTest(unittest.TestCase):
    @given(valid=st.booleans())
    def test_first_step_process_features_in_lm_iff_valid(self, valid: bool):
        feature_filter = FeatureChangeFilter(delta_thresholds={"distance": 0.5})
        result = feature_filter(
            create_percept(
                location=np.array([0.0, 0.0, 0.0]),
                on_object=valid,
                process_features_in_lm=valid,
            )
        )
        self.assertEqual(result.process_features_in_lm, valid)

    @given(valid=st.booleans(), feature_changed=st.booleans())
    def test_later_step_process_features_in_lm_iff_valid_and_changed(
        self, valid: bool, feature_changed: bool
    ):
        feature_filter = FeatureChangeFilter(delta_thresholds={"distance": 0.5})
        feature_filter(
            create_percept(
                location=np.array([0.0, 0.0, 0.0]),
                on_object=True,
                process_features_in_lm=True,
            )
        )
        location = np.array([10.0, 0.0, 0.0]) if feature_changed else np.zeros(3)
        result = feature_filter(
            create_percept(
                location=location, on_object=True, process_features_in_lm=valid
            )
        )
        self.assertEqual(result.process_features_in_lm, valid and feature_changed)


class PassthroughPerceptFilterTest(unittest.TestCase):
    @given(valid=st.booleans())
    def test_process_features_in_lm_iff_valid(self, valid: bool):
        percept_filter = PassthroughPerceptFilter()
        result = percept_filter(
            create_percept(
                location=np.array([0.0, 0.0, 0.0]),
                on_object=valid,
                process_features_in_lm=valid,
            )
        )
        self.assertEqual(result.process_features_in_lm, valid)


if __name__ == "__main__":
    unittest.main()
