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
from unittest.mock import Mock, sentinel

import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
from hypothesis import given
from hypothesis import strategies as st

from tbp.monty.cmp import AttentionRegion, Message
from tbp.monty.frameworks.models.sensor_modules import (
    CameraSM,
    FeatureChangeFilter,
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
        location = (
            np.array([10.0, 0.0, 0.0]) if feature_changed else np.array([0.0, 0.0, 0.0])
        )
        result = feature_filter(
            create_percept(
                location=location, on_object=True, process_features_in_lm=valid
            )
        )
        self.assertEqual(result.process_features_in_lm, valid and feature_changed)


def camera_sm(*proposers: Mock, gsg: Mock | None = None) -> Mock:
    # A CameraSM stand-in carrying only the goal generator and the region
    # proposers; the real methods under test are bound to it explicitly.
    sm = Mock(spec=CameraSM)
    sm.sensor_module_id = "patch_0"
    sm._gsg = Mock() if gsg is None else gsg
    sm._region_proposers = proposers
    sm._goals = []
    sm._percept = None
    sm.state = sentinel.sensor_state
    # What `step` and `state_dict` touch besides the above.
    sm.save_raw_obs = False
    sm.is_exploring = True
    sm.processed_obs = []
    sm._snapshot_telemetry = Mock()
    sm._snapshot_telemetry.state_dict.return_value = {}
    sm._observation_processor = Mock()
    sm._observation_processor.process.return_value = sentinel.percept
    sm._message_noise = Mock(side_effect=lambda percept, **_: percept)
    sm._percept_filter = Mock(side_effect=lambda percept: percept)
    return sm


def region_at(*locations: float) -> AttentionRegion:
    # One location per value, so concatenation order is visible.
    return AttentionRegion.uniform([[x, 0.0, 0.0] for x in locations], 1.0)


class CameraSMGoalGeneratorTest(unittest.TestCase):
    """CameraSM proposes the goals its goal generator returns for the step."""

    def setUp(self) -> None:
        self.gsg = Mock(return_value=[sentinel.goal])
        self.sm = camera_sm(gsg=self.gsg)

    def test_step_hands_the_percept_and_sensor_state_to_the_generator(self) -> None:
        CameraSM.step(self.sm, Mock(), sentinel.observation, motor_only_step=True)

        self.gsg.assert_called_once_with(
            sentinel.percept, sentinel.sensor_state, motor_only_step=True
        )

    def test_proposes_the_generators_goals(self) -> None:
        CameraSM.step(self.sm, Mock(), sentinel.observation)

        self.assertEqual(CameraSM.propose_goals(self.sm), [sentinel.goal])

    def test_state_dict_carries_the_generators_telemetry(self) -> None:
        self.gsg.state_dict.return_value = {"view_angle": [1.0]}

        self.assertEqual(CameraSM.state_dict(self.sm)["gsg"], {"view_angle": [1.0]})

    def test_reset_resets_the_generator_and_the_proposers(self) -> None:
        proposer = Mock()
        sm = camera_sm(proposer, gsg=self.gsg)

        CameraSM.reset(sm)

        self.gsg.reset.assert_called_once_with()
        proposer.reset.assert_called_once_with()
        self.assertEqual(sm._goals, [])

    def test_a_real_module_proposes_nothing_by_default(self) -> None:
        sm = CameraSM(
            sensor_module_id="patch_0", features=["pose_vectors", "on_object"]
        )
        self.assertEqual(sm.propose_goals(), [])
        self.assertIsNone(sm.propose_region())


class CameraSMProposeRegionTest(unittest.TestCase):
    """CameraSM concatenates its region proposers' output, like an LM does."""

    def test_calls_each_proposer_with_a_context_over_this_step(self) -> None:
        proposer = Mock(return_value=None)
        sm = camera_sm(proposer)
        sm._goals = [sentinel.goal]
        sm._percept = sentinel.percept

        CameraSM.propose_region(sm)

        (context,), _ = proposer.call_args
        self.assertEqual(context.sensor_module_id, "patch_0")
        self.assertEqual(context.goals, (sentinel.goal,))
        self.assertIs(context.percept, sentinel.percept)
        self.assertIs(context.sensor_state, sentinel.sensor_state)

    def test_concatenates_every_proposer_output_in_order(self) -> None:
        sm = camera_sm(
            Mock(return_value=region_at(1.0, 2.0)), Mock(return_value=region_at(3.0))
        )

        region = CameraSM.propose_region(sm)

        nptest.assert_array_equal(region.locations[:, 0], [1.0, 2.0, 3.0])
        self.assertEqual(region.sender_id, "patch_0")

    def test_passes_a_single_proposal_through_unchanged(self) -> None:
        proposal = region_at(1.0)
        sm = camera_sm(Mock(return_value=proposal))

        self.assertIs(CameraSM.propose_region(sm), proposal)

    def test_proposes_none_when_no_proposer_does(self) -> None:
        sm = camera_sm(Mock(return_value=None))

        self.assertIsNone(CameraSM.propose_region(sm))
