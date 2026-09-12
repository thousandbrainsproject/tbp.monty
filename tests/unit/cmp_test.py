# Copyright 2025-2026 Thousand Brains Project
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
from typing import Literal

import numpy as np
import numpy.typing as npt
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tbp.monty.cmp import AttentionRegion, Goal, Message, encode_goal, location_mean
from tbp.monty.frameworks.models.buffer import BufferEncoder
from tbp.monty.geometry import Rotation


def _message(
    sender_type: Literal["SM", "LM"],
    location: npt.NDArray[np.float64] | None,
) -> Message:
    return Message(
        location=location,
        morphological_features={},
        non_morphological_features={},
        confidence=1.0,
        pass_message=True,
        sender_id=f"{sender_type}_0",
        sender_type=sender_type,
        process_features_in_lm=False,
    )


class CMPMessageTest(unittest.TestCase):
    def test_is_from_sm(self):
        self.assertTrue(_message("SM", np.zeros(3)).is_from_sm())
        self.assertFalse(_message("LM", np.zeros(3)).is_from_sm())

    def test_location_mean_filters_out_none_locations(self):
        messages = [
            _message("SM", np.array([0.0, 0.0, 0.0])),
            _message("SM", None),
            _message("SM", np.array([2.0, 4.0, 6.0])),
        ]
        np.testing.assert_array_equal(
            location_mean(messages), np.mean([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]], axis=0)
        )

    def test_location_mean_none_when_no_location(self):
        self.assertIsNone(location_mean([]))
        self.assertIsNone(location_mean([_message("SM", None)]))


class EncodeGoalTest(unittest.TestCase):
    def setUp(self):
        self.goal_dict = {
            "location": np.array([0, 1.5, 0]),
            "morphological_features": {
                "pose_vectors": np.array(
                    [
                        -np.ones(3),
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                    ]
                ),
                "pose_fully_defined": None,
                "on_object": 1,
            },
            "non_morphological_features": None,
            "confidence": 1.0,
            "pass_message": True,
            "process_features_in_lm": True,
            "sender_id": "LM_0",
            "sender_type": "GSG",
            "goal_tolerances": None,
            "info": {
                "proposed_surface_loc": np.array([0, 1.5, 0]),
                "hypothesis_to_test": {
                    "graph_id": "mug",
                    "location": np.array([0, 1.5, 0]),
                    "rotation": Rotation.from_matrix(np.eye(3)),
                    "scale": 1.0,
                    "evidence": 1.0,
                },
                "achieved": False,
                "matching_step_when_output_goal_set": None,
            },
        }
        self.goal = Goal(**self.goal_dict)

    def test_encode(self):
        dct = encode_goal(self.goal)
        self.assertDictEqual(dct, self.goal_dict)

    def test_json_serialization(self):
        self.assertDictEqual(
            json.loads(json.dumps(self.goal, cls=BufferEncoder)),
            json.loads(json.dumps(self.goal_dict, cls=BufferEncoder)),
        )


@st.composite
def shape_not_n_by_3(draw: st.DrawFn) -> tuple[int, ...]:
    """Returns a shape that is not N by 3."""
    return draw(
        st.one_of(
            # 1D
            st.tuples(st.integers(min_value=0, max_value=10)),
            # 2D, not N by 3
            st.tuples(
                st.integers(min_value=0, max_value=10),
                st.one_of(
                    st.integers(min_value=0, max_value=2),
                    st.integers(min_value=4, max_value=10),
                ),
            ),
            # 3D+
            st.tuples(
                *(
                    st.integers(min_value=0, max_value=10)
                    for _ in range(draw(st.integers(min_value=3, max_value=10)))
                )
            ),
        )
    )


@st.composite
def shape_not_1d(draw: st.DrawFn) -> tuple[int, ...]:
    """Returns a shape that is not 1D."""
    return draw(
        st.tuples(
            *(
                st.integers(min_value=0, max_value=10)
                for _ in range(draw(st.integers(min_value=2, max_value=10)))
            )
        ),
    )


@st.composite
def locations_not_n_by_3(draw: st.DrawFn) -> npt.NDArray[np.float64]:
    """Returns an array of locations that is not N by 3."""
    return draw(
        arrays(
            dtype=np.float64,
            shape=shape_not_n_by_3(),
            elements=st.just(0.0),
            fill=st.just(0.0),
        )
    )


@st.composite
def weights_not_1d(draw: st.DrawFn) -> npt.NDArray[np.float64]:
    """Returns an array of weights that is not 1D."""
    return draw(
        arrays(
            dtype=np.float64,
            shape=shape_not_1d(),
            elements=st.just(0.0),
            fill=st.just(0.0),
        )
    )


@st.composite
def locations_n_by_3(draw: st.DrawFn) -> npt.NDArray[np.float64]:
    """Returns an array of locations that is N by 3."""
    return draw(
        arrays(
            dtype=np.float64,
            shape=st.tuples(st.integers(min_value=0, max_value=10), st.just(3)),
            elements=st.floats(allow_nan=False, allow_infinity=False),
            fill=st.just(0.0),
        )
    )


@st.composite
def mismatched_locations_and_weights(
    draw: st.DrawFn,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Returns a tuple of locations and weights with mismatched first dimensions."""
    locations = draw(locations_n_by_3())
    loc_length = len(locations)
    if loc_length == 0:
        weight_length = st.integers(min_value=1, max_value=10)
    else:
        weight_length = st.one_of(
            st.integers(min_value=0, max_value=loc_length - 1),
            st.integers(min_value=loc_length + 1, max_value=loc_length + 10),
        )

    weights = draw(
        arrays(
            dtype=np.float64,
            shape=st.tuples(weight_length),
            elements=st.just(0.0),
            fill=st.just(0.0),
        )
    )
    return locations, weights


@st.composite
def locations_and_a_weight(
    draw: st.DrawFn,
) -> tuple[npt.NDArray[np.float64], float]:
    """Returns a tuple of locations and weights with matching first dimensions."""
    locations = draw(locations_n_by_3())
    weight = draw(st.floats(allow_nan=False, allow_infinity=False))
    return locations, weight


@st.composite
def attention_region(draw: st.DrawFn) -> AttentionRegion:
    """Returns an AttentionRegion with valid locations and weights."""
    locations = draw(locations_n_by_3())
    weights = draw(
        arrays(
            dtype=np.float64,
            shape=st.tuples(st.just(locations.shape[0])),
            elements=st.floats(allow_nan=False, allow_infinity=False),
            fill=st.just(0.0),
        )
    )
    return AttentionRegion(locations=locations, weights=weights)


@st.composite
def attention_regions(draw: st.DrawFn) -> list[AttentionRegion]:
    """Returns a list of AttentionRegions with valid locations and weights."""
    return draw(st.lists(attention_region(), min_size=0, max_size=10))


class AttentionRegionTest(unittest.TestCase):
    @given(locations=locations_not_n_by_3())
    def test_initialized_with_locations_not_n_by_3_raises_value_error(
        self,
        locations: npt.NDArray[np.float64],
    ):
        with self.assertRaises(ValueError):
            AttentionRegion(locations=locations, weights=np.ones(locations.shape[0]))

    @given(weights=weights_not_1d())
    def test_initialized_with_weights_not_1d_raises_value_error(
        self,
        weights: npt.NDArray[np.float64],
    ):
        with self.assertRaises(ValueError):
            AttentionRegion(locations=np.ones((weights.shape[0], 3)), weights=weights)

    @given(mismatched=mismatched_locations_and_weights())
    def test_initialized_with_mismatched_number_of_locations_and_weights_raises_value_error(  # noqa: E501
        self,
        mismatched: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    ):
        locations, weights = mismatched
        with self.assertRaises(ValueError):
            AttentionRegion(locations=locations, weights=weights)

    @given(locations_and_a_weight=locations_and_a_weight())
    def test_uniform_gives_every_location_the_weight(
        self,
        locations_and_a_weight: tuple[npt.NDArray[np.float64], float],
    ):
        locations, weight = locations_and_a_weight
        attention_region = AttentionRegion.uniform(locations=locations, weight=weight)
        np.testing.assert_array_equal(
            attention_region.weights, np.full(locations.shape[0], weight)
        )

    @settings(deadline=None)
    @given(attention_regions=attention_regions())
    def test_concat_keeps_every_location_and_weight_in_order(
        self, attention_regions: list[AttentionRegion]
    ):
        attention_region = AttentionRegion.concat(attention_regions)
        if len(attention_regions) == 0:
            self.assertEqual(len(attention_region.locations), 0)
            self.assertEqual(len(attention_region.weights), 0)
            self.assertEqual(len(attention_region), 0)
        else:
            np.testing.assert_array_equal(
                attention_region.locations,
                np.concatenate([ar.locations for ar in attention_regions]),
            )
            np.testing.assert_array_equal(
                attention_region.weights,
                np.concatenate([ar.weights for ar in attention_regions]),
            )
            self.assertEqual(
                len(attention_region), sum(len(ar) for ar in attention_regions)
            )
