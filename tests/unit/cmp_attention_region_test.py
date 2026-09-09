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
import numpy.testing as nptest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tbp.monty.cmp import MAX_ATTENTION_WEIGHT, MIN_ATTENTION_WEIGHT, AttentionRegion
from tbp.monty.frameworks.models.buffer import BufferEncoder

MAX_POINTS = 20
MAX_REGIONS = 5

coordinates = st.floats(min_value=-10.0, max_value=10.0)
weight_values = st.floats(
    min_value=MIN_ATTENTION_WEIGHT, max_value=MAX_ATTENTION_WEIGHT
)
sender_ids = st.sampled_from(["", "SM_0", "SM_3", "learning_module_2"])


@st.composite
def regions(draw: st.DrawFn) -> AttentionRegion:
    num_points = draw(st.integers(min_value=0, max_value=MAX_POINTS))
    return AttentionRegion(
        draw(arrays(dtype=np.float64, shape=(num_points, 3), elements=coordinates)),
        draw(arrays(dtype=np.float64, shape=(num_points,), elements=weight_values)),
        draw(sender_ids),
        draw(st.booleans()),
    )


class AttentionRegionTest(unittest.TestCase):
    @given(region=regions())
    def test_len_counts_the_locations(self, region: AttentionRegion) -> None:
        self.assertEqual(len(region), len(region.locations))
        self.assertEqual(len(region), len(region.weights))

    @given(
        num_locations=st.integers(min_value=0, max_value=MAX_POINTS),
        num_weights=st.integers(min_value=0, max_value=MAX_POINTS),
    )
    def test_raises_value_error_given_mismatched_lengths(
        self,
        num_locations: int,
        num_weights: int,
    ) -> None:
        if num_locations == num_weights:
            num_weights += 1

        with self.assertRaises(ValueError):
            AttentionRegion(np.zeros((num_locations, 3)), np.zeros(num_weights))

    def test_coerces_a_single_flat_location(self) -> None:
        region = AttentionRegion([1.0, 2.0, 3.0], [0.5])

        self.assertEqual(region.locations.shape, (1, 3))
        self.assertEqual(region.locations.dtype, np.float64)

    def test_empty_has_no_locations(self) -> None:
        empty = AttentionRegion.empty()

        self.assertEqual(len(empty), 0)
        self.assertEqual(empty.locations.shape, (0, 3))

    def test_sender_id_is_empty_unless_given(self) -> None:
        self.assertEqual(AttentionRegion(np.zeros((1, 3)), np.zeros(1)).sender_id, "")
        self.assertEqual(AttentionRegion.empty().sender_id, "")

    def test_inhibit_all_is_off_unless_given(self) -> None:
        self.assertFalse(AttentionRegion(np.zeros((1, 3)), np.zeros(1)).inhibit_all)
        self.assertFalse(AttentionRegion.empty().inhibit_all)
        self.assertFalse(AttentionRegion.uniform(np.zeros((2, 3)), 1.0).inhibit_all)

    def test_empty_can_carry_the_inhibit_all_signal(self) -> None:
        signal = AttentionRegion.empty(inhibit_all=True)

        self.assertTrue(signal.inhibit_all)
        self.assertEqual(len(signal), 0)

    @given(sender_id=sender_ids)
    def test_empty_and_uniform_carry_the_sender_id(self, sender_id: str) -> None:
        self.assertEqual(AttentionRegion.empty(sender_id).sender_id, sender_id)
        uniform = AttentionRegion.uniform(np.zeros((2, 3)), 1.0, sender_id=sender_id)
        self.assertEqual(uniform.sender_id, sender_id)

    @given(
        locations=arrays(
            dtype=np.float64,
            shape=st.tuples(st.integers(min_value=0, max_value=MAX_POINTS), st.just(3)),
            elements=coordinates,
        ),
        weight=weight_values,
    )
    def test_uniform_gives_every_location_the_weight(
        self,
        locations: np.ndarray,
        weight: float,
    ) -> None:
        region = AttentionRegion.uniform(locations, weight)

        nptest.assert_array_equal(region.locations, locations)
        nptest.assert_array_equal(region.weights, np.full(len(locations), weight))

    @given(parts=st.lists(regions(), min_size=0, max_size=MAX_REGIONS))
    def test_concat_keeps_every_location_and_weight_in_order(
        self,
        parts: list[AttentionRegion],
    ) -> None:
        joined = AttentionRegion.concat(parts)

        expected_locations = np.concatenate(
            [part.locations for part in parts] or [np.empty((0, 3))]
        )
        expected_weights = np.concatenate(
            [part.weights for part in parts] or [np.empty(0)]
        )
        nptest.assert_array_equal(joined.locations, expected_locations)
        nptest.assert_array_equal(joined.weights, expected_weights)

    @given(parts=st.lists(regions(), min_size=0, max_size=MAX_REGIONS))
    def test_concat_signals_inhibit_all_if_any_part_does(
        self,
        parts: list[AttentionRegion],
    ) -> None:
        joined = AttentionRegion.concat(parts)

        self.assertEqual(joined.inhibit_all, any(part.inhibit_all for part in parts))

    @given(
        parts=st.lists(regions(), min_size=0, max_size=MAX_REGIONS),
        sender_id=sender_ids,
    )
    def test_concat_takes_the_given_sender_id_not_its_inputs(
        self,
        parts: list[AttentionRegion],
        sender_id: str,
    ) -> None:
        self.assertEqual(AttentionRegion.concat(parts, sender_id).sender_id, sender_id)
        self.assertEqual(AttentionRegion.concat(parts).sender_id, "")

    @given(region=regions())
    def test_is_json_encodable_as_locations_and_weights(
        self,
        region: AttentionRegion,
    ) -> None:
        encoded = json.loads(json.dumps(region, cls=BufferEncoder))

        self.assertEqual(
            set(encoded), {"locations", "weights", "sender_id", "inhibit_all"}
        )
        self.assertEqual(encoded["sender_id"], region.sender_id)
        self.assertEqual(encoded["inhibit_all"], region.inhibit_all)
        # An empty region encodes its locations as [], which has no width.
        decoded_locations = np.asarray(encoded["locations"]).reshape(-1, 3)
        nptest.assert_array_equal(decoded_locations, region.locations)
        nptest.assert_array_equal(encoded["weights"], region.weights)
