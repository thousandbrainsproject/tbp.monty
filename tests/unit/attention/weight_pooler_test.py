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

from hypothesis import given
from hypothesis import strategies as st

from tbp.monty.attention.weight_pooler import negative_priority_max_pool
from tests.strategies.floats import with_negative_floats


class NegativePriorityMaxPoolTest(unittest.TestCase):
    @given(
        values=st.lists(
            st.floats(min_value=0.0, allow_nan=False, allow_infinity=False), min_size=1
        )
    )
    def test_call_returns_max_value_if_all_are_non_negative(
        self, values: list[float]
    ) -> None:
        result = negative_priority_max_pool(values)
        self.assertEqual(result, max(values))

    @given(values=with_negative_floats())
    def test_call_returns_most_negative_value_if_any_are_negative(
        self, values: list[float]
    ) -> None:
        result = negative_priority_max_pool(values)
        self.assertEqual(result, min(values))
