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
from unittest.mock import Mock

import numpy as np
import numpy.typing as npt
import pytest
import quaternion as qt
from hypothesis import example, given
from hypothesis import strategies as st

from tbp.monty.frameworks.models.salience.sensor_module import (
    SalienceSM,
)
from tbp.monty.frameworks.models.salience.strategies.vocus2 import Pyramid, Vocus2


class PyramidTest(unittest.TestCase):

    @given(ndim=st.integers(min_value=3, max_value=10))
    @example(ndim=1)
    def test_cannot_create_pyramid_with_non_2d_contents(self, ndim: int):
        with self.assertRaises(AssertionError):
            Pyramid(np.zeros((5,)*ndim, dtype=object))

    def test_can_create_pyramid_with_2d_contents(self):
        Pyramid(np.zeros((5, 5), dtype=object))

    @given(
        dim1=st.integers(min_value=1, max_value=10),
        dim2=st.integers(min_value=1, max_value=10),
    )
    def test_apply_applies_function_to_each_element(self, dim1: int, dim2: int):
        data = np.array([[Mock() for _ in range(dim2)] for _ in range(dim1)], dtype=object)
        pyr = Pyramid(data)
        fn = Mock()
        returned = pyr.apply(fn)
        self.assertEqual(len(fn.call_args_list), data.size)
        for i, call in enumerate(fn.call_args_list):
            self.assertEqual(call.args[0], data.flatten()[i])
        self.assertIsInstance(returned, Pyramid)
        self.assertEqual(returned.shape, pyr.shape)
