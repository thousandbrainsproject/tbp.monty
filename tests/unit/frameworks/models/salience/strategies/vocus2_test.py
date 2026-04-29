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
from typing import cast
from unittest.mock import Mock

import numpy as np
import numpy.typing as npt
import pytest
import quaternion as qt
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tbp.monty.frameworks.models.salience.strategies.vocus2 import (
    Pyramid,
    gaussian_pyramid,
    pyramid_octave_shapes,
)
from tbp.monty.frameworks.sensors import Resolution2D
from tests.unit.statistics import total_variation


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


class PyramidOctaveShapesTest(unittest.TestCase):
    @given(
        image_shape=st.tuples(
            st.integers(min_value=1, max_value=1024),
            st.integers(min_value=1, max_value=1024),
        ),
    )
    def test_generates_all_octaves_when_no_level_or_size_constraints(
        self,
        image_shape: Resolution2D,
    ):
        computed_shapes = pyramid_octave_shapes(image_shape)
        expected_shapes = []
        while min(image_shape) >= 1:
            expected_shapes.append(image_shape)
            image_shape = cast(
                "Resolution2D", (image_shape[0] // 2, image_shape[1] // 2)
            )
        self.assertEqual(expected_shapes, computed_shapes)

    @given(
        image_shape=st.tuples(
            st.integers(min_value=1, max_value=1024),
            st.integers(min_value=1, max_value=1024),
        ),
        max_octaves=st.integers(min_value=1, max_value=int(2 * np.log2(1024))),
        min_size=st.integers(min_value=1, max_value=1024 * 2),
    )
    def test_max_octaves_limits_number_of_octaves(
        self,
        image_shape: Resolution2D,
        max_octaves: int,
        min_size: int,
    ):
        computed_shapes = pyramid_octave_shapes(
            image_shape, max_octaves=max_octaves, min_size=min_size
        )
        self.assertLessEqual(len(computed_shapes), max_octaves)

    @given(
        image_shape=st.tuples(
            st.integers(min_value=1, max_value=1024),
            st.integers(min_value=1, max_value=1024),
        ),
        max_octaves=st.integers(min_value=1, max_value=int(2 * np.log2(1024))),
        min_size=st.integers(min_value=1, max_value=1024 * 2),
    )
    def test_min_size_limits_number_of_octaves(
        self,
        image_shape: Resolution2D,
        max_octaves: int,
        min_size: int,
    ):
        computed_shapes = pyramid_octave_shapes(
            image_shape, max_octaves=max_octaves, min_size=min_size
        )
        smaller_dims = np.array([min(shape) for shape in computed_shapes], dtype=int)
        # assert all of smaller_dims are greater than or equal to min_size
        self.assertTrue(all(smaller_dims >= min_size))


class GaussianPyramidTest(unittest.TestCase):
    @given(
        image=arrays(dtype=np.float32, shape=(1024, 1024)),
        n_scales=st.integers(min_value=1, max_value=10),
        max_octaves=st.integers(min_value=1, max_value=int(2 * np.log2(1024))),
        min_size=st.integers(min_value=1, max_value=1024 * 2),
    )
    def test_gaussian_pyramid_has_correct_shape(
        self,
        image: np.ndarray,
        n_scales: int,
        max_octaves: int,
        min_size: int,
    ) -> None:
        sigma = 3.0

        expected_octave_shapes = pyramid_octave_shapes(
            cast("Resolution2D", image.shape),
            max_octaves=max_octaves,
            min_size=min_size,
        )
        pyr = gaussian_pyramid(
            image,
            sigma=sigma,
            n_scales=n_scales,
            max_octaves=max_octaves,
            min_size=min_size,
        )
        # Test pyramid is correct shape.
        self.assertEqual(pyr.n_octaves, len(expected_octave_shapes))
        self.assertEqual(pyr.n_scales, n_scales)

        # Test each plane in pyramid is correct shape.
        for octave in range(pyr.n_octaves):
            for scale in range(pyr.n_scales):
                self.assertEqual(
                    pyr.data[octave, scale].shape, expected_octave_shapes[octave]
                )

    @given(
        image=arrays(
            dtype=np.float32,
            shape=(1024, 1024),
            elements=st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                width=32,
            ),
        ),
        sigma=st.floats(min_value=0.5, max_value=3.0),
        n_scales=st.integers(min_value=1, max_value=3),
        max_octaves=st.integers(min_value=1, max_value=int(2 * np.log2(1024))),
        min_size=st.integers(min_value=1, max_value=1024 * 2),
    )
    def test_subsequent_planes_have_decreasing_total_variation(
        self,
        image: np.ndarray,
        sigma: float,
        n_scales: int,
        max_octaves: int,
        min_size: int,
    ) -> None:
        pyr = gaussian_pyramid(
            image,
            sigma=sigma,
            n_scales=n_scales,
            max_octaves=max_octaves,
            min_size=min_size,
        )
        variations = np.array([total_variation(plane) for plane in pyr.flat])
        diffs = np.ediff1d(variations)
        self.assertTrue(all(diffs <= 0))
