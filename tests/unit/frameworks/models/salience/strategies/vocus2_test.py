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
from dataclasses import dataclass
from enum import Enum
from typing import Callable, cast
from unittest.mock import Mock, patch, sentinel

import cv2
import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
import scipy.signal
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tbp.monty.frameworks.models.salience.strategies.vocus2 import (
    ColorChannelSalience,
    DepthSalience,
    Pyramid,
    SafeOperatingLimits,
    center_surround_pyramids,
    gaussian_pyramid,
    laplacian_pyramid,
    pyramid_collapse,
    pyramid_combine,
    pyramid_octave_shapes,
)
from tbp.monty.frameworks.sensors import Resolution2D
from tbp.monty.math import DEFAULT_TOLERANCE
from tests.unit.statistics import mean_local_variation, total_variation


class PyramidTest(unittest.TestCase):
    @given(ndim=st.integers(min_value=3, max_value=10))
    @example(ndim=1)
    def test_cannot_create_pyramid_with_non_2d_contents(self, ndim: int) -> None:
        with self.assertRaises(AssertionError):
            Pyramid(np.zeros((5,) * ndim, dtype=object))

    def test_can_create_pyramid_with_2d_contents(self) -> None:
        Pyramid(np.zeros((5, 5), dtype=object))

    @given(
        dim1=st.integers(min_value=1, max_value=10),
        dim2=st.integers(min_value=1, max_value=10),
    )
    def test_apply_applies_function_to_each_element(self, dim1: int, dim2: int) -> None:
        data = np.array(
            [[Mock() for _ in range(dim2)] for _ in range(dim1)], dtype=object
        )
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
    ) -> None:
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
    ) -> None:
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
    ) -> None:
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
    def test_has_correct_shape(
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


@st.composite
def center_surround_sigmas(
    draw: st.DrawFn,
    min_center_sigma: float,
    max_center_sigma: float,
) -> tuple[float, float]:
    center_sigma = draw(
        st.floats(min_value=min_center_sigma, max_value=max_center_sigma)
    )
    surround_sigma_factor = draw(
        st.floats(min_value=1.0, max_value=10.0, exclude_min=True)
    )
    return center_sigma, surround_sigma_factor * center_sigma


@st.composite
def solid_float32_image(
    draw: st.DrawFn, min_dim_size: int = 1, max_dim_size: int = 1024
) -> npt.NDArray[np.float32]:
    height = draw(st.integers(min_value=min_dim_size, max_value=max_dim_size))
    width = draw(st.integers(min_value=min_dim_size, max_value=max_dim_size))
    fill_value = draw(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, width=32)
    )
    return np.full((height, width), fill_value, dtype=np.float32)


@st.composite
def filled_float32_image(
    draw: st.DrawFn,
    fill_value: float = 1.0,
    min_dim_size: int = 1,
    max_dim_size: int = 1024,
) -> npt.NDArray[np.float32]:
    height = draw(st.integers(min_value=min_dim_size, max_value=max_dim_size))
    width = draw(st.integers(min_value=min_dim_size, max_value=max_dim_size))
    return np.full((height, width), fill_value, dtype=np.float32)


@st.composite
def float32_image(draw: st.DrawFn) -> npt.NDArray[np.float32]:
    height = draw(st.integers(min_value=1, max_value=1024))
    width = draw(st.integers(min_value=1, max_value=1024))
    return draw(
        arrays(
            dtype=np.float32,
            shape=(height, width),
            elements=st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                width=32,
            ),
        )
    )


@st.composite
def sufficiently_variable_float32_image(draw: st.DrawFn) -> npt.NDArray[np.float32]:
    return draw(
        float32_image().filter(
            lambda img: not np.allclose(img, img[0, 0], atol=DEFAULT_TOLERANCE)
        )
    )


@st.composite
def pyramid(draw: st.DrawFn, fill_value: float = 1.0) -> Pyramid:
    image_width = draw(st.integers(min_value=1, max_value=1024))
    image_height = draw(st.integers(min_value=1, max_value=1024))
    n_scales = draw(st.integers(min_value=1, max_value=10))
    max_octaves = draw(st.integers(min_value=1, max_value=int(2 * np.log2(1024))))
    min_size = draw(st.integers(min_value=1, max_value=min(image_width, image_height)))
    octave_shapes = pyramid_octave_shapes(
        (image_height, image_width),
        max_octaves=max_octaves,
        min_size=min_size,
    )
    input_data = np.zeros((len(octave_shapes), n_scales), dtype=object)
    for octave_num, octave_shape in enumerate(octave_shapes):
        for scale_num in range(n_scales):
            input_data[octave_num, scale_num] = np.full(
                octave_shape, fill_value, dtype=np.float32
            )
    return Pyramid(input_data)


@st.composite
def valid_input_pyramid_for_laplacian_pyramid(
    draw: st.DrawFn, fill_value: float = 1.0
) -> Pyramid:
    image_width = draw(st.integers(min_value=2, max_value=1024))
    image_height = draw(st.integers(min_value=2, max_value=1024))
    n_scales = draw(st.integers(min_value=1, max_value=10))
    max_octaves = draw(st.integers(min_value=2, max_value=int(2 * np.log2(1024))))
    octave_shapes = pyramid_octave_shapes(
        (image_height, image_width),
        max_octaves=max_octaves,
    )
    input_data = np.zeros((len(octave_shapes), n_scales), dtype=object)
    for octave_num, octave_shape in enumerate(octave_shapes):
        for scale_num in range(n_scales):
            input_data[octave_num, scale_num] = np.full(
                octave_shape, fill_value, dtype=np.float32
            )
    return Pyramid(input_data)


@st.composite
def differently_shaped_pyramids(
    draw: st.DrawFn, fill_value: float = 1.0
) -> list[Pyramid]:
    pyramid_1 = draw(valid_input_pyramid_for_laplacian_pyramid(fill_value=fill_value))
    pyramid_2 = draw(
        valid_input_pyramid_for_laplacian_pyramid(fill_value=fill_value).filter(
            lambda pyr: pyr.shape != pyramid_1.shape
        )
    )
    return [pyramid_1, pyramid_2]


class CenterSurroundPyramidsTest(unittest.TestCase):
    @given(
        center_sigma=st.floats(min_value=0.5, max_value=3.0),
        surround_sigma_factor=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_raises_value_error_if_center_sigma_is_greater_than_or_equal_to_surround_sigma(  # noqa: E501
        self,
        center_sigma: float,
        surround_sigma_factor: float,
    ) -> None:
        surround_sigma = center_sigma * surround_sigma_factor
        with self.assertRaises(ValueError):
            center_surround_pyramids(
                np.zeros((1024, 1024)),
                center_sigma=center_sigma,
                surround_sigma=surround_sigma,
                n_scales=2,
                max_octaves=5,
                min_size=16,
            )

    @given(
        image=arrays(dtype=np.float32, shape=(1024, 1024)),
        n_scales=st.integers(min_value=1, max_value=10),
        max_octaves=st.integers(min_value=1, max_value=int(2 * np.log2(1024))),
        min_size=st.integers(min_value=1, max_value=1024 * 2),
    )
    def test_center_and_surround_pyramids_have_same_shape(
        self,
        image: npt.NDArray[np.float32],
        n_scales: int,
        max_octaves: int,
        min_size: int,
    ) -> None:
        center, surround = center_surround_pyramids(
            image,
            center_sigma=3.0,
            surround_sigma=5.0,
            n_scales=n_scales,
            max_octaves=max_octaves,
            min_size=min_size,
        )
        self.assertEqual(center.shape, surround.shape)

    @given(
        image=float32_image(),
        sigmas=center_surround_sigmas(min_center_sigma=0.5, max_center_sigma=3.0),
        n_scales=st.integers(min_value=1, max_value=3),
        max_octaves=st.integers(min_value=1, max_value=int(2 * np.log2(1024))),
        min_size=st.integers(min_value=1, max_value=1024 * 2),
    )
    def test_surround_planes_have_higher_mean_local_variation_than_corresponding_center_planes(  # noqa: E501
        self,
        image: npt.NDArray[np.float32],
        sigmas: tuple[float, float],
        n_scales: int,
        max_octaves: int,
        min_size: int,
    ) -> None:
        center, surround = center_surround_pyramids(
            image,
            center_sigma=sigmas[0],
            surround_sigma=sigmas[1],
            n_scales=n_scales,
            max_octaves=max_octaves,
            min_size=min_size,
        )

        center_variations = np.array(
            [mean_local_variation(plane) for plane in center.flat]
        )
        surround_variations = np.array(
            [mean_local_variation(plane) for plane in surround.flat]
        )
        variations = center_variations - surround_variations
        tolerance = 1e-4  # opencv variation tolerance
        self.assertTrue(all(variations >= -tolerance))

    @given(
        image=solid_float32_image(),
        sigmas=center_surround_sigmas(min_center_sigma=0.5, max_center_sigma=3.0),
        n_scales=st.integers(min_value=1, max_value=3),
        max_octaves=st.integers(min_value=1, max_value=int(2 * np.log2(1024))),
        min_size=st.integers(min_value=1, max_value=1024 * 2),
    )
    def test_surround_planes_mean_local_variation_equals_corresponding_center_planes_for_solid_image(  # noqa: E501
        self,
        image: npt.NDArray[np.float32],
        sigmas: tuple[float, float],
        n_scales: int,
        max_octaves: int,
        min_size: int,
    ) -> None:
        center, surround = center_surround_pyramids(
            image,
            center_sigma=sigmas[0],
            surround_sigma=sigmas[1],
            n_scales=n_scales,
            max_octaves=max_octaves,
            min_size=min_size,
        )

        center_variations = np.array(
            [mean_local_variation(plane) for plane in center.flat]
        )
        surround_variations = np.array(
            [mean_local_variation(plane) for plane in surround.flat]
        )
        self.assertTrue(
            np.allclose(center_variations, surround_variations, atol=DEFAULT_TOLERANCE)
        )

    @given(
        image=sufficiently_variable_float32_image(),
        sigmas=center_surround_sigmas(min_center_sigma=0.5, max_center_sigma=3.0),
        n_scales=st.integers(min_value=1, max_value=3),
        max_octaves=st.integers(min_value=1, max_value=int(2 * np.log2(1024))),
        min_size=st.integers(min_value=1, max_value=1024 * 2),
    )
    def test_surround_planes_have_higher_mean_local_variation_than_corresponding_center_planes_for_sufficiently_variable_image(  # noqa: E501
        self,
        image: npt.NDArray[np.float32],
        sigmas: tuple[float, float],
        n_scales: int,
        max_octaves: int,
        min_size: int,
    ) -> None:
        center, surround = center_surround_pyramids(
            image,
            center_sigma=sigmas[0],
            surround_sigma=sigmas[1],
            n_scales=n_scales,
            max_octaves=max_octaves,
            min_size=min_size,
        )

        center_variations = np.array(
            [mean_local_variation(plane) for plane in center.flat]
        )
        surround_variations = np.array(
            [mean_local_variation(plane) for plane in surround.flat]
        )
        variations = center_variations - surround_variations
        self.assertTrue(all(variations >= 0))


class LaplacianPyramidTest(unittest.TestCase):
    FILL_VALUE = 1.0

    @given(
        input_pyramid=valid_input_pyramid_for_laplacian_pyramid(fill_value=FILL_VALUE),
    )
    def test_has_correct_shape(self, input_pyramid: Pyramid) -> None:
        pyramid = laplacian_pyramid(input_pyramid)
        self.assertEqual(input_pyramid.n_octaves - 1, pyramid.n_octaves)
        self.assertEqual(input_pyramid.n_scales, pyramid.n_scales)
        for octave in range(pyramid.n_octaves):
            for scale in range(pyramid.n_scales):
                self.assertEqual(
                    pyramid.data[octave, scale].shape,
                    input_pyramid.data[octave, scale].shape,
                )

    def test_raises_value_error_if_input_pyramid_has_less_than_two_octaves(
        self,
    ) -> None:
        data = np.zeros((1, 1), dtype=object)
        data[0, 0] = np.zeros((1, 1), dtype=np.float32)
        with self.assertRaises(ValueError):
            laplacian_pyramid(Pyramid(data))

    @given(
        input_pyramid=valid_input_pyramid_for_laplacian_pyramid(fill_value=FILL_VALUE),
    )
    def test_laplacian_planes_are_center_minus_resized_surround(
        self,
        input_pyramid: Pyramid,
    ) -> None:
        surround_fill = 0.7

        def mock_resize(
            image: np.ndarray,
            shape: tuple[int, int],
            interpolation: int,  # noqa: ARG001
        ) -> np.ndarray:
            return np.full(shape, surround_fill, dtype=image.dtype)

        with patch(
            "tbp.monty.frameworks.models.salience.strategies.vocus2.resize",
            side_effect=mock_resize,
        ) as mock_resize_patch:
            pyr = laplacian_pyramid(input_pyramid)
            for plane in pyr.flat:
                nptest.assert_allclose(
                    plane, self.FILL_VALUE - surround_fill, atol=DEFAULT_TOLERANCE
                )

            call_count = 0
            for scale in range(input_pyramid.n_scales):
                for octave in range(pyr.n_octaves):
                    expected_image = input_pyramid.data[octave + 1, scale]
                    expected_shape = input_pyramid.data[octave, scale].shape
                    call_args = mock_resize_patch.call_args_list[call_count]
                    nptest.assert_array_equal(call_args.args[0], expected_image)
                    self.assertEqual(call_args.args[1], expected_shape)
                    self.assertEqual(call_args.kwargs["interpolation"], cv2.INTER_CUBIC)
                    call_count += 1


class PyramidCombineTest(unittest.TestCase):
    def test_raises_value_error_if_no_pyramids_are_provided(self) -> None:
        with self.assertRaises(ValueError):
            pyramid_combine([], Mock())

    def test_returns_first_pyramid_if_only_one_pyramid_is_provided(self) -> None:
        pyramid = Pyramid(np.zeros((1, 1), dtype=object))
        result = pyramid_combine([pyramid], Mock())
        self.assertIs(result, pyramid)

    def test_does_not_apply_reduce_to_pyramids_if_only_one_pyramid_is_provided(
        self,
    ) -> None:
        pyramid = Pyramid(np.zeros((1, 1), dtype=object))
        reduce = Mock()
        result = pyramid_combine([pyramid], reduce)
        reduce.assert_not_called()
        self.assertIs(result, pyramid)

    @given(
        pyramids=differently_shaped_pyramids(),
    )
    def test_raises_value_error_if_pyramids_have_different_shapes(
        self,
        pyramids: list[Pyramid],
    ) -> None:
        with self.assertRaises(ValueError):
            pyramid_combine(pyramids, Mock())

    @given(
        pyramid=valid_input_pyramid_for_laplacian_pyramid(),
    )
    def test_returns_combined_pyramid_with_correct_shape_and_reduced_planes(
        self,
        pyramid: Pyramid,
    ) -> None:
        reduce = Mock()

        def mock_reduce(
            images: tuple[np.ndarray, ...],
        ) -> np.ndarray:
            return np.zeros_like(images[0])

        reduce.side_effect = mock_reduce

        pyramids = [pyramid, pyramid]
        result = pyramid_combine(pyramids, reduce)

        self.assertEqual(result.shape, pyramid.shape)
        self.assertEqual(result.n_octaves, pyramid.n_octaves)
        self.assertEqual(result.n_scales, pyramid.n_scales)

        self.assertEqual(reduce.call_count, pyramid.size)
        call_count = 0
        for octave in range(result.n_octaves):
            for scale in range(result.n_scales):
                call_args = reduce.call_args_list[call_count]
                self.assertIs(call_args.args[0][0], pyramid.data[octave, scale])
                self.assertIs(call_args.args[0][1], pyramid.data[octave, scale])
                nptest.assert_array_equal(
                    result.data[octave, scale],
                    mock_reduce(
                        (pyramid.data[octave, scale], pyramid.data[octave, scale])
                    ),
                )
                call_count += 1


class PyramidCollapseTest(unittest.TestCase):
    INPUT_FILL_VALUE = 0.0

    @settings(deadline=1000)
    @given(
        pyramid=pyramid(fill_value=INPUT_FILL_VALUE),
    )
    def test_resize_only_called_on_planes_with_shapes_different_from_first_plane_and_returns_what_reduce_returns(  # noqa: E501
        self,
        pyramid: Pyramid,
    ) -> None:
        resize_fill = 1.0

        def mock_resize(
            image: np.ndarray,
            shape: tuple[int, int],
            interpolation: int,  # noqa: ARG001
        ) -> np.ndarray:
            return np.full(shape, resize_fill, dtype=image.dtype)

        reduce_mock = Mock()
        reduce_mock.return_value = sentinel.reduce_return_value

        with patch(
            "tbp.monty.frameworks.models.salience.strategies.vocus2.resize",
            side_effect=mock_resize,
        ) as mock_resize_patch:
            result = pyramid_collapse(pyramid, reduce=reduce_mock)
            self.assertIs(result, sentinel.reduce_return_value)
            n_expected_calls_to_resize = pyramid.n_scales * (pyramid.n_octaves - 1)
            self.assertEqual(mock_resize_patch.call_count, n_expected_calls_to_resize)

            target_shape = pyramid.data[0, 0].shape

            call_count = 0
            for octave in range(1, pyramid.n_octaves):
                for scale in range(pyramid.n_scales):
                    call_args = mock_resize_patch.call_args_list[call_count]
                    self.assertIs(call_args.args[0], pyramid.data[octave, scale])
                    self.assertEqual(call_args.args[1], target_shape)
                    self.assertEqual(call_args.kwargs["interpolation"], cv2.INTER_CUBIC)
                    call_count += 1

            expected_reduce_input_array = np.zeros(pyramid.shape, dtype=object)
            for scale in range(pyramid.n_scales):
                expected_reduce_input_array[0, scale] = np.full(
                    target_shape,
                    self.INPUT_FILL_VALUE,
                    dtype=np.float32,
                )
            for octave in range(1, pyramid.n_octaves):
                for scale in range(pyramid.n_scales):
                    expected_reduce_input_array[octave, scale] = np.full(
                        target_shape,
                        resize_fill,
                        dtype=np.float32,
                    )
            expected_reduce_input = list(expected_reduce_input_array.flat)
            reduce_mock.assert_called_once()
            reduce_input = reduce_mock.call_args_list[0].args[0]
            self.assertEqual(len(reduce_input), len(expected_reduce_input))
            for i in range(len(reduce_input)):
                nptest.assert_array_equal(reduce_input[i], expected_reduce_input[i])


@dataclass
class CenterAndSurroundParams:
    center_sigma: float
    surround_sigma: float
    n_scales: int
    max_octaves: int
    min_size: int


@st.composite
def safe_center_and_surround_params(
    draw: st.DrawFn, image: npt.NDArray[np.float32]
) -> CenterAndSurroundParams:
    center_sigma = draw(
        st.floats(
            min_value=SafeOperatingLimits.min_center_sigma,
            max_value=SafeOperatingLimits.max_center_sigma,
        )
    )
    surround_sigma = draw(
        st.floats(
            min_value=center_sigma * SafeOperatingLimits.center_surround_sigma_ratio,
            max_value=SafeOperatingLimits.max_surround_sigma,
        )
    )
    n_scales = draw(st.integers(min_value=1, max_value=5))
    max_octaves = draw(
        st.integers(min_value=1, max_value=int(1 + np.log2(max(image.shape))))
    )
    min_size = draw(st.integers(min_value=1, max_value=min(image.shape)))

    return CenterAndSurroundParams(
        center_sigma=center_sigma,
        surround_sigma=surround_sigma,
        n_scales=n_scales,
        max_octaves=max_octaves,
        min_size=min_size,
    )


@st.composite
def unsafe_min_size_in_safe_center_and_surround_params(
    draw: st.DrawFn, image: npt.NDArray[np.float32]
) -> CenterAndSurroundParams:
    center_sigma = draw(
        st.floats(
            min_value=SafeOperatingLimits.min_center_sigma,
            max_value=SafeOperatingLimits.max_center_sigma,
        )
    )
    surround_sigma = draw(
        st.floats(
            min_value=center_sigma * SafeOperatingLimits.center_surround_sigma_ratio,
            max_value=SafeOperatingLimits.max_surround_sigma,
        )
    )
    n_scales = draw(st.integers(min_value=1, max_value=5))
    max_octaves = draw(
        st.integers(min_value=1, max_value=int(1 + np.log2(max(image.shape))))
    )
    min_size = draw(
        st.integers(min_value=min(image.shape) + 1, max_value=max(image.shape) * 3)
    )

    return CenterAndSurroundParams(
        center_sigma=center_sigma,
        surround_sigma=surround_sigma,
        n_scales=n_scales,
        max_octaves=max_octaves,
        min_size=min_size,
    )


@st.composite
def color_channel_salience_processor(
    draw: st.DrawFn,
    image: npt.NDArray[np.float32],
    center_and_surround_params_strategy: Callable[
        [npt.NDArray[np.float32]], st.SearchStrategy[CenterAndSurroundParams]
    ] = safe_center_and_surround_params,
) -> ColorChannelSalience:
    center_and_surround_params = draw(center_and_surround_params_strategy(image))
    return ColorChannelSalience(
        center_sigma=center_and_surround_params.center_sigma,
        surround_sigma=center_and_surround_params.surround_sigma,
        n_scales=center_and_surround_params.n_scales,
        max_octaves=center_and_surround_params.max_octaves,
        min_size=center_and_surround_params.min_size,
    )


@st.composite
def color_channel_salience_processor_without_operating_limits(
    draw: st.DrawFn,
    image: npt.NDArray[np.float32],
    center_and_surround_params_strategy: Callable[
        [npt.NDArray[np.float32]], st.SearchStrategy[CenterAndSurroundParams]
    ] = safe_center_and_surround_params,
) -> ColorChannelSalience:
    center_and_surround_params = draw(center_and_surround_params_strategy(image))
    return ColorChannelSalience.without_operating_limits(
        center_sigma=center_and_surround_params.center_sigma,
        surround_sigma=center_and_surround_params.surround_sigma,
        n_scales=center_and_surround_params.n_scales,
        max_octaves=center_and_surround_params.max_octaves,
        min_size=center_and_surround_params.min_size,
    )


@st.composite
def color_channel_salience_setup(
    draw: st.DrawFn,
    salience_processor_strategy: st.SearchStrategy[ColorChannelSalience],
    image_strategy: st.SearchStrategy[npt.NDArray[np.float32]],
    center_and_surround_params_strategy: Callable[
        [npt.NDArray[np.float32]], st.SearchStrategy[CenterAndSurroundParams]
    ] = safe_center_and_surround_params,
) -> tuple[npt.NDArray[np.float32], ColorChannelSalience]:
    image = draw(image_strategy)
    processor = draw(
        salience_processor_strategy(image, center_and_surround_params_strategy)
    )
    return image, processor


@st.composite
def solid_left_half_float32_image_color_channel_salience_setup(
    draw: st.DrawFn, fill_value: float = 1.0
) -> tuple[npt.NDArray[np.float32], ColorChannelSalience]:
    image = draw(
        filled_float32_image(
            fill_value=fill_value, min_dim_size=SafeOperatingLimits.min_image_dim_size
        )
    )
    image[:, image.shape[1] // 2 :] = 0.0
    processor = draw(color_channel_salience_processor(image))
    return image, processor


class Direction(Enum):
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


@st.composite
def unsafe_center_and_surround_sigmas(draw: st.DrawFn) -> tuple[float, float]:
    center_sigma = draw(
        st.one_of(
            st.floats(
                min_value=SafeOperatingLimits.min_center_sigma - 1,
                max_value=SafeOperatingLimits.min_center_sigma,
                exclude_max=True,
            ),
            st.floats(
                min_value=SafeOperatingLimits.max_center_sigma,
                max_value=SafeOperatingLimits.max_center_sigma + 1,
                exclude_min=True,
            ),
        )
    )
    surround_sigma = draw(
        st.one_of(
            st.floats(
                min_value=SafeOperatingLimits.center_surround_sigma_ratio
                * SafeOperatingLimits.min_center_sigma
                - 1,
                max_value=SafeOperatingLimits.center_surround_sigma_ratio
                * SafeOperatingLimits.min_center_sigma,
                exclude_max=True,
            ),
            st.floats(
                min_value=SafeOperatingLimits.max_surround_sigma,
                max_value=SafeOperatingLimits.max_surround_sigma + 1,
                exclude_min=True,
            ),
        )
    )
    return center_sigma, surround_sigma


class ColorChannelSalienceTest(unittest.TestCase):
    MINIMUM_SALIENCE_THRESHOLD = 1e-3

    @given(
        image_and_processor=color_channel_salience_setup(
            color_channel_salience_processor,
            filled_float32_image(
                max_dim_size=SafeOperatingLimits.min_image_dim_size - 1
            ),
            unsafe_min_size_in_safe_center_and_surround_params,
        ),
    )
    def test_process_raises_value_error_if_image_has_smaller_dimension_than_min_size(
        self,
        image_and_processor: tuple[npt.NDArray[np.float32], ColorChannelSalience],
    ) -> None:
        image, processor = image_and_processor
        with self.assertRaises(ValueError):
            processor.process(Mock(), image)

    @given(
        image_and_processor=color_channel_salience_setup(
            color_channel_salience_processor, solid_float32_image()
        ),
    )
    def test_solid_image_not_salient(
        self, image_and_processor: tuple[npt.NDArray[np.float32], ColorChannelSalience]
    ) -> None:
        image, processor = image_and_processor
        feature_map, _ = processor.process(Mock(), image)
        self.assertTrue(np.all(feature_map < self.MINIMUM_SALIENCE_THRESHOLD))

    @given(
        vertical_edge_case=solid_left_half_float32_image_color_channel_salience_setup(),
        edge_orientation=st.sampled_from(Direction),
    )
    def test_edge_not_salient_but_flanks_are_salient(
        self,
        vertical_edge_case: tuple[npt.NDArray[np.float32], ColorChannelSalience],
        edge_orientation: Direction,
    ) -> None:
        # Over our tested range of center_sigma and surround_sigma values, an edge
        # should look qualitatively like
        #      /\    /\
        #     /  \  /  \
        # ___/    \/    \___
        # where central minimum is located at the edge.
        vertical_edge_image, processor = vertical_edge_case

        if edge_orientation == Direction.VERTICAL:
            image = vertical_edge_image
        elif edge_orientation == Direction.HORIZONTAL:
            image = vertical_edge_image.T
        else:
            raise ValueError(f"Invalid edge orientation: {edge_orientation}")

        feature_map, _ = processor.process(Mock(), image)

        if edge_orientation == Direction.VERTICAL:
            band = feature_map[feature_map.shape[0] // 2]
        elif edge_orientation == Direction.HORIZONTAL:
            band = feature_map[:, feature_map.shape[1] // 2]
        else:
            raise ValueError(f"Invalid edge orientation: {edge_orientation}")

        local_maxima, _ = scipy.signal.find_peaks(band)
        local_minima, _ = scipy.signal.find_peaks(-band)
        index_of_edge = len(band) // 2
        peaks_below_edge = local_maxima[local_maxima < index_of_edge]
        peaks_above_edge = local_maxima[local_maxima > index_of_edge]
        self.assertTrue(
            index_of_edge in local_minima or index_of_edge - 1 in local_minima
        )
        peaks_below_edge = local_maxima[local_maxima < index_of_edge]
        self.assertTrue(len(peaks_below_edge) > 0)
        peaks_above_edge = local_maxima[local_maxima > index_of_edge]
        self.assertTrue(len(peaks_above_edge) > 0)

    @given(
        center_and_surround_sigmas=unsafe_center_and_surround_sigmas(),
    )
    def test_constructing_with_unsafe_sigmas_raises_value_error(
        self,
        center_and_surround_sigmas: tuple[float, float],
    ) -> None:
        center_sigma, surround_sigma = center_and_surround_sigmas
        with self.assertRaises(ValueError):
            ColorChannelSalience(
                center_sigma=center_sigma,
                surround_sigma=surround_sigma,
            )

    @given(
        image_and_processor=color_channel_salience_setup(
            color_channel_salience_processor,
            filled_float32_image(
                max_dim_size=SafeOperatingLimits.min_image_dim_size - 1
            ),
        ),
    )
    def test_process_raises_value_error_if_image_has_smaller_dimension_than_min_safe_dim_size_and_suppress_runtimes_errors_is_false(  # noqa: E501
        self,
        image_and_processor: tuple[npt.NDArray[np.float32], ColorChannelSalience],
    ) -> None:
        image, processor = image_and_processor
        ctx = Mock(suppress_runtime_errors=False)
        with self.assertRaises(ValueError):
            processor.process(ctx, image)

    @given(
        image_and_processor=color_channel_salience_setup(
            color_channel_salience_processor,
            filled_float32_image(
                max_dim_size=SafeOperatingLimits.min_image_dim_size - 1
            ),
        ),
    )
    def test_process_does_not_raise_value_error_if_image_has_smaller_dimension_than_min_safe_dim_size_and_suppress_runtimes_errors_is_true(  # noqa: E501
        self,
        image_and_processor: tuple[npt.NDArray[np.float32], ColorChannelSalience],
    ) -> None:
        image, processor = image_and_processor
        ctx = Mock(suppress_runtime_errors=True)
        processor.process(ctx, image)


class ColorChannelSalienceWithoutOperatingLimitsTest(unittest.TestCase):
    @given(
        center_and_surround_sigmas=unsafe_center_and_surround_sigmas(),
    )
    def test_constructing_with_unsafe_sigmas_does_not_raise(
        self,
        center_and_surround_sigmas: tuple[float, float],
    ) -> None:
        center_sigma, surround_sigma = center_and_surround_sigmas
        ColorChannelSalience.without_operating_limits(
            center_sigma=center_sigma,
            surround_sigma=surround_sigma,
        )

    @given(
        image_and_processor=color_channel_salience_setup(
            color_channel_salience_processor_without_operating_limits,
            filled_float32_image(
                max_dim_size=SafeOperatingLimits.min_image_dim_size - 1
            ),
        ),
        suppress_runtime_errors=st.booleans(),
    )
    def test_process_does_not_raise_value_error_if_image_has_smaller_dimension_than_min_safe_dim_size(  # noqa: E501
        self,
        image_and_processor: tuple[npt.NDArray[np.float32], ColorChannelSalience],
        suppress_runtime_errors: bool,
    ) -> None:
        image, processor = image_and_processor
        ctx = Mock(suppress_runtime_errors=suppress_runtime_errors)
        processor.process(ctx, image)


@st.composite
def depth_salience_processor(
    draw: st.DrawFn,
    image: npt.NDArray[np.float32],
    center_and_surround_params_strategy: Callable[
        [npt.NDArray[np.float32]], st.SearchStrategy[CenterAndSurroundParams]
    ] = safe_center_and_surround_params,
) -> DepthSalience:
    center_and_surround_params = draw(center_and_surround_params_strategy(image))
    return DepthSalience(
        center_sigma=center_and_surround_params.center_sigma,
        surround_sigma=center_and_surround_params.surround_sigma,
        n_scales=center_and_surround_params.n_scales,
        max_octaves=center_and_surround_params.max_octaves,
        min_size=center_and_surround_params.min_size,
    )


@st.composite
def depth_salience_setup(
    draw: st.DrawFn,
    salience_processor_strategy: st.SearchStrategy[DepthSalience],
    image_strategy: st.SearchStrategy[npt.NDArray[np.float32]],
    center_and_surround_params_strategy: Callable[
        [npt.NDArray[np.float32]], st.SearchStrategy[CenterAndSurroundParams]
    ] = safe_center_and_surround_params,
) -> tuple[npt.NDArray[np.float32], DepthSalience]:
    image = draw(image_strategy)
    processor = draw(
        salience_processor_strategy(image, center_and_surround_params_strategy)
    )
    return image, processor


@st.composite
def near_left_half_float32_image_depth_salience_setup(
    draw: st.DrawFn, near_value: float = 0.5
) -> tuple[npt.NDArray[np.float32], DepthSalience]:
    image = draw(
        filled_float32_image(
            fill_value=near_value, min_dim_size=SafeOperatingLimits.min_image_dim_size
        )
    )
    image[:, image.shape[1] // 2 :] = near_value * 2.0
    processor = draw(depth_salience_processor(image))
    return image, processor


@st.composite
def depth_salience_processor_without_operating_limits(
    draw: st.DrawFn,
    image: npt.NDArray[np.float32],
    center_and_surround_params_strategy: Callable[
        [npt.NDArray[np.float32]], st.SearchStrategy[CenterAndSurroundParams]
    ] = safe_center_and_surround_params,
) -> DepthSalience:
    center_and_surround_params = draw(center_and_surround_params_strategy(image))
    return DepthSalience.without_operating_limits(
        center_sigma=center_and_surround_params.center_sigma,
        surround_sigma=center_and_surround_params.surround_sigma,
        n_scales=center_and_surround_params.n_scales,
        max_octaves=center_and_surround_params.max_octaves,
        min_size=center_and_surround_params.min_size,
    )


class DepthSalienceTest(unittest.TestCase):
    MINIMUM_SALIENCE_THRESHOLD = 1e-3

    @given(
        image_and_processor=depth_salience_setup(
            depth_salience_processor,
            filled_float32_image(
                max_dim_size=SafeOperatingLimits.min_image_dim_size - 1
            ),
            unsafe_min_size_in_safe_center_and_surround_params,
        ),
    )
    def test_process_raises_value_error_if_image_has_smaller_dimension_than_min_size(
        self,
        image_and_processor: tuple[npt.NDArray[np.float32], DepthSalience],
    ) -> None:
        image, processor = image_and_processor
        with self.assertRaises(ValueError):
            processor.process(Mock(), image)

    @given(
        image_and_processor=depth_salience_setup(
            depth_salience_processor,
            solid_float32_image(min_dim_size=SafeOperatingLimits.min_image_dim_size),
        ),
    )
    def test_uniform_depth_image_not_salient(
        self, image_and_processor: tuple[npt.NDArray[np.float32], DepthSalience]
    ) -> None:
        image, processor = image_and_processor
        feature_map = processor.process(Mock(), image)
        self.assertTrue(np.all(feature_map < self.MINIMUM_SALIENCE_THRESHOLD))

    @given(
        vertical_edge_case=near_left_half_float32_image_depth_salience_setup(),
        edge_orientation=st.sampled_from(Direction),
    )
    def test_near_more_salient_than_far(
        self,
        vertical_edge_case: tuple[npt.NDArray[np.float32], DepthSalience],
        edge_orientation: Direction,
    ) -> None:
        vertical_edge_image, processor = vertical_edge_case

        if edge_orientation == Direction.VERTICAL:
            image = vertical_edge_image
        elif edge_orientation == Direction.HORIZONTAL:
            image = vertical_edge_image.T
        else:
            raise ValueError(f"Invalid edge orientation: {edge_orientation}")

        feature_map = processor.process(Mock(), image)

        if edge_orientation == Direction.VERTICAL:
            band = feature_map[feature_map.shape[0] // 2]
        elif edge_orientation == Direction.HORIZONTAL:
            band = feature_map[:, feature_map.shape[1] // 2]
        else:
            raise ValueError(f"Invalid edge orientation: {edge_orientation}")

        left_of_edge = band[: len(band) // 2]
        right_of_edge = band[len(band) // 2 :]
        self.assertTrue(left_of_edge.sum() > right_of_edge.sum())

    @given(
        center_and_surround_sigmas=unsafe_center_and_surround_sigmas(),
    )
    def test_constructing_with_unsafe_sigmas_raises_value_error(
        self,
        center_and_surround_sigmas: tuple[float, float],
    ) -> None:
        center_sigma, surround_sigma = center_and_surround_sigmas
        with self.assertRaises(ValueError):
            DepthSalience(
                center_sigma=center_sigma,
                surround_sigma=surround_sigma,
            )

    @given(
        image_and_processor=depth_salience_setup(
            depth_salience_processor,
            filled_float32_image(
                max_dim_size=SafeOperatingLimits.min_image_dim_size - 1
            ),
        ),
    )
    def test_process_raises_value_error_if_image_has_smaller_dimension_than_min_safe_dim_size_and_suppress_runtimes_errors_is_false(  # noqa: E501
        self,
        image_and_processor: tuple[npt.NDArray[np.float32], DepthSalience],
    ) -> None:
        image, processor = image_and_processor
        ctx = Mock(suppress_runtime_errors=False)
        with self.assertRaises(ValueError):
            processor.process(ctx, image)

    @given(
        image_and_processor=depth_salience_setup(
            depth_salience_processor,
            filled_float32_image(
                max_dim_size=SafeOperatingLimits.min_image_dim_size - 1
            ),
        ),
    )
    def test_process_does_not_raise_value_error_if_image_has_smaller_dimension_than_min_safe_dim_size_and_suppress_runtimes_errors_is_true(  # noqa: E501
        self,
        image_and_processor: tuple[npt.NDArray[np.float32], DepthSalience],
    ) -> None:
        image, processor = image_and_processor
        ctx = Mock(suppress_runtime_errors=True)
        processor.process(ctx, image)


class DepthSalienceWithoutOperatingLimitsTest(unittest.TestCase):
    @given(
        center_and_surround_sigmas=unsafe_center_and_surround_sigmas(),
    )
    def test_constructing_with_unsafe_sigmas_does_not_raise(
        self,
        center_and_surround_sigmas: tuple[float, float],
    ) -> None:
        center_sigma, surround_sigma = center_and_surround_sigmas
        DepthSalience.without_operating_limits(
            center_sigma=center_sigma,
            surround_sigma=surround_sigma,
        )

    @given(
        image_and_processor=depth_salience_setup(
            depth_salience_processor_without_operating_limits,
            filled_float32_image(
                max_dim_size=SafeOperatingLimits.min_image_dim_size - 1
            ),
        ),
        suppress_runtime_errors=st.booleans(),
    )
    def test_process_does_not_raise_value_error_if_image_has_smaller_dimension_than_min_safe_dim_size(  # noqa: E501
        self,
        image_and_processor: tuple[npt.NDArray[np.float32], DepthSalience],
        suppress_runtime_errors: bool,
    ) -> None:
        image, processor = image_and_processor
        ctx = Mock(suppress_runtime_errors=suppress_runtime_errors)
        processor.process(ctx, image)
