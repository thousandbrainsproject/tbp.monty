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
from typing import cast
from unittest.mock import Mock, patch, sentinel

import cv2
import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tbp.monty.frameworks.models.salience.strategies.vocus2.pyramids import (
    Pyramid,
    center_surround_pyramids,
    gaussian_pyramid,
    laplacian_pyramid,
    pyramid_collapse,
    pyramid_combine,
    pyramid_octave_shapes,
)
from tbp.monty.frameworks.sensors import Resolution2D
from tbp.monty.math import DEFAULT_TOLERANCE
from tests.unit.statistics import mean_local_variation

# Common upper limits used in these tests. Not the same thing
# as safe operating limits.
MAX_DIM_SIZE = 1024
MAX_OCTAVES = int(np.log2(MAX_DIM_SIZE)) + 1
MAX_SCALES = 5


# Parameters
# -----------------------------------------------------------------------------


@st.composite
def default_resolutions(
    draw: st.DrawFn,
    min_dim_size: int = 1,
    max_dim_size: int = MAX_DIM_SIZE,
) -> Resolution2D:
    height = draw(st.integers(min_value=min_dim_size, max_value=max_dim_size))
    width = draw(st.integers(min_value=min_dim_size, max_value=max_dim_size))
    return cast("Resolution2D", (height, width))


@st.composite
def resolutions_with_a_zero_length_dimension(
    draw: st.DrawFn,
) -> Resolution2D:
    height = 0
    width = draw(st.integers(min_value=1, max_value=MAX_DIM_SIZE))
    if draw(st.booleans()):
        height, width = width, height
    return cast("Resolution2D", (height, width))


@st.composite
def resolutions_with_more_than_one_pixel(
    draw: st.DrawFn,
) -> Resolution2D:
    height = draw(st.integers(min_value=1, max_value=MAX_DIM_SIZE))
    min_width = 2 if height == 1 else 1
    width = draw(st.integers(min_value=min_width, max_value=MAX_DIM_SIZE))
    return cast("Resolution2D", (height, width))


@st.composite
def default_n_scales(
    draw: st.DrawFn,
    min_value: int = 1,
    max_value: int = MAX_SCALES,
) -> int:
    return draw(st.integers(min_value=min_value, max_value=max_value))


@st.composite
def default_max_octaves(
    draw: st.DrawFn,
    min_value: int = 1,
    max_value: int = int(np.log2(MAX_DIM_SIZE)) + 2,
    allow_none: bool = True,
) -> int | None:
    if allow_none:
        return draw(
            st.one_of(st.none(), st.integers(min_value=min_value, max_value=max_value))
        )
    return draw(st.integers(min_value=min_value, max_value=max_value))


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
def default_sigmas(
    draw: st.DrawFn,
    image: npt.NDArray[np.float32],
    min_fractional_sigma: float | None = None,
    max_fractional_sigma: float = 0.2,
) -> float:
    major_dim = max(image.shape)

    hard_min_fractional_sigma = 1 / major_dim
    if min_fractional_sigma is None:
        min_fractional_sigma = hard_min_fractional_sigma
    else:
        min_fractional_sigma = max(min_fractional_sigma, hard_min_fractional_sigma)

    sigma_max = max(max_fractional_sigma * major_dim, 1.0)
    sigma_min = max(min_fractional_sigma * major_dim, 1.0)
    return draw(st.floats(min_value=sigma_min, max_value=sigma_max))


# Images
# -----------------------------------------------------------------------------


@st.composite
def default_elements(
    draw: st.DrawFn,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> float:
    return draw(
        st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            width=32,
        )
    )


@st.composite
def default_images(
    draw: st.DrawFn,
    resolution_strategy: st.SearchStrategy[Resolution2D] | None = None,
    elements_strategy: st.SearchStrategy[float] | None = None,
    unique: bool = False,
) -> npt.NDArray[np.float32]:
    resolution_strategy = resolution_strategy or default_resolutions()
    elements_strategy = elements_strategy or default_elements()
    return draw(
        arrays(
            dtype=np.float32,
            shape=draw(resolution_strategy),
            elements=elements_strategy,
            unique=unique,
        )
    )


@st.composite
def full_images(
    draw: st.DrawFn,
    resolution_strategy: st.SearchStrategy[Resolution2D] | None = None,
    elements_strategy: st.SearchStrategy[float] | None = None,
) -> npt.NDArray[np.float32]:
    resolution_strategy = resolution_strategy or default_resolutions()
    elements_strategy = elements_strategy or default_elements()
    return np.full(
        draw(resolution_strategy),
        draw(elements_strategy),
        dtype=np.float32,
    )


@st.composite
def random_images(
    draw: st.DrawFn,
    resolution_strategy: st.SearchStrategy[Resolution2D] | None = None,
) -> npt.NDArray[np.float32]:
    resolution_strategy = resolution_strategy or default_resolutions()
    resolution = draw(resolution_strategy)
    return np.random.uniform(size=resolution).astype(np.float32)


# Pyramids
# -----------------------------------------------------------------------------


@st.composite
def full_pyramids(draw: st.DrawFn, fill_value: float = 1.0) -> Pyramid:
    resolution = draw(default_resolutions())
    n_scales = draw(default_n_scales())
    max_octaves = draw(default_max_octaves())
    octave_shapes = pyramid_octave_shapes(
        resolution,
        max_octaves=max_octaves,
    )

    n_octaves = len(octave_shapes)
    pyramid_data = np.zeros((n_octaves, n_scales), dtype=object)
    for octave, octave_shape in enumerate(octave_shapes):
        for scale in range(n_scales):
            pyramid_data[octave, scale] = np.full(
                octave_shape,
                fill_value,
                dtype=np.float32,
            )
    return Pyramid(pyramid_data)


@st.composite
def valid_input_pyramid_for_laplacian_pyramid(
    draw: st.DrawFn, fill_value: float = 1.0
) -> Pyramid:
    image_width = draw(st.integers(min_value=2, max_value=MAX_DIM_SIZE))
    image_height = draw(st.integers(min_value=2, max_value=MAX_DIM_SIZE))
    n_scales = draw(st.integers(min_value=1, max_value=10))
    max_octaves = draw(
        st.integers(min_value=2, max_value=int(2 * np.log2(MAX_DIM_SIZE)))
    )
    octave_shapes = pyramid_octave_shapes(
        (image_height, image_width),
        max_octaves=max_octaves,
    )
    pyramid_data = np.zeros((len(octave_shapes), n_scales), dtype=object)
    for octave_num, octave_shape in enumerate(octave_shapes):
        for scale_num in range(n_scales):
            pyramid_data[octave_num, scale_num] = np.full(
                octave_shape,
                fill_value,
                dtype=np.float32,
            )
    return Pyramid(pyramid_data)


@st.composite
def differently_shaped_pyramids(
    draw: st.DrawFn,
    fill_value: float = 1.0,
) -> tuple[Pyramid, Pyramid]:
    pyramid_1 = draw(valid_input_pyramid_for_laplacian_pyramid(fill_value=fill_value))
    pyramid_2 = draw(
        valid_input_pyramid_for_laplacian_pyramid(fill_value=fill_value).filter(
            lambda pyr: pyr.shape != pyramid_1.shape
        )
    )
    return (pyramid_1, pyramid_2)


# -----------------------------------------------------------------------------


class PyramidTest(unittest.TestCase):
    @given(ndim=st.sampled_from([0, 1, 3, 4]))
    def test_cannot_create_pyramid_with_non_2d_contents(self, ndim: int) -> None:
        with self.assertRaises(AssertionError):
            Pyramid(np.zeros((2,) * ndim, dtype=object))

    def test_can_create_pyramid_with_2d_contents(self) -> None:
        Pyramid(np.zeros((2, 2), dtype=object))

    @given(
        n_octaves=st.integers(min_value=1, max_value=MAX_OCTAVES),
        n_scales=st.integers(min_value=1, max_value=MAX_SCALES),
    )
    def test_apply_applies_function_to_each_element(
        self,
        n_octaves: int,
        n_scales: int,
    ) -> None:
        data = np.array(
            [[Mock() for _ in range(n_scales)] for _ in range(n_octaves)], dtype=object
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
    @given(resolution=default_resolutions())
    def test_generates_maximum_possible_shapes_when_max_level_is_none(
        self,
        resolution: Resolution2D,
    ) -> None:
        computed_shapes: list[Resolution2D] = pyramid_octave_shapes(resolution)
        expected_shapes: list[Resolution2D] = []
        while min(resolution) >= 1:
            expected_shapes.append(resolution)
            resolution = cast("Resolution2D", (resolution[0] // 2, resolution[1] // 2))
        self.assertEqual(expected_shapes, computed_shapes)

    @given(
        resolution=default_resolutions(),
        max_octaves=default_max_octaves(allow_none=False),
    )
    def test_max_octaves_limits_number_of_shapes(
        self,
        resolution: Resolution2D,
        max_octaves: int,
    ) -> None:
        computed_shapes = pyramid_octave_shapes(resolution, max_octaves=max_octaves)
        self.assertLessEqual(len(computed_shapes), max_octaves)


@dataclass(frozen=True)
class GaussianPyramidParams:
    image: npt.NDArray[np.float32]
    sigma: float
    n_scales: int
    max_octaves: int | None


@st.composite
def gaussian_pyramid_params(
    draw: st.DrawFn,
    image_strategy: st.SearchStrategy[npt.NDArray[np.float32]] | None = None,
    sigma_strategy: st.SearchStrategy[float] | None = None,
    n_scales_strategy: st.SearchStrategy[int] | None = None,
    max_octaves_strategy: st.SearchStrategy[int | None] | None = None,
) -> GaussianPyramidParams:
    """Generate parameters for calls to `gaussian_pyramid`.

    Args:
        draw: The hypothesis draw function.
        image_strategy: A strategy for generating images or None. Defaults to
          `random_images`.
        sigma_strategy: A strategy for generating sigmas or None. Defaults to
          `sigmas`.
        n_scales_strategy: A strategy for generating n_scales or None. Defaults to
          `default_n_scales`.
        max_octaves_strategy: A strategy for max_octaves or None. Defaults to
          `default_max_octaves`.

    Returns:
        The parameters for a call to `gaussian_pyramid`.
    """
    image_strategy = image_strategy or random_images()
    image = draw(image_strategy)

    sigma_strategy = sigma_strategy or default_sigmas(image)
    sigma = draw(sigma_strategy)

    n_scales_strategy = n_scales_strategy or default_n_scales()
    n_scales = draw(n_scales_strategy)

    max_octaves_strategy = max_octaves_strategy or default_max_octaves()
    max_octaves = draw(max_octaves_strategy)

    return GaussianPyramidParams(
        image=image,
        sigma=sigma,
        n_scales=n_scales,
        max_octaves=max_octaves,
    )


@st.composite
def gaussian_pyramid_params_for_multiplane_pyramids(
    draw: st.DrawFn,
    image_strategy: st.SearchStrategy[npt.NDArray[np.float32]] | None = None,
    sigma_strategy: st.SearchStrategy[float] | None = None,
) -> GaussianPyramidParams:
    """Generate parameters for calls to `gaussian_pyramid` for a multi-plane pyramid.

    Args:
        draw: The hypothesis draw function.
        image_strategy: A strategy for generating images or None. Defaults to
          `random_images`.
        sigma_strategy: A strategy for generating sigmas or None. Defaults to
          `sigmas`.

    Returns:
        The parameters for a call to `gaussian_pyramid`.

    Raises:
        ValueError: If the image has no size.
    """
    if image_strategy is None:
        resolution_strategy = resolutions_with_more_than_one_pixel()
        default_image_strategy = default_images(resolution_strategy=resolution_strategy)
        random_image_strategy = random_images(resolution_strategy=resolution_strategy)
        image_strategy = st.one_of(default_image_strategy, random_image_strategy)

    image = draw(image_strategy)
    octave_shapes = pyramid_octave_shapes(image.shape)
    if len(octave_shapes) == 0:
        raise ValueError(
            "Cannot generate parameters for a multi-plane pyramid with no octaves."
        )

    # If we can't get more than one octave, we need to use a minimum of 2 scales.
    if len(octave_shapes) == 1:
        n_scales = draw(default_n_scales(min_value=2))
        max_octaves = 1
    else:
        # If only one scale, ensure we have at least one octave.
        n_scales = draw(default_n_scales())
        min_max_octaves = 2 if n_scales == 1 else 1
        max_octaves = draw(default_max_octaves(min_value=min_max_octaves))

    sigma_strategy = sigma_strategy or default_sigmas(image)

    return GaussianPyramidParams(
        image=image,
        sigma=draw(sigma_strategy),
        n_scales=n_scales,
        max_octaves=max_octaves,
    )


class GaussianPyramidTest(unittest.TestCase):
    INTRA_OCTAVE_VARIANCE_TOLERANCE = 1e-6
    INTER_OCTAVE_VARIANCE_TOLERANCE = 1e-4

    @given(resolution=resolutions_with_a_zero_length_dimension())
    @example(resolution=cast("Resolution2D", (0, 0)))
    def test_raises_value_error_if_image_has_no_values(
        self,
        resolution: Resolution2D,
    ) -> None:
        image = np.zeros(resolution, dtype=np.float32)
        with self.assertRaises(ValueError):
            gaussian_pyramid(image, sigma=1.0, n_scales=1)

    @given(params=gaussian_pyramid_params(sigma_strategy=st.just(1.0)))
    def test_has_correct_shape(
        self,
        params: GaussianPyramidParams,
    ) -> None:

        expected_octave_shapes = pyramid_octave_shapes(
            cast("Resolution2D", params.image.shape),
            max_octaves=params.max_octaves,
        )
        pyr = gaussian_pyramid(
            params.image,
            sigma=params.sigma,
            n_scales=params.n_scales,
            max_octaves=params.max_octaves,
        )
        # Test pyramid is correct shape.
        self.assertEqual(pyr.n_octaves, len(expected_octave_shapes))
        self.assertEqual(pyr.n_scales, params.n_scales)

        # Test each plane in pyramid is correct shape.
        for octave in range(pyr.n_octaves):
            for scale in range(pyr.n_scales):
                self.assertEqual(
                    pyr.data[octave, scale].shape, expected_octave_shapes[octave]
                )

    @settings(deadline=1000)
    @given(params=gaussian_pyramid_params_for_multiplane_pyramids())
    def test_subsequent_planes_have_decreasing_total_variation(
        self,
        params: GaussianPyramidParams,
    ) -> None:
        pyr = gaussian_pyramid(
            params.image,
            sigma=params.sigma,
            n_scales=params.n_scales,
            max_octaves=params.max_octaves,
        )
        # Check that variance decreases as scales increase within each octave.
        for octave in range(pyr.n_octaves):
            intra_octave_variances = [np.var(plane) for plane in pyr.data[octave]]
            deltas = np.diff(intra_octave_variances)
            nptest.assert_array_less(deltas, self.INTRA_OCTAVE_VARIANCE_TOLERANCE)

        # Check that variance decreases when jumping to the next octave.
        # Note, however, that the error in the variance estimate increases as we
        # use smaller and smaller images. Therefore, we have to use a larger tolerance
        # value for this comparison than for the intra-octave comparisons.
        for octave in range(pyr.n_octaves - 1):
            last_plane, next_plane = pyr.data[octave, -1], pyr.data[octave + 1, 0]
            delta = np.var(next_plane) - np.var(last_plane)
            self.assertLessEqual(delta, self.INTER_OCTAVE_VARIANCE_TOLERANCE)

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
                np.zeros((MAX_DIM_SIZE, MAX_DIM_SIZE)),
                center_sigma=center_sigma,
                surround_sigma=surround_sigma,
                n_scales=2,
                max_octaves=5,
            )

    @given(
        image=arrays(dtype=np.float32, shape=(MAX_DIM_SIZE, MAX_DIM_SIZE)),
        n_scales=st.integers(min_value=1, max_value=10),
        max_octaves=st.integers(min_value=1, max_value=int(2 * np.log2(MAX_DIM_SIZE))),
    )
    def test_center_and_surround_pyramids_have_same_shape(
        self,
        image: npt.NDArray[np.float32],
        n_scales: int,
        max_octaves: int,
    ) -> None:
        center, surround = center_surround_pyramids(
            image,
            center_sigma=3.0,
            surround_sigma=5.0,
            n_scales=n_scales,
            max_octaves=max_octaves,
        )
        self.assertEqual(center.shape, surround.shape)

    @given(
        image=default_images(),
        sigmas=center_surround_sigmas(min_center_sigma=1.0, max_center_sigma=3.0),
        n_scales=st.integers(min_value=1, max_value=MAX_SCALES),
        max_octaves=st.integers(min_value=1, max_value=MAX_OCTAVES),
    )
    def test_surround_planes_have_higher_mean_local_variation_than_corresponding_center_planes(  # noqa: E501
        self,
        image: npt.NDArray[np.float32],
        sigmas: tuple[float, float],
        n_scales: int,
        max_octaves: int,
    ) -> None:
        center, surround = center_surround_pyramids(
            image,
            center_sigma=sigmas[0],
            surround_sigma=sigmas[1],
            n_scales=n_scales,
            max_octaves=max_octaves,
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
        image=full_images(),
        sigmas=center_surround_sigmas(min_center_sigma=0.5, max_center_sigma=3.0),
        n_scales=st.integers(min_value=1, max_value=3),
        max_octaves=st.integers(min_value=1, max_value=int(2 * np.log2(MAX_DIM_SIZE))),
    )
    def test_surround_planes_mean_local_variation_equals_corresponding_center_planes_for_solid_image(  # noqa: E501
        self,
        image: npt.NDArray[np.float32],
        sigmas: tuple[float, float],
        n_scales: int,
        max_octaves: int,
    ) -> None:
        center, surround = center_surround_pyramids(
            image,
            center_sigma=sigmas[0],
            surround_sigma=sigmas[1],
            n_scales=n_scales,
            max_octaves=max_octaves,
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
        image=default_images(unique=False),
        sigmas=center_surround_sigmas(min_center_sigma=1.0, max_center_sigma=3.0),
        n_scales=default_n_scales(),
        max_octaves=default_max_octaves(),
    )
    def test_surround_planes_have_higher_mean_local_variation_than_corresponding_center_planes_for_sufficiently_variable_image(  # noqa: E501
        self,
        image: npt.NDArray[np.float32],
        sigmas: tuple[float, float],
        n_scales: int,
        max_octaves: int,
    ) -> None:
        center, surround = center_surround_pyramids(
            image,
            center_sigma=sigmas[0],
            surround_sigma=sigmas[1],
            n_scales=n_scales,
            max_octaves=max_octaves,
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
            "tbp.monty.frameworks.models.salience.strategies.vocus2.pyramids.resize",
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
        pyramid=full_pyramids(fill_value=INPUT_FILL_VALUE),
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
            "tbp.monty.frameworks.models.salience.strategies.vocus2.pyramids.resize",
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


