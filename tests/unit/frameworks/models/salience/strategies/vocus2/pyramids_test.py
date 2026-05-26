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
from unittest.mock import Mock, patch, sentinel

import cv2
import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
from hypothesis import given, settings
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
from tbp.monty.math import DEFAULT_TOLERANCE
from tests.unit.statistics import mean_local_variation

# Common upper limits used in these tests. Not the same thing
# as safe operating limits.
MAX_DIM_SIZE = 512
MAX_OCTAVES = 2 * (int(np.log2(MAX_DIM_SIZE)) + 1)
MAX_SCALES = 5

MAX_FRACTIONAL_CENTER_SIGMA = 0.1
MAX_FRACTIONAL_SURROUND_SIGMA = 0.5
MIN_FRACTIONAL_SIGMA_SEPARATION = 0.02


@st.composite
def default_resolutions(
    draw: st.DrawFn,
) -> tuple[int, int]:
    height = draw(st.integers(min_value=1, max_value=MAX_DIM_SIZE))
    width = draw(st.integers(min_value=1, max_value=MAX_DIM_SIZE))
    return (height, width)


@st.composite
def default_n_scales(draw: st.DrawFn) -> int:
    return draw(st.integers(min_value=1, max_value=MAX_SCALES))


@st.composite
def default_max_octaves(
    draw: st.DrawFn,
    min_value: int = 1,
    max_value: int = MAX_OCTAVES,
    allow_none: bool = True,
) -> int | None:
    if allow_none:
        return draw(
            st.one_of(st.none(), st.integers(min_value=min_value, max_value=max_value))
        )
    return draw(st.integers(min_value=min_value, max_value=max_value))


@st.composite
def default_sigmas(
    draw: st.DrawFn,
    resolution: tuple[int, int],
) -> float:
    min_dim_size = min(resolution)
    if min_dim_size == 1:
        return 1.0

    min_fractional_sigma = 1 / min_dim_size
    fractional_sigma = draw(
        st.floats(
            min_value=min_fractional_sigma, max_value=MAX_FRACTIONAL_SURROUND_SIGMA
        )
    )
    return fractional_sigma * min_dim_size


@st.composite
def default_image_values(draw: st.DrawFn) -> float:
    return draw(
        st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            width=32,
        )
    )


@st.composite
def default_images(
    draw: st.DrawFn,
    resolution: st.SearchStrategy[tuple[int, int]] | None = None,
    elements: st.SearchStrategy[float] | None = None,
    unique: bool = False,
) -> npt.NDArray[np.float32]:
    resolution = resolution or default_resolutions()
    elements = elements or default_image_values()
    return draw(
        arrays(
            dtype=np.float32,
            shape=resolution,
            elements=elements,
            unique=unique,
        )
    )


@st.composite
def solid_images(
    draw: st.DrawFn,
    resolution: st.SearchStrategy[tuple[int, int]] | None = None,
    elements: st.SearchStrategy[float] | None = None,
) -> npt.NDArray[np.float32]:
    resolution = resolution or default_resolutions()
    elements = elements or default_image_values()
    return np.full(
        draw(resolution),
        draw(elements),
        dtype=np.float32,
    )


@st.composite
def random_images(
    draw: st.DrawFn,
    resolution: st.SearchStrategy[tuple[int, int]] | None = None,
) -> npt.NDArray[np.float32]:
    resolution = resolution or default_resolutions()
    resolution = draw(resolution)
    return np.random.uniform(size=resolution).astype(np.float32)


@st.composite
def nonsolid_images(draw: st.DrawFn) -> npt.NDArray[np.float32]:
    return draw(
        default_images().filter(
            lambda img: not np.allclose(img, img[0, 0], atol=DEFAULT_TOLERANCE)
        )
    )


@st.composite
def default_pyramids(
    draw: st.DrawFn,
    fill_value: float = 0.0,
    resolution: st.SearchStrategy[tuple[int, int]] | None = None,
    n_scales: st.SearchStrategy[int] | None = None,
    max_octaves: st.SearchStrategy[int | None] | None = None,
) -> Pyramid:
    resolution = resolution or default_resolutions()
    _resolution = draw(resolution)

    n_scales = n_scales or default_n_scales()
    _n_scales = draw(n_scales)

    max_octaves = max_octaves or default_max_octaves()
    _max_octaves = draw(max_octaves)

    octave_shapes = pyramid_octave_shapes(_resolution, max_octaves=_max_octaves)
    n_octaves = len(octave_shapes)
    pyramid_data = np.zeros((n_octaves, _n_scales), dtype=object)
    for octave, octave_shape in enumerate(octave_shapes):
        for scale in range(_n_scales):
            pyramid_data[octave, scale] = np.full(
                octave_shape,
                fill_value,
                dtype=np.float32,
            )
    return Pyramid(pyramid_data)


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
        resolution: tuple[int, int],
    ) -> None:
        computed_shapes = pyramid_octave_shapes(resolution)
        expected_shapes = []
        while min(resolution) >= 1:
            expected_shapes.append(resolution)
            resolution = (resolution[0] // 2, resolution[1] // 2)
        self.assertEqual(expected_shapes, computed_shapes)

    @given(
        resolution=default_resolutions(),
        max_octaves=default_max_octaves(allow_none=False),
    )
    def test_max_octaves_limits_number_of_shapes(
        self,
        resolution: tuple[int, int],
        max_octaves: int,
    ) -> None:
        computed_shapes = pyramid_octave_shapes(resolution, max_octaves=max_octaves)
        self.assertLessEqual(len(computed_shapes), max_octaves)


# --------------------------------------------------------------------------------------
# Gaussian Pyramid Tests
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GaussianPyramidParams:
    image: npt.NDArray[np.float32]
    sigma: float
    n_scales: int
    max_octaves: int | None


@st.composite
def gaussian_pyramid_params(
    draw: st.DrawFn,
    image: st.SearchStrategy[npt.NDArray[np.float32]] | None = None,
    sigma: st.SearchStrategy[float] | None = None,
    n_scales: st.SearchStrategy[int] | None = None,
    max_octaves: st.SearchStrategy[int | None] | None = None,
) -> GaussianPyramidParams:
    """Generate parameters for calls to `gaussian_pyramid`.

    Args:
        draw: The hypothesis draw function.
        image: A strategy for generating images or None. Defaults to
          `random_images`.
        sigma: A strategy for generating sigmas or None. Defaults to
          `sigmas`.
        n_scales: A strategy for generating n_scales or None. Defaults to
          `default_n_scales`.
        max_octaves: A strategy for max_octaves or None. Defaults to
          `default_max_octaves`.

    Returns:
        The parameters for a call to `gaussian_pyramid`.
    """
    image = image or default_images()
    _image = draw(image)

    sigma = sigma or default_sigmas(_image.shape)
    _sigma = draw(sigma)

    n_scales = n_scales or default_n_scales()
    _n_scales = draw(n_scales)

    max_octaves = max_octaves or default_max_octaves()
    _max_octaves = draw(max_octaves)

    return GaussianPyramidParams(
        image=_image,
        sigma=_sigma,
        n_scales=_n_scales,
        max_octaves=_max_octaves,
    )


class GaussianPyramidTest(unittest.TestCase):
    @given(
        resolution=st.one_of(
            st.tuples(st.just(0), st.integers(min_value=0, max_value=MAX_DIM_SIZE)),
            st.tuples(st.integers(min_value=0, max_value=MAX_DIM_SIZE), st.just(0)),
        )
    )
    def test_raises_value_error_if_image_has_size_zero(
        self,
        resolution: tuple[int, int],
    ) -> None:
        image = np.zeros(resolution, dtype=np.float32)
        with self.assertRaises(ValueError):
            gaussian_pyramid(image, sigma=Mock(), n_scales=Mock())

    @given(params=gaussian_pyramid_params(sigma=st.just(1.0)))
    def test_has_correct_shape(self, params: GaussianPyramidParams) -> None:
        expected_octave_shapes = pyramid_octave_shapes(
            params.image.shape,
            max_octaves=params.max_octaves,
        )
        pyr = gaussian_pyramid(
            params.image,
            sigma=params.sigma,
            n_scales=params.n_scales,
            max_octaves=params.max_octaves,
        )
        # Check the pyramid itself has the correct shape.
        self.assertEqual(pyr.n_octaves, len(expected_octave_shapes))
        self.assertEqual(pyr.n_scales, params.n_scales)

        # Check each plane in the pyramid has the correct shape.
        for octave in range(pyr.n_octaves):
            for scale in range(pyr.n_scales):
                self.assertEqual(
                    pyr.data[octave, scale].shape, expected_octave_shapes[octave]
                )

    @settings(deadline=1000)
    @given(
        params=gaussian_pyramid_params(
            image=default_images(),
        ),
    )
    def test_subsequent_planes_have_decreasing_variance(
        self,
        params: GaussianPyramidParams,
    ) -> None:
        # Test that each plane in the pyramid is blurrier than its predecessor.
        #
        # Here we use standard deviation to quantify each plane's blurriness.
        # Therefore, we want to show that the following holds for all i:
        #
        #                   std(plane[i]) > std(plane[i+1])
        #
        # In reality, the above condition will not hold due several factors.
        #
        #   1. Downsampling an image can slightly increase variance. There are mundane
        #      and not concerning reasons for this.
        #   2. Gaussian blur can also increase variance around the border. As planes
        #      get smaller, the border pixels take up a larger fraction of the plane,
        #      so boundary artifacts have an outsized impact on global variance.
        #   3. Small planes also means fewer pixels, meaning our statistics get noisier
        #      and less stable.
        #   4. For solid (or nearly solid) images, artifacts due to the above causes
        #      won't be counteracted out by variance reductions elsewhere in the image.
        #
        # If we naively add a tolerance to our checks and use it for every comparison,
        # then we'd never be able to check that variance ever decreases. Instead,
        # we'll start each comparison assuming a tolerance of 0 and widen it
        # as necessary.
        #
        #   1. When comparing the last plane in octave i with the first plane in the
        #      octave i+1, pad the tolerance with `downsampling_tolerance` to
        #      accommodate for downsampling artifacts.
        #   2. When comparing very small planes, pad the tolerance with
        #      `small_plane_tolerance` to accommodate for statistical noise.
        #      to accommodate for statistical noise.
        #   3. When planes already have very low variance, we grant extra tolerance.
        pyr = gaussian_pyramid(
            params.image,
            sigma=params.sigma,
            n_scales=params.n_scales,
            max_octaves=params.max_octaves,
        )
        downsampling_tolerance = 1e-3
        small_plane_threshold = 4
        small_plane_tolerance = 1e-6
        ignore_below_variance = 1e-6

        planes = list(pyr.flat)
        for i in range(len(planes) - 1):
            tolerance = 0.0
            plane_a = planes[i]
            plane_b = planes[i + 1]

            if plane_a.shape != plane_b.shape:
                tolerance += downsampling_tolerance

            if (
                min(plane_a.shape) < small_plane_threshold
                or min(plane_b.shape) < small_plane_threshold
            ):
                tolerance += small_plane_tolerance

            var_a = np.std(plane_a)
            var_b = np.std(plane_b)

            delta = var_b - var_a

            if plane_a.size == 1 and plane_b.size == 1:
                self.assertEqual(delta, 0.0)
                continue

            if var_a <= ignore_below_variance and var_b <= ignore_below_variance:
                tolerance += ignore_below_variance

            self.assertLess(delta, tolerance)


class GaussianPyramidCalibrationTest(unittest.TestCase):
    def test_subsequent_planes_have_decreasing_variance(
        self,
    ) -> None:
        # This is a more specific and stricter test than the one above.
        #
        # While the more general test a wider range of inputs, it also has to make
        # complex allowances for the many ways that spurious increases in global
        # variance can be introduced. This made it impossible to definitely check
        # for the basic properties we expect of a multi-scale gaussian pyramid 99.99%
        # of the time.
        #
        # This test complements the more general test by performing maximally strict
        # checks on a known example.
        #
        # If this fails, then something has changed that should probably be attended to.
        rng = np.random.RandomState(67)
        image = rng.uniform(0.0, 1.0, size=(512, 512))
        pyr = gaussian_pyramid(image, sigma=3.0, n_scales=2)

        planes = list(pyr.flat)
        for i in range(len(planes) - 1):
            plane_a = planes[i]
            plane_b = planes[i + 1]

            var_a = np.std(plane_a)
            var_b = np.std(plane_b)
            delta = var_b - var_a

            if plane_a.size == 1 and plane_b.size == 1:
                self.assertEqual(delta, 0.0)
            else:
                self.assertLess(delta, 0.0)


@dataclass(frozen=True)
class CenterSurroundPyramidsParams:
    image: npt.NDArray[np.float32]
    center_sigma: float
    surround_sigma: float
    n_scales: int
    max_octaves: int | None


@st.composite
def default_cs_sigmas(
    draw: st.DrawFn,
    resolution: tuple[int, int],
) -> tuple[float, float]:
    min_dim_size = min(resolution)
    min_center_sigma = 1.0
    max_center_sigma = min_dim_size * MAX_FRACTIONAL_CENTER_SIGMA
    max_surround_sigma = min_dim_size * MAX_FRACTIONAL_SURROUND_SIGMA
    min_sigma_separation = min_dim_size * MIN_FRACTIONAL_SIGMA_SEPARATION

    if min_dim_size == 1:
        return (1.0, 1.0 + MIN_FRACTIONAL_SIGMA_SEPARATION)

    center_sigma = draw(
        st.floats(
            min_value=min_center_sigma,
            max_value=max(min_center_sigma, max_center_sigma),
        )
    )
    min_surround_sigma = center_sigma + min_sigma_separation
    surround_sigma = draw(
        st.floats(
            min_value=min_surround_sigma,
            max_value=max(min_surround_sigma, max_surround_sigma),
        )
    )
    return (center_sigma, surround_sigma)


@st.composite
def center_surround_pyramids_params(
    draw: st.DrawFn,
    image: st.SearchStrategy[npt.NDArray[np.float32]] | None = None,
) -> CenterSurroundPyramidsParams:
    """Generate parameters for calls to `gaussian_pyramid`.

    Args:
        draw: The hypothesis draw function.
        image: A strategy for generating images or None.
        cs_sigmas: A strategy for generating sigmas or None.

    Returns:
        The parameters for a call to `gaussian_pyramid`.
    """
    image = image or default_images()
    _image = draw(image)

    center_sigma, surround_sigma = draw(default_cs_sigmas(_image.shape))

    _n_scales = draw(default_n_scales())

    _max_octaves = draw(default_max_octaves())

    return CenterSurroundPyramidsParams(
        image=_image,
        center_sigma=center_sigma,
        surround_sigma=surround_sigma,
        n_scales=_n_scales,
        max_octaves=_max_octaves,
    )


class CenterSurroundPyramidsTest(unittest.TestCase):
    @given(
        center_sigma=st.floats(min_value=1.0, max_value=10.0),
        ratio=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_raises_value_error_if_center_sigma_is_greater_than_or_equal_to_surround_sigma(  # noqa: E501
        self,
        center_sigma: float,
        ratio: float,
    ) -> None:
        surround_sigma = center_sigma * ratio
        with self.assertRaises(ValueError):
            center_surround_pyramids(
                Mock(), center_sigma, surround_sigma, Mock(), Mock()
            )

    @given(params=center_surround_pyramids_params())
    def test_center_and_surround_pyramids_have_same_shape(
        self,
        params: CenterSurroundPyramidsParams,
    ) -> None:
        center, surround = center_surround_pyramids(
            params.image,
            center_sigma=params.center_sigma,
            surround_sigma=params.surround_sigma,
            n_scales=params.n_scales,
            max_octaves=params.max_octaves,
        )
        self.assertEqual(center.shape, surround.shape)

    @settings(deadline=1000)
    @given(params=center_surround_pyramids_params())
    def test_center_planes_have_higher_variaiance_than_corresponding_surround_planes(
        self,
        params: CenterSurroundPyramidsParams,
    ) -> None:
        center, surround = center_surround_pyramids(
            params.image,
            center_sigma=params.center_sigma,
            surround_sigma=params.surround_sigma,
            n_scales=params.n_scales,
            max_octaves=params.max_octaves,
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

    @settings(deadline=1000)
    @given(params=center_surround_pyramids_params(image=solid_images()))
    def test_center_plane_variance_equals_corresponding_surround_plane_variance_for_solid_image(  # noqa: E501
        self,
        params: CenterSurroundPyramidsParams,
    ) -> None:
        center, surround = center_surround_pyramids(
            params.image,
            center_sigma=params.center_sigma,
            surround_sigma=params.surround_sigma,
            n_scales=params.n_scales,
            max_octaves=params.max_octaves,
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


class CenterSurroundPyramidsCalibrationTest(unittest.TestCase):
    def test_center_planes_have_variance_than_corresponding_surround_planes(
        self,
    ) -> None:
        rng = np.random.RandomState(67)
        image = rng.uniform(0.0, 1.0, size=(512, 512))
        center_sigma = 3.0
        surround_sigma = 5.0
        n_scales = 2
        max_octaves = 5

        center, surround = center_surround_pyramids(
            image,
            center_sigma=center_sigma,
            surround_sigma=surround_sigma,
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


@st.composite
def valid_input_pyramid_for_laplacian_pyramid(
    draw: st.DrawFn,
    fill_value: float = 0.0,
) -> Pyramid:
    height = draw(st.integers(min_value=2, max_value=MAX_DIM_SIZE))
    width = draw(st.integers(min_value=2, max_value=MAX_DIM_SIZE))
    resolution = (height, width)
    n_scales = draw(default_n_scales())
    max_octaves = draw(st.one_of(st.none(), default_max_octaves(min_value=2)))

    octave_shapes = pyramid_octave_shapes(resolution, max_octaves=max_octaves)
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


# --------------------------------------------------------------------------------------
# Pyramid Combine Tests
# --------------------------------------------------------------------------------------


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

    @given(pyramids=differently_shaped_pyramids())
    def test_raises_value_error_if_pyramids_have_different_shapes(
        self,
        pyramids: list[Pyramid],
    ) -> None:
        with self.assertRaises(ValueError):
            pyramid_combine(pyramids, Mock())

    @given(pyramid=valid_input_pyramid_for_laplacian_pyramid())
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


# --------------------------------------------------------------------------------------
# Pyramid Collapse Tests
# --------------------------------------------------------------------------------------


class PyramidCollapseTest(unittest.TestCase):
    INPUT_FILL_VALUE = 0.0

    @settings(deadline=1000)
    @given(
        pyramid=default_pyramids(fill_value=INPUT_FILL_VALUE),
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
