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
MAX_DIM_SIZE = 512
MAX_OCTAVES = int(np.log2(MAX_DIM_SIZE)) + 1
MAX_SCALES = 5


# Parameters
# -----------------------------------------------------------------------------


@st.composite
def default_resolutions(
    draw: st.DrawFn,
    min_dim_size: int = 1,
    max_dim_size: int = MAX_DIM_SIZE,
) -> tuple[int, int]:
    height = draw(st.integers(min_value=min_dim_size, max_value=max_dim_size))
    width = draw(st.integers(min_value=min_dim_size, max_value=max_dim_size))
    return (height, width)


@st.composite
def resolutions_with_more_than_one_pixel(
    draw: st.DrawFn,
) -> tuple[int, int]:
    height = draw(st.integers(min_value=1, max_value=MAX_DIM_SIZE))
    min_width = 2 if height == 1 else 1
    width = draw(st.integers(min_value=min_width, max_value=MAX_DIM_SIZE))
    return (height, width)


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
    max_fractional_sigma: float = 0.5,
) -> float:
    smallest_dim = min(image.shape)
    if smallest_dim == 1:
        return draw(st.just(1.0))

    smallest_fractional_sigma = 1 / smallest_dim
    if min_fractional_sigma is None:
        min_fractional_sigma = smallest_fractional_sigma
    else:
        min_fractional_sigma = max(min_fractional_sigma, smallest_fractional_sigma)

    sigma_min = max(min_fractional_sigma * smallest_dim, 1.0)
    sigma_max = max_fractional_sigma * smallest_dim
    sigma_max = max(sigma_min, sigma_max)

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
def solid_images(
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
def default_pyramids(
    draw: st.DrawFn,
    fill_value: float = 0.0,
    resolution_strategy: st.SearchStrategy[Resolution2D] | None = None,
    n_scales_strategy: st.SearchStrategy[int] | None = None,
    max_octaves_strategy: st.SearchStrategy[int | None] | None = None,
) -> Pyramid:
    resolution_strategy = resolution_strategy or default_resolutions()
    resolution = draw(resolution_strategy)

    n_scales_strategy = n_scales_strategy or default_n_scales()
    n_scales = draw(n_scales_strategy)

    max_octaves_strategy = max_octaves_strategy or default_max_octaves()
    max_octaves = draw(max_octaves_strategy)

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
    image_strategy = image_strategy or default_images()
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


# -----------------------------------------------------------------------------
# Gaussian Pyramid Tests
# -----------------------------------------------------------------------------


@st.composite
def images_with_size_zero(
    draw: st.DrawFn,
) -> tuple[int, int]:
    height, width = draw(
        st.one_of(
            st.tuples(st.just(0), st.integers(min_value=1, max_value=MAX_DIM_SIZE)),
            st.tuples(st.integers(min_value=1, max_value=MAX_DIM_SIZE), st.just(0)),
            st.tuples(st.just(0), st.just(0)),
        )
    )
    return np.zeros((height, width), dtype=np.float32)


class GaussianPyramidTest(unittest.TestCase):
    @given(image=images_with_size_zero())
    def test_raises_value_error_if_image_has_size_zero(
        self,
        image: npt.NDArray[np.float32],
    ) -> None:
        with self.assertRaises(ValueError):
            gaussian_pyramid(image, sigma=Mock(), n_scales=Mock())

    @given(params=gaussian_pyramid_params(sigma_strategy=st.just(1.0)))
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
            image_strategy=st.one_of(default_images(), random_images()),
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
        downsampling_tolerance = 5e-4
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

    def test_subsequent_planes_have_decreasing_variance_strict_with_specific_example(
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


def save_params(**kwargs) -> None:
    from pathlib import Path

    data_dir = Path.home() / "params"
    data_dir.mkdir(exist_ok=True)
    for key, value in kwargs.items():
        path = data_dir / f"{key}.npy"
        np.save(path, value)
