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
from typing import Callable
from unittest.mock import Mock

import numpy as np
import numpy.typing as npt
import scipy.signal
from hypothesis import given
from hypothesis import strategies as st

from tbp.monty.frameworks.models.salience.strategies.vocus2.vocus2 import (
    ColorChannelSalience,
    DepthSalience,
    SafeOperatingLimits,
)
from tbp.monty.frameworks.sensors import Resolution2D
from tests.unit.frameworks.models.salience.strategies.vocus2.pyramids_test import (
    MAX_DIM_SIZE,
    solid_images,
)

# Common upper limits used in these tests. Not the same thing
# as safe operating limits.
MAX_DIM_SIZE = 1024
MAX_SCALES = 5


# Parameters
# -----------------------------------------------------------------------------


@st.composite
def safe_resolutions(
    draw: st.DrawFn,
    min_dim_size: int = SafeOperatingLimits.min_image_dim_size,
    max_dim_size: int = MAX_DIM_SIZE,
) -> Resolution2D:
    height = draw(st.integers(min_value=min_dim_size, max_value=max_dim_size))
    width = draw(st.integers(min_value=min_dim_size, max_value=max_dim_size))
    return (height, width)


# @st.composite
# def default_cs_sigmas(
#     draw: st.DrawFn,
#     min_center_sigma: float,
#     max_center_sigma: float,
# ) -> tuple[float, float]:
#     center_sigma = draw(
#         st.floats(min_value=min_center_sigma, max_value=max_center_sigma)
#     )
#     surround_sigma_factor = draw(
#         st.floats(min_value=1.0, max_value=10.0, exclude_min=True)
#     )
#     return center_sigma, surround_sigma_factor * center_sigma


# # Images
# # -----------------------------------------------------------------------------


# @st.composite
# def solid_images(
#     draw: st.DrawFn, min_dim_size: int = 1, max_dim_size: int = MAX_DIM_SIZE
# ) -> npt.NDArray[np.float32]:
#     height = draw(st.integers(min_value=min_dim_size, max_value=max_dim_size))
#     width = draw(st.integers(min_value=min_dim_size, max_value=max_dim_size))
#     fill_value = draw(
#         st.floats(min_value=0.0, max_value=1.0, allow_nan=False, width=32)
#     )
#     return np.full((height, width), fill_value, dtype=np.float32)


@st.composite
def filled_images(
    draw: st.DrawFn,
    fill_value: float = 1.0,
    min_dim_size: int = 1,
    max_dim_size: int = MAX_DIM_SIZE,
) -> npt.NDArray[np.float32]:
    height = draw(st.integers(min_value=min_dim_size, max_value=max_dim_size))
    width = draw(st.integers(min_value=min_dim_size, max_value=max_dim_size))
    return np.full((height, width), fill_value, dtype=np.float32)


@dataclass
class CenterAndSurroundParams:
    center_sigma: float
    surround_sigma: float
    n_scales: int
    max_octaves: int


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

    return CenterAndSurroundParams(
        center_sigma=center_sigma,
        surround_sigma=surround_sigma,
        n_scales=n_scales,
        max_octaves=max_octaves,
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
        filled_images(
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


@dataclass
class ProcessParams:
    image: npt.NDArray[np.float32]


class ColorChannelSalienceTest(unittest.TestCase):
    MINIMUM_SALIENCE_THRESHOLD = 1e-3

    @given(
        image_and_processor=color_channel_salience_setup(
            color_channel_salience_processor, solid_images()
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
            filled_images(max_dim_size=SafeOperatingLimits.min_image_dim_size - 1),
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
            filled_images(max_dim_size=SafeOperatingLimits.min_image_dim_size - 1),
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
            filled_images(max_dim_size=SafeOperatingLimits.min_image_dim_size - 1),
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
        filled_images(
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
    )


class DepthSalienceTest(unittest.TestCase):
    MINIMUM_SALIENCE_THRESHOLD = 1e-3

    @given(
        image_and_processor=depth_salience_setup(
            depth_salience_processor,
            solid_images(min_dim_size=SafeOperatingLimits.min_image_dim_size),
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
            filled_images(max_dim_size=SafeOperatingLimits.min_image_dim_size - 1),
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
            filled_images(max_dim_size=SafeOperatingLimits.min_image_dim_size - 1),
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
            filled_images(max_dim_size=SafeOperatingLimits.min_image_dim_size - 1),
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
