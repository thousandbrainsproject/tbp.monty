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
from unittest.mock import Mock

import numpy as np
import numpy.typing as npt
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.ndimage import binary_dilation

from tbp.monty.frameworks.models.salience.strategies.vocus2.vocus2 import (
    ColorChannelSalience,
    DepthSalience,
    SafeOperatingLimits,
)
from tbp.monty.frameworks.sensors import Resolution2D
from tests.unit.frameworks.models.salience.strategies.vocus2.pyramids_test import (
    MAX_DIM_SIZE,
    default_cs_sigmas,
    default_image_values,
    default_images,
    default_max_octaves,
    default_n_scales,
)

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


@st.composite
def safe_cs_sigmas(
    draw: st.DrawFn,
    resolution: tuple[int, int],
) -> tuple[float, float]:
    center_sigma, surround_sigma = draw(
        default_cs_sigmas(
            resolution,
            min_fractional_sigma_separation=SafeOperatingLimits.min_fractional_sigma_separation,
            max_fractional_sigma=SafeOperatingLimits.max_fractional_sigma,
        )
    )
    return (center_sigma, surround_sigma)


# Images
# -----------------------------------------------------------------------------


@st.composite
def safe_images(
    draw: st.DrawFn,
    elements: st.SearchStrategy[float] | None = None,
    unique: bool = False,
) -> npt.NDArray[np.float32]:
    return draw(
        default_images(resolution=safe_resolutions(), elements=elements, unique=unique)
    )


@st.composite
def safe_solid_images(
    draw: st.DrawFn,
    elements: st.SearchStrategy[float] | None = None,
) -> npt.NDArray[np.float32]:
    elements = elements or default_image_values()
    resolution = draw(safe_resolutions())
    return np.full(
        resolution,
        draw(elements),
        dtype=np.float32,
    )


@st.composite
def safe_filled_images(
    draw: st.DrawFn,
    fill_value: float = 1.0,
) -> npt.NDArray[np.float32]:
    resolution = draw(safe_resolutions())
    return np.full(resolution, fill_value, dtype=np.float32)


# --------------------------------------------------------------------------------------
# Color Channel Salience
# --------------------------------------------------------------------------------------


@dataclass
class ColorChannelSalienceSetup:
    processor: ColorChannelSalience
    image: npt.NDArray[np.float32]
    box: npt.NDArray[np.bool_] | None = None


@st.composite
def color_channel_salience_setup(
    draw: st.DrawFn,
    image: st.SearchStrategy[npt.NDArray[np.float32]] | None = None,
) -> ColorChannelSalienceSetup:
    image = image or safe_images()
    _image = draw(image)

    center_sigma, surround_sigma = draw(safe_cs_sigmas(resolution=_image.shape))
    processor = ColorChannelSalience(
        center_sigma=center_sigma,
        surround_sigma=surround_sigma,
        n_scales=draw(default_n_scales()),
        max_octaves=draw(default_max_octaves()),
        operating_limits=SafeOperatingLimits(),
    )

    return ColorChannelSalienceSetup(
        processor=processor,
        image=_image,
        box=None,
    )


@st.composite
def box_salience_setup(
    draw: st.DrawFn,
    on_value: float = 1.0,
    off_value: float = 0.0,
) -> ColorChannelSalienceSetup:
    resolution = draw(safe_resolutions())
    cs_sigmas = safe_cs_sigmas(resolution=resolution)
    center_sigma, surround_sigma = draw(cs_sigmas)

    # Create a boolean mask that contains a box.
    # Then draw it on a background image.
    center = draw(
        st.tuples(
            st.integers(min_value=0, max_value=resolution[0]),
            st.integers(min_value=0, max_value=resolution[1]),
        )
    )
    width = draw(st.integers(min_value=1, max_value=resolution[1] // 2))
    height = width
    box = rectangular_mask(
        resolution=resolution,
        center=center,
        width=width,
        height=height,
    )
    image = np.full(resolution, off_value, dtype=np.float32)
    image[box] = on_value

    processor = ColorChannelSalience(
        center_sigma=center_sigma,
        surround_sigma=surround_sigma,
        n_scales=draw(default_n_scales()),
        max_octaves=draw(default_max_octaves()),
        operating_limits=SafeOperatingLimits(),
    )

    return ColorChannelSalienceSetup(
        processor=processor,
        image=image,
        box=box,
    )


class ColorChannelSalienceTest(unittest.TestCase):
    MINIMUM_SALIENCE_THRESHOLD = 1e-4

    @settings(deadline=1000)
    @given(
        setup=color_channel_salience_setup(
            image=safe_solid_images(),
        )
    )
    def test_solid_image_not_salient(
        self,
        setup: ColorChannelSalienceSetup,
    ) -> None:
        processor = setup.processor
        image = setup.image
        feature_map, _ = processor.process(Mock(), image)
        self.assertTrue(np.all(feature_map < self.MINIMUM_SALIENCE_THRESHOLD))

    @settings(deadline=1000)
    @given(
        setup=box_salience_setup(
            on_value=1.0,
            off_value=0.0,
        )
    )
    def test_box_is_more_salient_than_surround(
        self,
        setup: ColorChannelSalienceSetup,
    ) -> None:
        processor = setup.processor
        image = setup.image
        feature_map, _ = processor.process(Mock(), image)

        box = setup.box
        dilated = binary_dilation(box, iterations=5)
        surround = ~dilated
        surround = ~box

        box_salience = feature_map[box].mean()
        surround_salience = feature_map[surround].mean()
        self.assertTrue(box_salience > surround_salience)


# --------------------------------------------------------------------------------------
# Depth Salience
# --------------------------------------------------------------------------------------


@dataclass
class DepthSalienceSetup:
    processor: DepthSalience
    image: npt.NDArray[np.float32]
    box: npt.NDArray[np.bool_] | None = None


@st.composite
def depth_salience_setup(
    draw: st.DrawFn,
    image: st.SearchStrategy[npt.NDArray[np.float32]] | None = None,
) -> DepthSalienceSetup:
    image = image or safe_images()
    _image = draw(image)

    center_sigma, surround_sigma = draw(safe_cs_sigmas(resolution=_image.shape))
    processor = DepthSalience(
        center_sigma=center_sigma,
        surround_sigma=surround_sigma,
        n_scales=draw(default_n_scales()),
        max_octaves=draw(default_max_octaves()),
        operating_limits=SafeOperatingLimits(),
    )

    return DepthSalienceSetup(
        processor=processor,
        image=_image,
        box=None,
    )


@st.composite
def depth_box_salience_setup(
    draw: st.DrawFn,
    on_value: float = 1.0,
    off_value: float = 0.0,
) -> DepthSalienceSetup:
    resolution = draw(safe_resolutions())
    cs_sigmas = safe_cs_sigmas(resolution=resolution)
    center_sigma, surround_sigma = draw(cs_sigmas)

    # Create a boolean mask that contains a box.
    # Then draw it on a background image.
    center = draw(
        st.tuples(
            st.integers(min_value=0, max_value=resolution[0]),
            st.integers(min_value=0, max_value=resolution[1]),
        )
    )
    width = draw(st.integers(min_value=1, max_value=resolution[1] // 2))
    height = width
    box = rectangular_mask(
        resolution=resolution,
        center=center,
        width=width,
        height=height,
    )
    image = np.full(resolution, off_value, dtype=np.float32)
    image[box] = on_value

    processor = DepthSalience(
        center_sigma=center_sigma,
        surround_sigma=surround_sigma,
        n_scales=draw(default_n_scales()),
        max_octaves=draw(default_max_octaves()),
        operating_limits=SafeOperatingLimits(),
    )

    return DepthSalienceSetup(
        processor=processor,
        image=image,
        box=box,
    )


class DepthSalienceTest(unittest.TestCase):
    MINIMUM_SALIENCE_THRESHOLD = 1e-4

    @settings(deadline=1000)
    @given(
        setup=depth_salience_setup(
            image=safe_solid_images(),
        )
    )
    def test_solid_image_not_salient(
        self,
        setup: DepthSalienceSetup,
    ) -> None:
        processor = setup.processor
        image = setup.image
        feature_map = processor.process(Mock(), image)
        self.assertTrue(np.all(feature_map < self.MINIMUM_SALIENCE_THRESHOLD))

    @settings(deadline=1000)
    @given(
        setup=depth_box_salience_setup(
            on_value=0.5,
            off_value=1.0,
        )
    )
    def test_box_is_more_salient_than_surround(
        self,
        setup: DepthSalienceSetup,
    ) -> None:
        processor = setup.processor
        image = setup.image
        feature_map = processor.process(Mock(), image)

        box = setup.box
        dilated = binary_dilation(box, iterations=5)
        surround = ~dilated
        surround = ~box

        box_salience = feature_map[box].mean()
        surround_salience = feature_map[surround].mean()
        self.assertTrue(box_salience > surround_salience)


# class ColorChannelSalienceWithoutOperatingLimitsTest(unittest.TestCase):
#     @given(
#         center_and_surround_sigmas=unsafe_center_and_surround_sigmas(),
#     )
#     def test_constructing_with_unsafe_sigmas_does_not_raise(
#         self,
#         center_and_surround_sigmas: tuple[float, float],
#     ) -> None:
#         center_sigma, surround_sigma = center_and_surround_sigmas
#         ColorChannelSalience.without_operating_limits(
#             center_sigma=center_sigma,
#             surround_sigma=surround_sigma,
#         )

#     @given(
#         image_and_processor=color_channel_salience_setup(
#             color_channel_salience_processor_without_operating_limits,
#             safe_filled_images(max_dim_size=SafeOperatingLimits.min_image_dim_size - 1),
#         ),
#         suppress_runtime_errors=st.booleans(),
#     )
#     def test_process_does_not_raise_value_error_if_image_has_smaller_dimension_than_min_safe_dim_size(  # noqa: E501
#         self,
#         image_and_processor: tuple[npt.NDArray[np.float32], ColorChannelSalience],
#         suppress_runtime_errors: bool,
#     ) -> None:
#         image, processor = image_and_processor
#         ctx = Mock(suppress_runtime_errors=suppress_runtime_errors)
#         processor.process(ctx, image)


def rectangular_mask(
    resolution: tuple[int, int],
    center: tuple[int, int],
    width: int,
    height: int,
) -> npt.NDArray[np.bool_]:
    y, x = center
    if width == 1 and height == 1:
        half_width = 1
        half_height = 1
    else:
        half_width = int(width // 2)
        half_height = int(height // 2)

    data = np.zeros(resolution, dtype=bool)
    y1 = max(y - half_height, 0)
    y2 = min(y + half_height, resolution[0])
    x1 = max(x - half_width, 0)
    x2 = min(x + half_width, resolution[1])
    data[y1:y2, x1:x2] = True
    return data
