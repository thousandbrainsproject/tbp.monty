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

from tbp.monty.frameworks.models.salience.strategies.vocus2.pyramids import Pyramid
from tbp.monty.frameworks.models.salience.strategies.vocus2.vocus2 import (
    ColorChannelSalience,
    DepthSalience,
    OrientationSalience,
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
    @given(setup=color_channel_salience_setup(image=safe_solid_images()))
    def test_solid_image_not_salient(
        self,
        setup: ColorChannelSalienceSetup,
    ) -> None:
        processor = setup.processor
        image = setup.image
        feature_map, _ = processor.process(Mock(), image)
        self.assertTrue(np.all(feature_map < self.MINIMUM_SALIENCE_THRESHOLD))

    @settings(deadline=1000)
    @given(setup=box_salience_setup(on_value=1.0, off_value=0.0))
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
    @given(setup=depth_salience_setup(image=safe_solid_images()))
    def test_solid_image_not_salient(
        self,
        setup: DepthSalienceSetup,
    ) -> None:
        processor = setup.processor
        image = setup.image
        feature_map = processor.process(Mock(), image)
        self.assertTrue(np.all(feature_map < self.MINIMUM_SALIENCE_THRESHOLD))

    @settings(deadline=1000)
    @given(setup=depth_box_salience_setup(on_value=0.5, off_value=1.0))
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


# --------------------------------------------------------------------------------------
# Orientation Salience
# --------------------------------------------------------------------------------------


@dataclass
class OrientationSalienceSetup:
    processor: OrientationSalience
    pyramid: Pyramid
    box: npt.NDArray[np.bool_] | None = None


@st.composite
def orientation_salience_setup(
    draw: st.DrawFn,
    image: st.SearchStrategy[npt.NDArray[np.float32]] | None = None,
) -> OrientationSalienceSetup:
    image = image or safe_images()
    _image = draw(image)
    center_sigma, surround_sigma = draw(safe_cs_sigmas(resolution=_image.shape))

    # Create the input pyramid
    color_channel_salience = ColorChannelSalience(
        center_sigma=center_sigma,
        surround_sigma=surround_sigma,
        n_scales=draw(default_n_scales()),
        max_octaves=draw(default_max_octaves(min_value=2)),
        operating_limits=SafeOperatingLimits(),
    )
    _, input_pyramid = color_channel_salience.process(Mock(), _image)

    return OrientationSalienceSetup(
        processor=OrientationSalience(period=2 * center_sigma),
        pyramid=input_pyramid,
        box=None,
    )


@st.composite
def orientation_box_salience_setup(
    draw: st.DrawFn,
    off_value: float = 0.0,
) -> DepthSalienceSetup:
    resolution = draw(safe_resolutions())
    cs_sigmas = safe_cs_sigmas(resolution=resolution)
    center_sigma, surround_sigma = draw(cs_sigmas)

    # Create image that has a box with a sinusoidal grating in it.
    min_box_width = round(2 * center_sigma)
    max_box_width = min(resolution) // 2
    box_width = draw(st.integers(min_value=min_box_width, max_value=max_box_width))
    box_height = box_width

    min_box_y = box_height // 2
    max_box_y = resolution[0] - box_height // 2
    box_y = draw(st.integers(min_value=min_box_y, max_value=max_box_y))

    min_box_x = box_width // 2
    max_box_x = resolution[1] - box_width // 2
    box_x = draw(st.integers(min_value=min_box_x, max_value=max_box_x))

    box_center = (box_y, box_x)

    box = rectangular_mask(
        resolution=resolution,
        center=box_center,
        width=box_width,
        height=box_height,
    )
    min_wavelength = 2 * center_sigma
    max_wavelength = box_width * 2
    grating_wavelength = draw(
        st.floats(min_value=min_wavelength, max_value=max_wavelength)
    )
    grating_angle = draw(st.floats(min_value=0.0, max_value=180.0))
    grating_image = generate_grating(
        resolution=resolution,
        wavelength=grating_wavelength,
        angle_degrees=grating_angle,
    )
    grating_image[~box] = off_value

    # Create the input pyramid for OrientationSalience
    color_channel_salience = ColorChannelSalience(
        center_sigma=center_sigma,
        surround_sigma=surround_sigma,
        n_scales=draw(default_n_scales()),
        max_octaves=draw(default_max_octaves(min_value=2)),
        operating_limits=SafeOperatingLimits(),
    )

    _, input_pyramid = color_channel_salience.process(Mock(), grating_image)

    processor = OrientationSalience(
        period=2 * center_sigma,
    )

    return OrientationSalienceSetup(
        processor=processor,
        pyramid=input_pyramid,
        box=box,
    )


class OrientationSalienceTest(unittest.TestCase):
    MINIMUM_SALIENCE_THRESHOLD = 1e-4

    @settings(deadline=1000)
    @given(setup=orientation_salience_setup(image=safe_solid_images()))
    def test_solid_image_not_salient(
        self,
        setup: OrientationSalienceSetup,
    ) -> None:
        processor = setup.processor
        pyramid = setup.pyramid
        feature_map = processor.process(Mock(), pyramid)
        self.assertTrue(np.all(feature_map < self.MINIMUM_SALIENCE_THRESHOLD))

    @settings(deadline=1000)
    @given(setup=orientation_box_salience_setup(off_value=0.0))
    def test_box_is_more_salient_than_surround(
        self,
        setup: OrientationSalienceSetup,
    ) -> None:
        processor = setup.processor
        pyramid = setup.pyramid
        feature_map = processor.process(Mock(), pyramid)

        box = setup.box
        dilated = binary_dilation(box, iterations=5)
        surround = ~dilated
        surround = ~box

        box_salience = feature_map[box].mean()
        surround_salience = feature_map[surround].mean()
        self.assertTrue(box_salience > surround_salience)



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


def generate_grating(
    resolution: tuple[int, int],
    wavelength: float,
    angle_degrees: float,
    phase: float = 0,
) -> npt.NDArray[np.uint8]:
    height, width = resolution
    # Convert angle to radians
    angle_rad = np.deg2rad(angle_degrees)

    # Create coordinate grid
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    # Calculate orientation components
    # Rotate the coordinates to orient the grating
    x_prime = x * np.cos(angle_rad) + y * np.sin(angle_rad)

    # Calculate the spatial frequency
    frequency = 2 * np.pi / wavelength

    # Evaluate the sinusoidal grating formula
    # Output values are in the range [-1, 1]
    grating = np.sin(frequency * x_prime + phase)

    # Put in [0, 1] range
    grating = (grating + 1) / 2

    return grating.astype(np.float32)
