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
from unittest.mock import ANY, Mock, call, patch, sentinel

import numpy as np
import numpy.typing as npt
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.ndimage import binary_dilation

from tbp.monty.frameworks.models.salience.strategies.vocus2.images import (
    rgb_to_opponent,
)
from tbp.monty.frameworks.models.salience.strategies.vocus2.pyramids import Pyramid
from tbp.monty.frameworks.models.salience.strategies.vocus2.vocus2 import (
    ColorChannelSalience,
    DepthSalience,
    OrientationSalience,
    SafeOperatingLimits,
    Vocus2,
    Vocus2SalienceConfig,
    range_normalize,
)
from tests.unit.frameworks.models.salience.strategies.vocus2.pyramids_test import (
    MAX_DIM_SIZE,
    default_image_values,
    default_images,
    default_max_octaves,
    default_n_scales,
)


@st.composite
def safe_resolutions(draw: st.DrawFn) -> tuple[int, int]:
    height = draw(
        st.integers(
            min_value=SafeOperatingLimits.min_image_dim_size, max_value=MAX_DIM_SIZE
        )
    )
    width = draw(
        st.integers(
            min_value=SafeOperatingLimits.min_image_dim_size, max_value=MAX_DIM_SIZE
        )
    )
    return (height, width)


@st.composite
def unsafe_resolutions(draw: st.DrawFn) -> tuple[int, int]:
    height = draw(
        st.integers(
            min_value=1,
            max_value=SafeOperatingLimits.min_image_dim_size,
            exclude_max=True,
        )
    )
    width = draw(
        st.integers(
            min_value=1,
            max_value=SafeOperatingLimits.min_image_dim_size,
            exclude_max=True,
        )
    )
    return (height, width)


def sigma_strategy_limits(
    resolution: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Returns limits for generating center and surround sigmas for a given resolution.

    Args:
        resolution: The resolution of the image to generate sigmas for.

    Returns:
        Limits for generating center and surround sigmas.
    """
    min_dim_size = min(resolution)
    min_center_sigma = 1.0
    max_center_sigma = min_dim_size * SafeOperatingLimits.max_fractional_center_sigma
    min_sigma_separation = (
        min_dim_size * SafeOperatingLimits.min_fractional_sigma_separation
    )
    max_surround_sigma = (
        min_dim_size * SafeOperatingLimits.max_fractional_surround_sigma
    )
    return (
        min_center_sigma,
        max_center_sigma,
        min_sigma_separation,
        max_surround_sigma,
    )


@st.composite
def safe_cs_sigmas(
    draw: st.DrawFn,
    resolution: tuple[int, int],
) -> tuple[float, float]:
    min_center_sigma, max_center_sigma, min_sigma_separation, max_surround_sigma = (
        sigma_strategy_limits(resolution)
    )

    center_sigma = draw(
        st.floats(min_value=min_center_sigma, max_value=max_center_sigma)
    )
    surround_sigma = draw(
        st.floats(
            min_value=center_sigma + min_sigma_separation,
            max_value=max_surround_sigma,
        )
    )
    return (center_sigma, surround_sigma)


@st.composite
def unsafe_cs_sigmas(
    draw: st.DrawFn,
    resolution: tuple[int, int],
) -> tuple[float, float]:
    min_center_sigma, max_center_sigma, min_sigma_separation, max_surround_sigma = (
        sigma_strategy_limits(resolution)
    )

    center_sigma_too_low = st.floats(
        min_value=0,
        max_value=min_center_sigma,
        exclude_max=True,
    )
    center_sigma_too_high = st.floats(
        min_value=max_center_sigma,
        max_value=max_surround_sigma,
        exclude_min=True,
    )
    center_sigma = draw(st.one_of(center_sigma_too_low, center_sigma_too_high))

    surround_sigma_too_low = st.floats(
        min_value=0,
        max_value=center_sigma + min_sigma_separation,
        exclude_max=True,
    )
    surround_sigma_too_high = st.floats(
        min_value=max_surround_sigma,
        max_value=2 * max_surround_sigma,
        exclude_min=True,
    )
    surround_sigma = draw(st.one_of(surround_sigma_too_low, surround_sigma_too_high))

    return (center_sigma, surround_sigma)


@st.composite
def safe_images(draw: st.DrawFn) -> npt.NDArray[np.float32]:
    return draw(default_images(resolution=safe_resolutions()))


@st.composite
def safe_solid_images(draw: st.DrawFn) -> npt.NDArray[np.float32]:
    return np.full(
        draw(safe_resolutions()),
        draw(default_image_values()),
        dtype=np.float32,
    )


@dataclass
class ColorChannelSalienceSetup:
    processor: ColorChannelSalience
    image: npt.NDArray[np.float32]
    box: npt.NDArray[np.bool_] | None = None


@st.composite
def color_channel_salience_setup(
    draw: st.DrawFn,
    image: st.SearchStrategy[npt.NDArray[np.float32]],
) -> ColorChannelSalienceSetup:
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

@dataclass
class BoxSetup:
    center_sigma: float
    surround_sigma: float
    image: npt.NDArray[np.float32]
    box: npt.NDArray[np.bool_]


@st.composite
def box_setup(
    draw: st.DrawFn,
    on_value: float = 1.0,
    off_value: float = 0.0,
) -> BoxSetup:
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

    return BoxSetup(
        center_sigma=center_sigma,
        surround_sigma=surround_sigma,
        image=image,
        box=box,
    )


@st.composite
def box_salience_setup(
    draw: st.DrawFn,
    on_value: float = 1.0,
    off_value: float = 0.0,
) -> ColorChannelSalienceSetup:
    _box_setup = draw(box_setup(on_value=on_value, off_value=off_value))

    processor = ColorChannelSalience(
        center_sigma=_box_setup.center_sigma,
        surround_sigma=_box_setup.surround_sigma,
        n_scales=draw(default_n_scales()),
        max_octaves=draw(default_max_octaves()),
        operating_limits=SafeOperatingLimits(),
    )

    return ColorChannelSalienceSetup(
        processor=processor,
        image=_box_setup.image,
        box=_box_setup.box,
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

    @given(image=default_images())
    def test_sigmas_outside_of_operating_limits_raises_value_error_if_suppress_runtime_errors_is_false(  # noqa: E501
        self,
        image: npt.NDArray[np.float32],
    ) -> None:
        operating_limits = Mock()
        operating_limits.validate.return_value = ValueError
        color = ColorChannelSalience(operating_limits=operating_limits)
        ctx = Mock(suppress_runtime_errors=False)

        with self.assertRaises(ValueError):
            color.process(ctx, image)

    @settings(deadline=1000)
    @given(image=default_images())
    def test_sigmas_outside_of_operating_limits_does_not_raise_value_error_if_suppress_runtime_errors_is_true(  # noqa: E501
        self,
        image: npt.NDArray[np.float32],
    ) -> None:
        operating_limits = Mock()
        operating_limits.validate.return_value = ValueError
        color = ColorChannelSalience(operating_limits=operating_limits)
        ctx = Mock(suppress_runtime_errors=True)

        color.process(ctx, image)


@dataclass
class DepthSalienceSetup:
    processor: DepthSalience
    image: npt.NDArray[np.float32]
    box: npt.NDArray[np.bool_] | None = None


@st.composite
def depth_salience_setup(
    draw: st.DrawFn,
    image: st.SearchStrategy[npt.NDArray[np.float32]],
) -> DepthSalienceSetup:
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
    _box_setup = draw(box_setup(on_value=on_value, off_value=off_value))

    processor = DepthSalience(
        center_sigma=_box_setup.center_sigma,
        surround_sigma=_box_setup.surround_sigma,
        n_scales=draw(default_n_scales()),
        max_octaves=draw(default_max_octaves()),
        operating_limits=SafeOperatingLimits(),
    )

    return DepthSalienceSetup(
        processor=processor,
        image=_box_setup.image,
        box=_box_setup.box,
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

    @given(image=default_images())
    def test_sigmas_outside_of_operating_limits_raises_value_error_if_suppress_runtime_errors_is_false(  # noqa: E501
        self,
        image: npt.NDArray[np.float32],
    ) -> None:
        operating_limits = Mock()
        operating_limits.validate.return_value = ValueError
        depth = DepthSalience(operating_limits=operating_limits)
        ctx = Mock(suppress_runtime_errors=False)

        with self.assertRaises(ValueError):
            depth.process(ctx, image)

    @settings(deadline=1000)
    @given(image=default_images())
    def test_sigmas_outside_of_operating_limits_does_not_raise_value_error_if_suppress_runtime_errors_is_true(  # noqa: E501
        self,
        image: npt.NDArray[np.float32],
    ) -> None:
        operating_limits = Mock()
        operating_limits.validate.return_value = ValueError
        depth = DepthSalience(operating_limits=operating_limits)
        ctx = Mock(suppress_runtime_errors=True)

        depth.process(ctx, image)


@dataclass
class OrientationSalienceSetup:
    processor: OrientationSalience
    pyramid: Pyramid
    box: npt.NDArray[np.bool_] | None = None


@st.composite
def orientation_salience_setup(
    draw: st.DrawFn,
    image: st.SearchStrategy[npt.NDArray[np.float32]],
) -> OrientationSalienceSetup:
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
def orientation_box_salience_setup(draw: st.DrawFn) -> DepthSalienceSetup:
    resolution = draw(safe_resolutions())
    cs_sigmas = safe_cs_sigmas(resolution=resolution)
    center_sigma, surround_sigma = draw(cs_sigmas)

    # Select location and size of the box that will contain the sinusoidal grating.
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

    # Create sinusoisal grating image.
    min_wavelength = 2 * center_sigma
    max_wavelength = box_width * 2
    wavelength = draw(st.floats(min_value=min_wavelength, max_value=max_wavelength))
    angle = draw(st.floats(min_value=0.0, max_value=np.pi))
    phase = draw(st.floats(min_value=0.0, max_value=2 * np.pi))

    y, x = np.meshgrid(
        np.arange(resolution[0]), np.arange(resolution[1]), indexing="ij"
    )
    x_prime = x * np.cos(angle) + y * np.sin(angle)
    frequency = 2 * np.pi / wavelength
    image = (np.sin(frequency * x_prime + phase) + 1) / 2

    # Set the area outside the box to solid black.
    image[~box] = 0.0
    image = image.astype(np.float32)

    # Generate the input pyramid for the OrientationSalience processor.
    color_channel_salience = ColorChannelSalience(
        center_sigma=center_sigma,
        surround_sigma=surround_sigma,
        n_scales=draw(default_n_scales()),
        max_octaves=draw(default_max_octaves(min_value=2)),
        operating_limits=SafeOperatingLimits(),
    )
    _, input_pyramid = color_channel_salience.process(Mock(), image)

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
    @given(setup=orientation_box_salience_setup())
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


class Vocus2FromConfigTest(unittest.TestCase):
    @patch(
        "tbp.monty.frameworks.models.salience.strategies.vocus2.vocus2.ColorChannelSalience",
        return_value=Mock(),
    )
    @patch(
        "tbp.monty.frameworks.models.salience.strategies.vocus2.vocus2.DepthSalience",
        return_value=Mock(),
    )
    @patch(
        "tbp.monty.frameworks.models.salience.strategies.vocus2.vocus2.OrientationSalience",
        return_value=Mock(),
    )
    @patch(
        "tbp.monty.frameworks.models.salience.strategies.vocus2.Vocus2",
        return_value=Mock(),
    )
    def test_sigmas_scales_and_octaves_are_coupled_across_color_depth_and_orientation(
        self,
        vocus2_mock: Mock,
        orientation_salience_mock: Mock,
        depth_salience_mock: Mock,
        color_channel_salience_mock: Mock,
    ) -> None:
        sentinel_center_sigma = 1.0
        config = Vocus2SalienceConfig(
            center_sigma=sentinel_center_sigma,
            surround_sigma=sentinel.surround_sigma,
            n_scales=sentinel.n_scales,
            max_octaves=sentinel.max_octaves,
        )
        color_channel_salience_mock.return_value = sentinel.color
        depth_salience_mock.return_value = sentinel.depth
        orientation_salience_mock.return_value = sentinel.orientation

        Vocus2.from_config.__func__(vocus2_mock, config)

        color_channel_salience_mock.assert_called_once_with(
            center_sigma=sentinel_center_sigma,
            surround_sigma=sentinel.surround_sigma,
            n_scales=sentinel.n_scales,
            max_octaves=sentinel.max_octaves,
        )
        depth_salience_mock.assert_called_once_with(
            center_sigma=sentinel_center_sigma,
            surround_sigma=sentinel.surround_sigma,
            n_scales=sentinel.n_scales,
            max_octaves=sentinel.max_octaves,
        )
        orientation_salience_mock.assert_called_once_with(
            period=2 * sentinel_center_sigma,
        )
        vocus2_mock.assert_called_once_with(
            color=sentinel.color,
            depth=sentinel.depth,
            orientation=sentinel.orientation,
            color_space_converter=rgb_to_opponent,
            combine=None,
            normalize=range_normalize,
        )

    @patch(
        "tbp.monty.frameworks.models.salience.strategies.vocus2.vocus2.ColorChannelSalience",
        return_value=Mock(),
    )
    @patch(
        "tbp.monty.frameworks.models.salience.strategies.vocus2.vocus2.DepthSalience",
        return_value=Mock(),
    )
    @patch(
        "tbp.monty.frameworks.models.salience.strategies.vocus2.vocus2.OrientationSalience",
        return_value=Mock(),
    )
    @patch(
        "tbp.monty.frameworks.models.salience.strategies.vocus2.Vocus2",
        return_value=Mock(),
    )
    def test_no_depth_if_use_depth_is_false(
        self,
        vocus2_mock: Mock,
        orientation_salience_mock: Mock,
        depth_salience_mock: Mock,
        color_channel_salience_mock: Mock,
    ) -> None:
        sentinel_center_sigma = 1.0
        config = Vocus2SalienceConfig(
            center_sigma=sentinel_center_sigma,
            surround_sigma=sentinel.surround_sigma,
            n_scales=sentinel.n_scales,
            max_octaves=sentinel.max_octaves,
            use_depth=False,
        )
        color_channel_salience_mock.return_value = sentinel.color
        depth_salience_mock.return_value = sentinel.depth
        orientation_salience_mock.return_value = sentinel.orientation

        Vocus2.from_config.__func__(vocus2_mock, config)

        depth_salience_mock.assert_not_called()
        vocus2_mock.assert_called_once_with(
            color=sentinel.color,
            depth=None,
            orientation=sentinel.orientation,
            color_space_converter=rgb_to_opponent,
            combine=None,
            normalize=range_normalize,
        )

    @patch(
        "tbp.monty.frameworks.models.salience.strategies.vocus2.vocus2.ColorChannelSalience",
        return_value=Mock(),
    )
    @patch(
        "tbp.monty.frameworks.models.salience.strategies.vocus2.vocus2.DepthSalience",
        return_value=Mock(),
    )
    @patch(
        "tbp.monty.frameworks.models.salience.strategies.vocus2.vocus2.OrientationSalience",
        return_value=Mock(),
    )
    @patch(
        "tbp.monty.frameworks.models.salience.strategies.vocus2.Vocus2",
        return_value=Mock(),
    )
    def test_no_orientation_if_use_orientation_is_false(
        self,
        vocus2_mock: Mock,
        orientation_salience_mock: Mock,
        depth_salience_mock: Mock,
        color_channel_salience_mock: Mock,
    ) -> None:
        sentinel_center_sigma = 1.0
        config = Vocus2SalienceConfig(
            center_sigma=sentinel_center_sigma,
            surround_sigma=sentinel.surround_sigma,
            n_scales=sentinel.n_scales,
            max_octaves=sentinel.max_octaves,
            use_orientation=False,
        )
        color_channel_salience_mock.return_value = sentinel.color
        depth_salience_mock.return_value = sentinel.depth
        orientation_salience_mock.return_value = sentinel.orientation

        Vocus2.from_config.__func__(vocus2_mock, config)

        orientation_salience_mock.assert_not_called()
        vocus2_mock.assert_called_once_with(
            color=sentinel.color,
            depth=sentinel.depth,
            orientation=None,
            color_space_converter=rgb_to_opponent,
            combine=None,
            normalize=range_normalize,
        )

    @patch(
        "tbp.monty.frameworks.models.salience.strategies.vocus2.Vocus2",
        return_value=Mock(),
    )
    def test_color_space_converter_combine_and_normalize_are_passed_to_constructor(
        self,
        vocus2_mock: Mock,
    ) -> None:
        config = Vocus2SalienceConfig()

        Vocus2.from_config.__func__(
            vocus2_mock,
            config,
            color_space_converter=sentinel.color_space_converter,
            combine=sentinel.combine,
            normalize=sentinel.normalize,
        )
        vocus2_mock.assert_called_once_with(
            color=ANY,
            depth=ANY,
            orientation=ANY,
            color_space_converter=sentinel.color_space_converter,
            combine=sentinel.combine,
            normalize=sentinel.normalize,
        )


class Vocus2Test(unittest.TestCase):
    def test_color_space_converts_rgba(self) -> None:
        color_mock = Mock()
        color_mock.process.return_value = (Mock(), Mock())
        Lab = np.array([[[1.0, 0, 0]]], dtype=np.float32)  # noqa: N806
        color_space_converter_mock = Mock()
        color_space_converter_mock.return_value = Lab

        vocus2 = Vocus2(
            color=color_mock,
            color_space_converter=color_space_converter_mock,
            combine=Mock(),
            normalize=Mock(),
        )

        rgba = np.array([[[255, 0, 0, 255]]], dtype=np.uint8)

        vocus2(Mock(), rgba, Mock())

        color_space_converter_mock.assert_called_once()
        call_args = color_space_converter_mock.call_args
        self.assertTrue(np.allclose(call_args[0][0], rgba[:, :, :3]))

    @patch("cv2.split", return_value=Mock())
    def test_creates_Lab_feature_maps(self, split_mock: Mock) -> None:  # noqa: N802
        split_mock.return_value = (sentinel.L, sentinel.a, sentinel.b)
        color_mock = Mock()
        color_mock.process.side_effect = [
            (sentinel.L, sentinel.L_center),
            (sentinel.a, Mock()),
            (sentinel.b, Mock()),
        ]
        color_space_converter_mock = Mock()
        combine_mock = Mock()

        vocus2 = Vocus2(
            color=color_mock,
            color_space_converter=color_space_converter_mock,
            combine=combine_mock,
            normalize=Mock(),
        )
        rgba = np.array([[[255, 0, 0, 255]]], dtype=np.uint8)
        ctx = Mock()

        vocus2(ctx, rgba, Mock())

        color_mock.process.assert_has_calls(
            [
                call(ctx, sentinel.L),
                call(ctx, sentinel.a),
                call(ctx, sentinel.b),
            ]
        )
        combine_mock.assert_called_once_with(
            {
                "L": sentinel.L,
                "a": sentinel.a,
                "b": sentinel.b,
            }
        )

    def test_creates_depth_feature_map_if_depth_processor_is_provided(self) -> None:
        color_mock = Mock()
        color_mock.process.return_value = (Mock(), Mock())
        combine_mock = Mock()
        depth_mock = Mock()
        depth_mock.process.return_value = sentinel.depth_feature_map

        vocus2 = Vocus2(
            color=color_mock,
            depth=depth_mock,
            combine=combine_mock,
            normalize=Mock(),
        )
        rgba = np.array([[[255, 0, 0, 255]]], dtype=np.uint8)
        depth = Mock()
        depth.astype.return_value = sentinel.depth
        ctx = Mock()

        vocus2(ctx, rgba, depth)

        depth_mock.process.assert_called_once_with(ctx, sentinel.depth)
        combine_mock.assert_called_once_with(
            {
                "L": ANY,
                "a": ANY,
                "b": ANY,
                "depth": sentinel.depth_feature_map,
            }
        )

    def test_creates_orientation_feature_map_if_orientation_processor_is_provided(
        self,
    ) -> None:
        color_mock = Mock()
        color_mock.process.side_effect = [
            (Mock(), sentinel.L_center),
            (Mock(), Mock()),
            (Mock(), Mock()),
        ]
        combine_mock = Mock()
        orientation_mock = Mock()
        orientation_mock.process.return_value = sentinel.orientation_feature_map

        vocus2 = Vocus2(
            color=color_mock,
            orientation=orientation_mock,
            combine=combine_mock,
            normalize=Mock(),
        )
        rgba = np.array([[[255, 0, 0, 255]]], dtype=np.uint8)
        ctx = Mock()

        vocus2(ctx, rgba, Mock())

        orientation_mock.process.assert_called_once_with(ctx, sentinel.L_center)
        combine_mock.assert_called_once_with(
            {
                "L": ANY,
                "a": ANY,
                "b": ANY,
                "orientation": sentinel.orientation_feature_map,
            }
        )

    def test_combines_feature_maps(self) -> None:
        color_mock = Mock()
        color_mock.process.side_effect = [
            (sentinel.L, Mock()),
            (sentinel.a, Mock()),
            (sentinel.b, Mock()),
        ]
        depth_mock = Mock()
        depth_mock.process.return_value = sentinel.depth_feature_map
        orientation_mock = Mock()
        orientation_mock.process.return_value = sentinel.orientation_feature_map
        combine_mock = Mock()

        vocus2 = Vocus2(
            color=color_mock,
            depth=depth_mock,
            orientation=orientation_mock,
            combine=combine_mock,
            normalize=Mock(),
        )
        rgba = np.array([[[255, 0, 0, 255]]], dtype=np.uint8)
        ctx = Mock()

        vocus2(ctx, rgba, Mock())

        combine_mock.assert_called_once_with(
            {
                "L": sentinel.L,
                "a": sentinel.a,
                "b": sentinel.b,
                "depth": sentinel.depth_feature_map,
                "orientation": sentinel.orientation_feature_map,
            }
        )

    def test_normalizes_salience_map(self) -> None:
        color_mock = Mock()
        color_mock.process.return_value = (Mock(), Mock())
        combine_mock = Mock()
        combine_mock.return_value = sentinel.combined
        normalize_mock = Mock()
        normalize_return_mock = Mock()
        normalize_return_mock.astype.return_value = sentinel.normalized
        normalize_mock.return_value = normalize_return_mock
        vocus2 = Vocus2(
            color=color_mock,
            combine=combine_mock,
            normalize=normalize_mock,
        )
        rgba = np.array([[[255, 0, 0, 255]]], dtype=np.uint8)

        result = vocus2(Mock(), rgba, Mock())

        normalize_mock.assert_called_once_with(sentinel.combined)
        self.assertIs(result, sentinel.normalized)
