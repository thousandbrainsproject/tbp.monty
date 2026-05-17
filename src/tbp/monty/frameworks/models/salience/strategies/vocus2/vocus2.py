# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, NewType, Protocol

import cv2
import numpy as np
import numpy.typing as npt

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.salience.strategies import SalienceStrategy

from .images import ColorSpaceConverter, rgb_to_opponent
from .pyramids import (
    Pyramid,
    PyramidCollapse,
    PyramidCombine,
    center_surround_pyramids,
    laplacian_pyramid,
    pyramid_collapse_mean,
    pyramid_combine_mean,
)

logger = logging.getLogger(__name__)

FeatureMaps = NewType("FeatureMaps", Dict[str, npt.NDArray[np.float32]])


class MapCombine(Protocol):
    def __call__(self, maps: FeatureMaps) -> npt.NDArray[np.float32]: ...


def map_max(maps: FeatureMaps) -> npt.NDArray[np.float32]:
    return np.max(list(maps.values()), axis=0)


def map_sum(maps: FeatureMaps) -> npt.NDArray[np.float32]:
    return np.sum(list(maps.values()), axis=0)


class WeightedSum(MapCombine):
    def __init__(self, weights: dict[str, float]):
        self._weights = weights

    def __call__(self, maps: FeatureMaps) -> npt.NDArray[np.float32]:
        return np.sum([self._weights[key] * img for key, img in maps.items()], axis=0)


def map_mean(maps: FeatureMaps) -> npt.NDArray[np.float32]:
    return np.mean(list(maps.values()), axis=0)


class WeightedMean(MapCombine):
    def __init__(self, weights: dict[str, float]):
        self._weights = weights

    def __call__(self, maps: FeatureMaps) -> npt.NDArray[np.float32]:
        weights = {key: self._weights[key] for key in maps}
        total_weight = sum(abs(weight) for weight in weights.values())
        normed_weights = {key: weight / total_weight for key, weight in weights.items()}
        return np.sum([normed_weights[key] * img for key, img in maps.items()], axis=0)


class OperatingLimits(Protocol):
    def validate_center_and_surround_sigma(
        self, center_sigma: float, surround_sigma: float
    ) -> ValueError | None: ...
    def validate_image_dim_size(self, image_dim_size: int) -> ValueError | None: ...

    @staticmethod
    def validate(
        min_image_dim_size: int,
        center_sigma: float,
        surround_sigma: float,
    ) -> ValueError | None: ...


class NoOperatingLimits(OperatingLimits):
    def validate_center_and_surround_sigma(
        self,
        center_sigma: float,  # noqa: ARG002
        surround_sigma: float,  # noqa: ARG002
    ) -> ValueError | None:
        return None

    def validate_image_dim_size(
        self,
        image_dim_size: int,  # noqa: ARG002
    ) -> ValueError | None:
        return None

    @staticmethod
    def validate(
        min_image_dim_size: int,  # noqa: ARG004
        center_sigma: float,  # noqa: ARG004
        surround_sigma: float,  # noqa: ARG004
    ) -> ValueError | None:
        return None


@dataclass(frozen=True)
class SafeOperatingLimits(OperatingLimits):
    min_center_sigma: float = 1.0
    max_center_sigma: float = 6.0
    max_surround_sigma: float = 12.0
    center_surround_sigma_ratio: float = 1.6

    min_image_dim_size: int = 64
    min_fractional_sigma_separation: float = 0.01
    max_fractional_sigma: float = 1.0

    @staticmethod
    def validate_center_and_surround_sigma(
        center_sigma: float,
        surround_sigma: float,
    ) -> ValueError | None:
        if center_sigma < SafeOperatingLimits.min_center_sigma:
            return ValueError(
                "Center sigma must be greater than or equal to "
                f"{SafeOperatingLimits.min_center_sigma}"
            )
        if center_sigma > SafeOperatingLimits.max_center_sigma:
            return ValueError(
                "Center sigma must be less than or equal to "
                f"{SafeOperatingLimits.max_center_sigma}"
            )

        if (
            surround_sigma
            < center_sigma * SafeOperatingLimits.center_surround_sigma_ratio
        ):
            return ValueError(
                "Surround sigma must be greater than or equal to "
                f"{center_sigma * SafeOperatingLimits.center_surround_sigma_ratio}"
            )
        if surround_sigma > SafeOperatingLimits.max_surround_sigma:
            return ValueError(
                "Surround sigma must be less than or equal to "
                f"{SafeOperatingLimits.max_surround_sigma}"
            )
        return None

    @staticmethod
    def validate_image_dim_size(
        image_dim_size: int,
    ) -> ValueError | None:
        if image_dim_size < SafeOperatingLimits.min_image_dim_size:
            return ValueError(
                "Image dimension size must be greater than or equal to "
                f"{SafeOperatingLimits.min_image_dim_size}"
            )
        return None

    @staticmethod
    def validate(
        min_image_dim_size: int,
        center_sigma: float,
        surround_sigma: float,
    ) -> ValueError | None:
        min_fractional_sigma = 1 / min_image_dim_size
        fractional_center_sigma = center_sigma / min_image_dim_size
        fractional_surround_sigma = surround_sigma / min_image_dim_size

        # Check surround < center
        if fractional_surround_sigma <= fractional_center_sigma:
            return ValueError("Surround sigma must be greater than center sigma")

        # Check surround >= center + buffer
        if (
            fractional_surround_sigma
            < fractional_center_sigma
            + SafeOperatingLimits.min_fractional_sigma_separation
        ):
            return ValueError(
                "Surround sigma must be greater than or equal to center_sigma + "
                "min_fractional_sigma_separation."
            )

        # Check neither sigmas are outside of the allowed range.
        sigma_info = [
            (fractional_center_sigma, "Center"),
            (fractional_surround_sigma, "Surround"),
        ]
        for fractional_sigma, sigma_name in sigma_info:
            if fractional_sigma < min_fractional_sigma:
                return ValueError(
                    f"{sigma_name} sigma must be greater than or equal to "
                    f"min fractional sigma ({min_fractional_sigma})"
                )
            if fractional_sigma > SafeOperatingLimits.max_fractional_sigma:
                return ValueError(
                    f"{sigma_name} sigma must be less than or equal to "
                    f"max fractional sigma ({SafeOperatingLimits.max_fractional_sigma})"
                )

        return None


class ColorChannelSalience:
    def __init__(
        self,
        center_sigma: float = 2.0,
        surround_sigma: float = 3.0,
        n_scales: int = 2,
        max_octaves: int | None = None,
        combine: PyramidCombine = pyramid_combine_mean,
        collapse: PyramidCollapse = pyramid_collapse_mean,
        operating_limits: OperatingLimits | None = None,
    ):
        """Create a `ColorChannelSalience` with safe operating limits.

        `ColorChannelSalience` was designed and tested to be used within provided safe
        operating limits. Safe operating limits will raise a `ValueError` if the
        parameters are outside of the safe operating limits. Some are checked at
        construction time, others are checked at runtime and subject to
        `RuntimeContext.suppress_runtime_errors`. To opt-out of using safe operating
        limits, use the `without_operating_limits` class method instead of constructing
        directly.

        Args:
            center_sigma: The center sigma for the center/surround pyramids.
            surround_sigma: The surround sigma for the center/surround pyramids.
            n_scales: The number of pyramid scales.
            max_octaves: The maximum number of pyramid octaves to construct.
            combine: The function to combine the on/off pyramids into a single pyramid.
            collapse: The function to collapse the combined pyramid into a single image.
            operating_limits: The operating limits to use.
        """
        self._center_sigma = center_sigma
        self._surround_sigma = surround_sigma
        self._n_scales = n_scales
        self._max_octaves = max_octaves
        self._combine = combine
        self._collapse = collapse
        self._operating_limits = (
            operating_limits if operating_limits is not None else SafeOperatingLimits()
        )

    def process(
        self, ctx: RuntimeContext, image: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], Pyramid]:
        """Compute salience for a single color channel.

        Args:
            ctx: The runtime context.
            image: Must be float32 and in the range [0, 1].

        Returns:
            A tuple of the feature map and the center pyramid.

        Raises:
            ValueError: If operating limits reject the image size.
        """
        error = self._operating_limits.validate(
            min(image.shape),
            self._center_sigma,
            self._surround_sigma,
        )
        if error is not None:
            if ctx.suppress_runtime_errors:
                logger.warning(str(error))
            else:
                raise ValueError(str(error))

        center, surround = center_surround_pyramids(
            image,
            center_sigma=self._center_sigma,
            surround_sigma=self._surround_sigma,
            n_scales=self._n_scales,
            max_octaves=self._max_octaves,
        )

        diff: Pyramid = center - surround
        on: Pyramid = diff.apply(lambda img: np.maximum(img, 0))
        off: Pyramid = diff.apply(lambda img: np.maximum(-img, 0))

        feature_pyramid = self._combine([on, off])
        feature_map = self._collapse(feature_pyramid)

        return feature_map, center


class DepthSalience:
    def __init__(
        self,
        center_sigma: float = 2.0,
        surround_sigma: float = 3.0,
        n_scales: int = 2,
        max_octaves: int | None = None,
        collapse: PyramidCollapse = pyramid_collapse_mean,
        operating_limits: OperatingLimits | None = None,
    ):
        self._center_sigma = center_sigma
        self._surround_sigma = surround_sigma
        self._n_scales = n_scales
        self._max_octaves = max_octaves
        self._collapse = collapse
        self._operating_limits = (
            operating_limits if operating_limits is not None else SafeOperatingLimits()
        )

    def process(
        self, ctx: RuntimeContext, image: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Compute salience for a depth channel.

        Args:
            ctx: The runtime context.
            image: Must be float32.

        Returns:
            A DepthSalienceResult object.

        Raises:
            ValueError: If operating limits reject the image size.
        """
        error = self._operating_limits.validate(
            min(image.shape),
            self._center_sigma,
            self._surround_sigma,
        )
        if error is not None:
            if ctx.suppress_runtime_errors:
                logger.warning(str(error))
            else:
                raise ValueError(str(error))

        image = -np.log(image).astype(np.float32)
        image = np.nan_to_num(image, posinf=0.0)

        center, surround = center_surround_pyramids(
            image,
            center_sigma=self._center_sigma,
            surround_sigma=self._surround_sigma,
            n_scales=self._n_scales,
            max_octaves=self._max_octaves,
        )

        diff: Pyramid = center - surround
        feature_pyramid = diff.apply(lambda img: np.maximum(img, 0))
        return self._collapse(feature_pyramid)


class OrientationSalience:
    def __init__(
        self,
        period: float,
        sigma: float | None = None,
        phase: float = np.pi / 2,
        gamma: float = 0.75,
        n_orientations: int = 4,
        combine: MapCombine = map_mean,
        collapse: PyramidCollapse = pyramid_collapse_mean,
    ):
        """Computes orientation salience.

        Args:
            period: wavelength. Good default is center_sigma * 2
            sigma: mask sigma. Good default is 0.3 * period
            phase: phase. Good default is 90 degrees (pi / 2) for edge detection,
                0 for strip detection.
            gamma: Eccentricity. Good default is 0.75
            n_orientations: number of orientations. Good default is 4
            combine: function to combine the feature maps. Good default is `map_mean`.
            collapse: function to collapse the feature pyramids. Good default is
                `pyramid_collapse_mean`.

        """
        self._period = period
        self._sigma = 0.3 * self._period if sigma is None else sigma
        self._phase = phase
        self._gamma = gamma
        self._n_orientations = n_orientations
        self._combine = combine
        self._collapse = collapse

        self._kernels = self.make_kernels(
            period=self._period,
            sigma=self._sigma,
            phase=self._phase,
            gamma=self._gamma,
            n_orientations=self._n_orientations,
        )

    @staticmethod
    def make_kernels(
        period: float,
        sigma: float,
        phase: float = np.pi / 2,
        gamma: float = 0.75,
        n_orientations: int = 4,
    ) -> dict[str, npt.NDArray[np.float32]]:
        kernels = {}
        filter_size = int(7 * sigma + 1) | 1
        for ori in range(n_orientations):
            theta = ori * np.pi / n_orientations
            kernel = cv2.getGaborKernel(
                (filter_size, filter_size),
                sigma=sigma,
                theta=theta,
                lambd=period,
                gamma=gamma,
                psi=phase,
                ktype=cv2.CV_32F,
            )
            kernel = kernel - np.mean(kernel)  # balance excitation and suppression
            kernels[f"orientation_{ori}"] = kernel

        return kernels

    def process(
        self,
        ctx: RuntimeContext,  # noqa: ARG002
        pyr: Pyramid,
    ) -> npt.NDArray[np.float32]:
        feature_pyramids = {}
        feature_maps = {}
        lap = laplacian_pyramid(pyr)
        for ori, kernel in self._kernels.items():
            p = np.zeros(lap.shape, dtype=object)
            for level in range(lap.shape[0]):
                for scale in range(lap.shape[1]):
                    amt = cv2.filter2D(lap.data[level, scale], cv2.CV_32F, kernel)
                    p[level, scale] = np.abs(amt)
            feature_pyramids[ori] = Pyramid(p)
            feature_maps[ori] = self._collapse(feature_pyramids[ori])

        return self._combine(feature_maps)


@dataclass
class Vocus2SalienceConfig:
    center_sigma: float = 3.0
    surround_sigma: float = 5.0
    n_scales: int = 2
    max_octaves: int = 5
    use_depth: bool = True
    use_orientation: bool = True


class Vocus2(SalienceStrategy):
    def __init__(
        self,
        color: ColorChannelSalience,
        depth: DepthSalience | None = None,
        orientation: OrientationSalience | None = None,
        color_space_converter: ColorSpaceConverter = rgb_to_opponent,
        combine: MapCombine | None = None,
        normalize: bool = True,
    ):
        self._color = color
        self._depth = depth
        self._orientation = orientation
        self._color_space_converter = color_space_converter
        self._normalize = normalize

        if combine is None:
            self._combine = WeightedMean(
                {
                    "L": 1,
                    "a": 1,
                    "b": 1,
                    "depth": 0.1,
                    "orientation": 1,
                }
            )
        else:
            self._combine = combine

    @classmethod
    def from_config(
        cls,
        config: Vocus2SalienceConfig,
        color_space_converter: ColorSpaceConverter = rgb_to_opponent,
        combine: MapCombine | None = None,
    ) -> Vocus2:
        """Create a Vocus2 salience strategy from a configuration.

        Since Vocus2 uses color, depth, and orientation that all need to be configured
        in a compatible way, this method creates the necessary components and configures
        them all at once from a single configuration.

        Args:
            config: The configuration to use.
            color_space_converter: The color space converter to use.
            combine: The combine function to use.

        Returns:
            A Vocus2 salience strategy.
        """
        color = ColorChannelSalience(
            center_sigma=config.center_sigma,
            surround_sigma=config.surround_sigma,
            n_scales=config.n_scales,
            max_octaves=config.max_octaves,
        )

        depth = (
            DepthSalience(
                center_sigma=config.center_sigma,
                surround_sigma=config.surround_sigma,
                n_scales=config.n_scales,
                max_octaves=config.max_octaves,
            )
            if config.use_depth
            else None
        )

        orientation = (
            OrientationSalience(
                period=2 * config.center_sigma,
            )
            if config.use_orientation
            else None
        )

        return cls(
            color=color,
            depth=depth,
            orientation=orientation,
            color_space_converter=color_space_converter,
            combine=combine,
        )

    def __call__(
        self,
        ctx: RuntimeContext,
        rgba: npt.NDArray[np.int_],
        depth: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        # Get color and depth data into open-cv compatible formats.
        rgb = rgba[:, :, :3]
        depth = depth.astype(np.float32)

        feature_maps = FeatureMaps({})

        Lab = self._color_space_converter(rgb)  # noqa: N806
        L, a, b = cv2.split(Lab)  # noqa: N806
        feature_maps["L"], L_center = self._color.process(ctx, L)  # noqa: N806
        feature_maps["a"], _ = self._color.process(ctx, a)
        feature_maps["b"], _ = self._color.process(ctx, b)

        if self._depth:
            feature_maps["depth"] = self._depth.process(ctx, depth)

        if self._orientation:
            feature_maps["orientation"] = self._orientation.process(ctx, L_center)

        salience_map = self._combine(feature_maps)

        if self._normalize:
            salience_min = salience_map.min()
            salience_max = salience_map.max()
            scale = salience_max - salience_min
            if np.isclose(scale, 0):
                salience_map = np.clip(salience_map, 0, 1)
            else:
                salience_map = (salience_map - salience_min) / scale

        return salience_map.astype(np.float64)
