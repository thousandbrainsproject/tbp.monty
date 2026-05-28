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
from tbp.monty.frameworks.models.salience.strategies.vocus2.images import (
    ColorSpaceConverter,
    rgb_to_opponent,
)
from tbp.monty.frameworks.models.salience.strategies.vocus2.pyramids import (
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
    @staticmethod
    def validate(
        min_image_dim_size: int,
        center_sigma: float,
        surround_sigma: float,
    ) -> ValueError | None: ...


class NoOperatingLimits(OperatingLimits):
    @staticmethod
    def validate(
        min_image_dim_size: int,  # noqa: ARG004
        center_sigma: float,  # noqa: ARG004
        surround_sigma: float,  # noqa: ARG004
    ) -> ValueError | None:
        return None


@dataclass(frozen=True)
class SafeOperatingLimits(OperatingLimits):
    """Within these limits, Vocus2 was tested and has desired and predictable behavior.

    Outside of these limits, Vocus2 may work as intended, may work not as expected,
    or may not work at all. Use with caution.

    The meaning of a smoothing kernel (sigma) size depends upon the resolution of the
    image. For example, a smoothing kernel size of three may be considered large for a
    64x64 image, but small for a 128x128 image. This matters because the smoothing
    kernel size is what determines the _scale_ of the feature that is extracted from
    the image.

    For this reason, there are no suitable minimum and maximum sigmas that are
    appropriate for all image resolutions. Instead, we express the sigma ranges as
    fractions of the smallest image dimension (hence "fractional_" throughout).
    For example, an image with resolution of 64x96, has a smallest image dimension of
    64. The maximum fractional center sigma of 0.1 would mean a 6.4 pixels (0.1 * 64)
    is the highest value that a center sigma can take and still be within safe operating
    limits.


    Attributes:
        min_image_dim_size: The minimum (smallest) dimension of the images
          being processed.
        max_fractional_center_sigma: The largest (fractional) center sigma that is
            still within safe operating limits.
        max_fractional_surround_sigma: The largest (fractional) surround sigma that is
            within safe operating limits.
        min_fractional_sigma_separation: We expect that the surround sigma will be
            greater than the center sigma. However, the surround sigma should be
            meaningfully bigger than the center sigma. This is the minimum meaningful
            separation between the center and surround sigmas. As explained above, this
            is expressed as a fraction of the smallest image dimension.
    """

    min_image_dim_size: int = 64
    max_fractional_center_sigma: float = 0.1
    max_fractional_surround_sigma: float = 0.5
    min_fractional_sigma_separation: float = 0.02

    @staticmethod
    def validate(
        min_image_dim_size: int,
        center_sigma: float,
        surround_sigma: float,
    ) -> ValueError | None:
        min_fractional_sigma = 1 / min_image_dim_size
        fractional_center_sigma = center_sigma / min_image_dim_size
        fractional_surround_sigma = surround_sigma / min_image_dim_size

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

        # Check center sigma is within the allowed range.
        center_min = min_image_dim_size * min_fractional_sigma
        center_max = (
            min_image_dim_size * SafeOperatingLimits.max_fractional_center_sigma
        )
        if not (center_min <= center_sigma <= center_max):
            return ValueError(
                f"When smallest image dimension is {min_image_dim_size}, "
                f"Center sigma must be greater than or equal to {center_min} "
                f"and less than or equal to {center_max}. You provided {center_sigma}."
            )

        # Check surround sigma is within the allowed range.
        surround_min = min_image_dim_size * (
            fractional_center_sigma
            + SafeOperatingLimits.min_fractional_sigma_separation
        )
        surround_max = (
            min_image_dim_size * SafeOperatingLimits.max_fractional_surround_sigma
        )
        if not (surround_min <= surround_sigma <= surround_max):
            return ValueError(
                f"When smallest image dimension is {min_image_dim_size}, "
                f"Surround sigma must be greater than or equal to {surround_min} "
                f"and less than or equal to {surround_max}. "
                f"You provided {surround_sigma}."
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
        kernels: dict[str, npt.NDArray[np.float32]] = {}
        filter_size = int(7 * sigma + 1) | 1
        for ori in range(n_orientations):
            theta = ori * np.pi / n_orientations
            kernel: npt.NDArray[np.float32] = cv2.getGaborKernel(
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
        feature_maps = FeatureMaps({})
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


class Normalize(Protocol):
    def __call__(
        self, salience_map: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]: ...


def no_normalize(salience_map: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return salience_map


def range_normalize(
    salience_map: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    salience_min = salience_map.min()
    salience_max = salience_map.max()
    scale = salience_max - salience_min
    if np.isclose(scale, 0):
        return np.clip(salience_map, 0, 1)
    return (salience_map - salience_min) / scale


class Vocus2(SalienceStrategy):
    def __init__(
        self,
        color: ColorChannelSalience,
        depth: DepthSalience | None = None,
        orientation: OrientationSalience | None = None,
        color_space_converter: ColorSpaceConverter = rgb_to_opponent,
        combine: MapCombine | None = None,
        normalize: Normalize = range_normalize,
    ):
        self._color = color
        self._depth = depth
        self._orientation = orientation
        self._color_space_converter = color_space_converter
        self._normalize = normalize

        if combine is None:
            self._combine: MapCombine = WeightedMean(
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
        normalize: Normalize = range_normalize,
    ) -> Vocus2:
        """Create a Vocus2 salience strategy from a configuration.

        Since Vocus2 uses color, depth, and orientation that all need to be configured
        in a compatible way, this method creates the necessary components and configures
        them all at once from a single configuration.

        Args:
            config: The configuration to use.
            color_space_converter: The color space converter to use.
            combine: The combine function to use.
            normalize: The normalization callable to use.

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
            normalize=normalize,
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
        salience_map = self._normalize(salience_map)
        return salience_map.astype(np.float64)
