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
from typing import Callable, Dict, Iterator, NewType, Protocol, Sequence, cast

import cv2
import numpy as np
import numpy.typing as npt

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.salience.strategies import SalienceStrategy
from tbp.monty.frameworks.sensors import Resolution2D

logger = logging.getLogger(__name__)

"""
- Color
-------------------------------------------------------------------------------
"""


class ColorSpaceConverter(Protocol):
    def __call__(self, image: npt.NDArray[np.int_]) -> npt.NDArray[np.float32]: ...


def rgb_to_lab(image: npt.NDArray[np.int_]) -> npt.NDArray[np.float32]:
    """Returns the CIE Lab color space of the image."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2Lab).astype(np.float32) / 255.0


def rgb_to_opponent(image: npt.NDArray[np.int_]) -> npt.NDArray[np.float32]:
    """Returns the Opponent color space of the image shifted and scaled to [0, 1]."""
    r, g, b = cv2.split(image.astype(np.float32))
    L = (r + g + b) / (3 * 255.0)  # noqa: N806
    a = (r - g + 255.0) / (2 * 255.0)
    b = (b - (g + r) / 2.0 + 255.0) / (2 * 255.0)
    return cv2.merge([L, a, b])


def rgb_to_opponent_codi(image: npt.NDArray[np.int_]) -> npt.NDArray[np.float32]:
    """Returns the Opponent color space of the image (Klein/Frintrop DAGM 2012)."""
    r, g, b = cv2.split(image.astype(np.float32))
    L = (r + g + b) / (3 * 255.0)  # noqa: N806
    a = (r - g) / 255.0
    b = (b - (g + r) / 2.0) / 255.0
    return cv2.merge([L, a, b])


def gaussian_blur(
    image: npt.NDArray[np.float32], sigma: float, truncate: float = 2.5
) -> npt.NDArray[np.float32]:
    ksize = round(2 * truncate * sigma + 1) | 1  # Ensure odd
    return cv2.GaussianBlur(
        image, (ksize, ksize), sigma, borderType=cv2.BORDER_REPLICATE
    )


def resize(
    image: npt.NDArray[np.float32],
    shape: tuple[int, int],
    interpolation: int = cv2.INTER_NEAREST,
) -> npt.NDArray[np.float32]:
    return cv2.resize(image, (shape[1], shape[0]), interpolation=interpolation)


"""
 - Pyramid
-------------------------------------------------------------------------------
"""


@dataclass(frozen=True)
class Pyramid:
    data: npt.NDArray[np.object_]

    def __post_init__(self):
        assert self.data.ndim == 2

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def n_octaves(self) -> int:
        return self.shape[0]

    @property
    def n_scales(self) -> int:
        return self.shape[1]

    @property
    def flat(self) -> Iterator[npt.NDArray[np.float32]]:
        return self.data.flat  # type: ignore[return-value]

    def apply(self, func: Callable) -> Pyramid:
        data = np.zeros(self.data.size, dtype=object)
        for i, arr in enumerate(self.data.flat):
            data[i] = func(arr)
        return Pyramid(data.reshape(self.data.shape))

    def __add__(self, other: Pyramid) -> Pyramid:
        return Pyramid(self.data + other.data)

    def __sub__(self, other: Pyramid) -> Pyramid:
        return Pyramid(self.data - other.data)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Pyramid(shape={self.shape})"

    def __len__(self) -> int:
        return len(self.data)


"""
- Pyramid Building
------------------------------------------------------------------------------
"""


def pyramid_octave_shapes(
    image_shape: Resolution2D,
    max_octaves: int | None = None,
    min_size: int | None = None,
) -> list[tuple[int, int]]:
    """Compute the shapes of the pyramid levels.

    Args:
        image_shape: The shape of the image from which the pyramid will be built.
        max_octaves: The maximum number of levels in the pyramid.
        min_size: The minimum size of the pyramid levels.

    Returns:
        A list of tuples, each containing the shape of a pyramid level.
    """
    max_possible_octaves = int(np.log2(min(image_shape))) + 1
    if max_octaves:
        max_octaves = min(max_octaves, max_possible_octaves)
    else:
        max_octaves = max_possible_octaves

    min_size = min_size or 1

    cur_shape = image_shape
    shapes: list[tuple[int, int]] = []
    while len(shapes) < max_octaves and min(cur_shape) >= min_size:
        shapes.append(cur_shape)
        cur_shape = (cur_shape[0] // 2, cur_shape[1] // 2)

    return shapes


def gaussian_pyramid(
    image: npt.NDArray[np.float32],
    sigma: float,
    n_scales: int,
    max_octaves: int | None = None,
    min_size: int | None = None,
) -> Pyramid:
    """Build multi-scale pyramid following Lowe 2004.

    This creates a 2D pyramid structure:
    - Dimension 1 (octaves): Different resolutions (each half the previous)
    - Dimension 2 (scales): Different smoothing levels within each octave

    Args:
        image: Input image (single channel, float32)
        sigma: Base sigma for Gaussian smoothing
        n_scales: Number of scales in each octave
        max_octaves: Maximum number of levels in the pyramid
        min_size: Minimum size of the pyramid levels

    Returns:
        2D object-type array with shape (n_octaves, n_scales)

    Note that sigmas = [sigma * (2.0 ** (s / n_scales)) for s in range(pyr.size)]
    """
    # Calculate maximum number of octaves
    shapes = pyramid_octave_shapes(
        cast("Resolution2D", image.shape),
        max_octaves=max_octaves,
        min_size=min_size,
    )

    # Compute pyramid as in Lowe 2004
    data = np.zeros((len(shapes), n_scales + 1), dtype=object)
    for octave in range(len(shapes)):
        # Compute n_scales + 1 (extra scale used as first of next octave)
        for scale in range(n_scales + 1):
            # First scale of first octave: smooth tmp
            if octave == 0 and scale == 0:
                src = image
                dst = gaussian_blur(src, sigma)

            # First scale of other octaves: subsample additional scale of previous
            elif octave > 0 and scale == 0:
                src = data[octave - 1, n_scales]
                dst = resize(src, shapes[octave])

            # Intermediate scales: smooth previous scale
            else:
                target_sigma = sigma * 2.0 ** (scale / n_scales)
                previous_sigma = sigma * 2.0 ** ((scale - 1) / n_scales)
                sig_diff = np.sqrt(target_sigma**2 - previous_sigma**2)
                src = data[octave, scale - 1]
                dst = gaussian_blur(src, sig_diff)

            data[octave, scale] = dst

    # Erase the temporary scale in each octave that was just used to
    # compute the sigmas for the next octave.
    data = data[:, :-1]
    return Pyramid(data)


def center_surround_pyramids(
    image: npt.NDArray[np.float32],
    center_sigma: float,
    surround_sigma: float,
    n_scales: int,
    max_octaves: int | None = None,
    min_size: int | None = None,
) -> tuple[Pyramid, Pyramid]:
    """Build center and surround pyramids.

    Args:
        image: The image to build the pyramids from.
        center_sigma: The sigma for the center pyramid.
        surround_sigma: The sigma for the surround pyramid.
        n_scales: The number of scales in each pyramid.
        max_octaves: An optional maximum number of levels in the pyramids.
        min_size: An optional minimum size of the images in the last octave.

    Returns:
        A tuple of center and surround pyramids.

    Raises:
        ValueError: If center sigma is greater than or equal to surround sigma.
    """
    if center_sigma >= surround_sigma:
        raise ValueError("Center sigma must be less than surround sigma")

    center: Pyramid = gaussian_pyramid(
        image,
        sigma=center_sigma,
        n_scales=n_scales,
        max_octaves=max_octaves,
        min_size=min_size,
    )

    n_octaves, n_scales = center.shape

    # Use adapted surround sigma, a la VOCUS2.
    adapted_sigma = np.sqrt(surround_sigma**2 - center_sigma**2)
    surround_data = np.zeros((n_octaves, n_scales), dtype=object)
    for level in range(n_octaves):
        for scale in range(n_scales):
            scaled_sigma = adapted_sigma * (2.0 ** (scale / n_scales))
            center_img = center.data[level, scale]
            surround_data[level, scale] = gaussian_blur(center_img, scaled_sigma)

    surround: Pyramid = Pyramid(surround_data)

    return center, surround


def laplacian_pyramid(pyr: Pyramid) -> Pyramid:
    """Build a multiscale Laplacian pyramid.

    Args:
        pyr: The pyramid to build the Laplacian pyramid from.

    Returns:
        A laplacian pyramid. Has one fewer octaves than the input pyramid.

    Raises:
        ValueError: If input pyramid doesn't have at least two octaves.
    """
    if pyr.n_octaves <= 1:
        raise ValueError("Input pyramid must have at least 2 octaves.")

    lap_octaves = pyr.n_octaves - 1
    lap = np.zeros([lap_octaves, pyr.n_scales], dtype=object)
    for scale in range(pyr.n_scales):
        for octave in range(lap_octaves):
            center = pyr.data[octave, scale]
            surround = resize(
                pyr.data[octave + 1, scale], center.shape, interpolation=cv2.INTER_CUBIC
            )
            lap[octave, scale] = center - surround

    return Pyramid(lap)


"""
- Operations on Pyramids
-------------------------------------------------------------------------------
"""


class PyramidCombine(Protocol):
    def __call__(self, pyramids: Sequence[Pyramid]) -> Pyramid: ...


class PyramidCollapse(Protocol):
    def __call__(self, pyr: Pyramid) -> npt.NDArray[np.float32]: ...


def pyramid_combine(
    pyramids: Sequence[Pyramid],
    reduce: Callable[[Sequence[npt.NDArray[np.float32]]], npt.NDArray[np.float32]],
) -> Pyramid:
    """Combine multiple pyramids into a single pyramid.

    Args:
        pyramids: The pyramids to combine.
        reduce: The function to use to reduce the pyramids.

    Returns:
        A new pyramid.

    Raises:
        ValueError: If no pyramids are provided.
    """
    n_pyramids = len(pyramids)
    if n_pyramids == 0:
        raise ValueError("No pyramids to combine")
    if n_pyramids == 1:
        return pyramids[0]

    pyr_arrays = [pyr.data for pyr in pyramids]
    pyr_shape = pyr_arrays[0].shape
    if not all(pyr.shape == pyr_shape for pyr in pyr_arrays[1:]):
        raise ValueError("All pyramids must have the same shape")
    pyr_size = pyr_arrays[0].size
    planes = np.zeros(pyr_size, dtype=object)
    for i, images in enumerate(zip(*[pyr.flat for pyr in pyramids])):
        planes[i] = reduce(images)

    return Pyramid(planes.reshape(pyr_shape))


def pyramid_combine_max(pyramids: Sequence[Pyramid]) -> Pyramid:
    return pyramid_combine(pyramids, reduce=lambda x: np.max(x, axis=0))


def pyramid_combine_mean(pyramids: Sequence[Pyramid]) -> Pyramid:
    return pyramid_combine(pyramids, reduce=lambda x: np.mean(x, axis=0))


def pyramid_collapse(
    pyr: Pyramid,
    reduce: Callable[[Sequence[npt.NDArray[np.float32]]], npt.NDArray[np.float32]],
) -> npt.NDArray[np.float32]:
    """Collapse a pyramid into a single image.

    Args:
        pyr: The pyramid to collapse.
        reduce: The function to use to reduce the pyramid's planes into one.

    Returns:
        A new image.

    """
    images = list(pyr.flat)
    target_shape = images[0].shape
    resized = []
    for img in images:
        if img.shape != target_shape:
            resized.append(resize(img, target_shape, interpolation=cv2.INTER_CUBIC))
        else:
            resized.append(img)
    return reduce(resized)


def pyramid_collapse_max(pyr: Pyramid) -> npt.NDArray[np.float32]:
    return pyramid_collapse(pyr, lambda x: np.max(x, axis=0))


def pyramid_collapse_mean(pyr: Pyramid) -> npt.NDArray[np.float32]:
    return pyramid_collapse(pyr, lambda x: np.mean(x, axis=0))


"""
- Operations on Feature Maps
"""

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


@dataclass(frozen=True)
class SafeOperatingLimits(OperatingLimits):
    min_center_sigma: float = 1.0
    max_center_sigma: float = 6.0
    max_surround_sigma: float = 12.0
    center_surround_sigma_ratio: float = 1.6
    min_image_dim_size: int = 64

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


class ColorChannelSalience:
    def __init__(
        self,
        center_sigma: float = 2.0,
        surround_sigma: float = 3.0,
        n_scales: int = 2,
        max_octaves: int | None = None,
        min_size: int | None = None,
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
            min_size: The minimum image size to construct the pyramids at.
            combine: The function to combine the on/off pyramids into a single pyramid.
            collapse: The function to collapse the combined pyramid into a single image.
            operating_limits: The operating limits to use.
        """
        self._center_sigma = center_sigma
        self._surround_sigma = surround_sigma
        self._n_scales = n_scales
        self._max_octaves = max_octaves
        self._min_size = min_size
        self._combine = combine
        self._collapse = collapse
        self._operating_limits = (
            operating_limits if operating_limits is not None else SafeOperatingLimits()
        )

        error = self._operating_limits.validate_center_and_surround_sigma(
            self._center_sigma, self._surround_sigma
        )
        if error is not None:
            raise error

    @classmethod
    def without_operating_limits(
        cls,
        center_sigma: float = 2.0,
        surround_sigma: float = 3.0,
        n_scales: int = 2,
        max_octaves: int | None = None,
        min_size: int | None = None,
        combine: PyramidCombine = pyramid_combine_mean,
        collapse: PyramidCollapse = pyramid_collapse_mean,
    ) -> ColorChannelSalience:
        """Create a `ColorChannelSalience` without operating limits.

        It is up to you to ensure that the combination of parameters you select is
        valid for your use case.

        Args:
            center_sigma: The center sigma for the center/surround pyramids.
            surround_sigma: The surround sigma for the center/surround pyramids.
            n_scales: The number of pyramid scales.
            max_octaves: The maximum number of pyramid octaves to construct.
            min_size: The minimum image size to construct the pyramids at.
            combine: The function to combine the on/off pyramids into a single pyramid.
            collapse: The function to collapse the combined pyramid into a single image.

        Returns:
            A `ColorChannelSalience` without operating limits.
        """
        return cls(
            center_sigma=center_sigma,
            surround_sigma=surround_sigma,
            n_scales=n_scales,
            max_octaves=max_octaves,
            min_size=min_size,
            combine=combine,
            collapse=collapse,
            operating_limits=NoOperatingLimits(),
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
            ValueError: If the min size is greater than the image size.
        """
        if self._min_size is not None and self._min_size > min(image.shape):
            raise ValueError("Min size is greater than the image size")

        error = self._operating_limits.validate_image_dim_size(min(image.shape))
        if error is not None:
            if ctx.suppress_runtime_errors:
                logger.warning(str(error))
            else:
                raise error

        center, surround = center_surround_pyramids(
            image,
            center_sigma=self._center_sigma,
            surround_sigma=self._surround_sigma,
            n_scales=self._n_scales,
            max_octaves=self._max_octaves,
            min_size=self._min_size,
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
        min_size: int | None = None,
        collapse: PyramidCollapse = pyramid_collapse_mean,
        operating_limits: OperatingLimits | None = None,
    ):
        self._center_sigma = center_sigma
        self._surround_sigma = surround_sigma
        self._n_scales = n_scales
        self._max_octaves = max_octaves
        self._min_size = min_size
        self._collapse = collapse
        self._operating_limits = (
            operating_limits if operating_limits is not None else SafeOperatingLimits()
        )

        error = self._operating_limits.validate_center_and_surround_sigma(
            self._center_sigma, self._surround_sigma
        )
        if error is not None:
            raise error

    @classmethod
    def without_operating_limits(
        cls,
        center_sigma: float = 2.0,
        surround_sigma: float = 3.0,
        n_scales: int = 2,
        max_octaves: int | None = None,
        min_size: int | None = None,
        collapse: PyramidCollapse = pyramid_collapse_mean,
    ) -> DepthSalience:
        """Create a `DepthSalience` without operating limits.

        It is up to you to ensure that the combination of parameters you select is
        valid for your use case.

        Args:
            center_sigma: The center sigma for the center/surround pyramids.
            surround_sigma: The surround sigma for the center/surround pyramids.
            n_scales: The number of pyramid scales.
            max_octaves: The maximum number of pyramid octaves to construct.
            min_size: The minimum image size to construct the pyramids at.
            collapse: The function to collapse the combined pyramid into a single image.

        Returns:
            A `DepthSalience` without operating limits.
        """
        return cls(
            center_sigma=center_sigma,
            surround_sigma=surround_sigma,
            n_scales=n_scales,
            max_octaves=max_octaves,
            min_size=min_size,
            collapse=collapse,
            operating_limits=NoOperatingLimits(),
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
            ValueError: If the min size is greater than the image size.
        """
        if self._min_size is not None and self._min_size > min(image.shape):
            raise ValueError("Min size is greater than the image size")

        error = self._operating_limits.validate_image_dim_size(min(image.shape))
        if error is not None:
            if ctx.suppress_runtime_errors:
                logger.warning(str(error))
            else:
                raise error

        image = -np.log(image).astype(np.float32)
        image = np.nan_to_num(image, posinf=0.0)

        center, surround = center_surround_pyramids(
            image,
            center_sigma=self._center_sigma,
            surround_sigma=self._surround_sigma,
            n_scales=self._n_scales,
            max_octaves=self._max_octaves,
            min_size=self._min_size,
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

    def process(self, pyr: Pyramid) -> npt.NDArray[np.float32]:
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
    min_size: int = 16
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
            min_size=config.min_size,
        )

        depth = (
            DepthSalience(
                center_sigma=config.center_sigma,
                surround_sigma=config.surround_sigma,
                n_scales=config.n_scales,
                max_octaves=config.max_octaves,
                min_size=config.min_size,
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
        rgba: npt.NDArray[np.int_],
        depth: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        # Get color and depth data into open-cv compatible formats.
        rgb = rgba[:, :, :3]
        depth = depth.astype(np.float32)

        feature_maps = FeatureMaps({})

        Lab = self._color_space_converter(rgb)  # noqa: N806
        L, a, b = cv2.split(Lab)  # noqa: N806
        feature_maps["L"], L_center = self._color.process(L)  # noqa: N806
        feature_maps["a"], _ = self._color.process(a)
        feature_maps["b"], _ = self._color.process(b)

        if self._depth:
            feature_maps["depth"] = self._depth.process(depth)

        if self._orientation:
            feature_maps["orientation"] = self._orientation.process(L_center)

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
