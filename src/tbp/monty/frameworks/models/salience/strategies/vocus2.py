# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import timeit
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from tbp.monty.frameworks.models.salience.strategies import SalienceStrategy

"""
- Color
-------------------------------------------------------------------------------
"""


class ColorSpace(Enum):
    """Color space options."""

    LAB = "LAB"  # CIE Lab color space
    OPPONENT_CODI = "OPPONENT_CODI"  # Opponent color space (Klein/Frintrop DAGM 2012)
    OPPONENT = "OPPONENT"  # Opponent color space (shifted and scaled to [0, 1])


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2Lab).astype(np.float32) / 255.0


def rgb_to_opponent(image: np.ndarray) -> np.ndarray:
    r, g, b = cv2.split(image.astype(np.float32))
    L = (r + g + b) / (3 * 255.0)  # noqa: N806
    a = (r - g + 255.0) / (2 * 255.0)
    b = (b - (g + r) / 2.0 + 255.0) / (2 * 255.0)
    return cv2.merge([L, a, b])


def rgb_to_opponent_codi(image: np.ndarray) -> np.ndarray:
    r, g, b = cv2.split(image.astype(np.float32))
    L = (r + g + b) / (3 * 255.0)  # noqa: N806
    a = (r - g) / 255.0
    b = (b - (g + r) / 2.0) / 255.0
    return cv2.merge([L, a, b])


_COLOR_SPACE_CONVERTERS = {
    ColorSpace.LAB: rgb_to_lab,
    ColorSpace.OPPONENT: rgb_to_opponent,
    ColorSpace.OPPONENT_CODI: rgb_to_opponent_codi,
}


def gaussian_blur(image: np.ndarray, sigma: float, truncate: float = 2.5) -> np.ndarray:
    ksize = round(2 * truncate * sigma + 1) | 1  # Ensure odd
    return cv2.GaussianBlur(
        image, (ksize, ksize), sigma, borderType=cv2.BORDER_REPLICATE
    )


def resize(
    image: np.ndarray,
    shape: tuple[int, int],
    interpolation: int = cv2.INTER_NEAREST,
) -> np.ndarray:
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


def pyramid_level_shapes(
    image_shape: tuple[int, int],
    max_levels: int | None = None,
    min_size: int | None = None,
) -> list[tuple[int, int]]:
    """Compute the shapes of the pyramid levels.

    Args:
        image_shape: The shape of the image from which the pyramid will be built.
        max_levels: The maximum number of levels in the pyramid.
        min_size: The minimum size of the pyramid levels.

    Returns:
        A list of tuples, each containing the shape of a pyramid level.
    """
    max_possible_octaves = int(np.log2(min(image_shape))) + 1
    if max_levels:
        max_levels = min(max_levels, max_possible_octaves)
    else:
        max_levels = max_possible_octaves

    min_size = min_size or 1

    cur_shape = image_shape
    shapes = []
    while len(shapes) < max_levels and min(cur_shape) >= min_size:
        shapes.append(cur_shape)
        cur_shape = (cur_shape[0] // 2, cur_shape[1] // 2)

    return shapes


def gaussian_pyramid(
    image: np.ndarray,
    sigma: float,
    n_scales: int,
    max_levels: int | None = None,
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
        max_levels: Maximum number of levels in the pyramid
        min_size: Minimum size of the pyramid levels

    Returns:
        2D object-type array with shape (n_octaves, n_scales)

    Note: sigmas = [sigma * (2.0 ** (s / n_scales)) for s in range(pyr.size)]
    """
    # Calculate maximum number of octaves
    shapes = pyramid_level_shapes(image.shape, max_levels=max_levels, min_size=min_size)

    # Compute pyramid as in Lowe 2004
    pyr = np.zeros((len(shapes), n_scales + 1), dtype=object)
    for octave in range(len(shapes)):
        # Compute n_scales + 1 (extra scale used as first of next octave)
        for scale in range(n_scales + 1):
            # First scale of first octave: smooth tmp
            if octave == 0 and scale == 0:
                src = image
                dst = gaussian_blur(src, sigma)

            # First scale of other octaves: subsample additional scale of previous
            elif octave > 0 and scale == 0:
                src = pyr[octave - 1, n_scales]
                dst = resize(src, shapes[octave])

            # Intermediate scales: smooth previous scale
            else:
                target_sigma = sigma * 2.0 ** (scale / n_scales)
                previous_sigma = sigma * 2.0 ** ((scale - 1) / n_scales)
                sig_diff = np.sqrt(target_sigma**2 - previous_sigma**2)
                src = pyr[octave, scale - 1]
                dst = gaussian_blur(src, sig_diff)

            pyr[octave, scale] = dst

    # Erase the temporary scale in each octave that was just used to
    # compute the sigmas for the next octave.
    pyr = pyr[:, :-1]
    return Pyramid(pyr)


def center_surround_pyramids(
    image: np.ndarray,
    center_sigma: float,
    surround_sigma: float,
    n_scales: int,
    **kwargs,
) -> tuple[Pyramid, Pyramid]:
    center = gaussian_pyramid(image, sigma=center_sigma, n_scales=n_scales, **kwargs)

    center = center if isinstance(center, Pyramid) else Pyramid(center)
    n_octaves, n_scales = center.data.shape

    # Use adapted surround sigma, a la VOCUS2.
    adapted_sigma = np.sqrt(surround_sigma**2 - center_sigma**2)
    surround = np.zeros((n_octaves, n_scales), dtype=object)
    for level in range(n_octaves):
        for scale in range(n_scales):
            scaled_sigma = adapted_sigma * (2.0 ** (scale / n_scales))
            center_img = center.data[level, scale]
            surround[level, scale] = gaussian_blur(center_img, scaled_sigma)

    surround: Pyramid = Pyramid(surround)

    return center, surround


def laplacian_pyramid(
    pyr: Pyramid,
    max_levels: int | None = None,
    min_size: int | None = None,
) -> Pyramid:
    """Build a multiscale Laplacian pyramid.

    Args:
        pyr: The pyramid to build the Laplacian pyramid from.
        max_levels: The maximum number of levels in the pyramid.
        min_size: The minimum size of the pyramid levels.

    Returns:
        A new pyramid.
    """
    gauss = pyr.data
    n_levels_in = gauss.shape[0]
    n_levels_out = n_levels_in - 1
    if max_levels is not None:
        n_levels_out = min(n_levels_out, max_levels)
    if min_size is not None:
        level_sizes = np.array([min(arrays[0].shape) for arrays in gauss])
        n_levels_big_enough = sum(level_sizes >= min_size)
        n_levels_out = min(n_levels_out, n_levels_big_enough)

    lap = np.zeros([n_levels_out, gauss.shape[1]], dtype=object)
    for scale in range(lap.shape[1]):
        for octave in range(lap.shape[0]):
            center = gauss[octave, scale]
            surround = resize(
                gauss[octave + 1, scale], center.shape, interpolation=cv2.INTER_CUBIC
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
    def __call__(self, pyr: Pyramid) -> np.ndarray: ...


def pyramid_combine(
    pyramids: Sequence[Pyramid],
    fn: Callable[[Sequence[np.ndarray]], np.ndarray],
) -> np.ndarray:
    """Combine multiple pyramids into a single pyramid.

    Args:
        pyramids: The pyramids to combine.
        fn: The function to use to combine the pyramids.

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
    assert all(pyr.shape == pyr_shape for pyr in pyr_arrays[1:])
    pyr_size = pyr_arrays[0].size
    planes = np.zeros(pyr_size, dtype=object)
    for i, images in enumerate(zip(*[pyr.flat for pyr in pyramids])):
        target_shape = images[0].shape
        resized = []
        for img in images:
            if img.shape != target_shape:
                resized.append(resize(img, target_shape))
            else:
                resized.append(img)
        planes[i] = fn(resized)

    return Pyramid(planes.reshape(pyr_shape))


def pyramid_combine_max(pyramids: Sequence[Pyramid]) -> Pyramid:
    return pyramid_combine(pyramids, lambda x: np.max(x, axis=0))


def pyramid_combine_mean(pyramids: Sequence[Pyramid]) -> Pyramid:
    return pyramid_combine(pyramids, lambda x: np.mean(x, axis=0))


def pyramid_collapse(
    pyr: Pyramid,
    fn: Callable[[Sequence[np.ndarray]], np.ndarray],
) -> np.ndarray:
    """Collapse a pyramid into a single image.

    Args:
        pyr: The pyramid to collapse.
        fn: The function to use to collapse the pyramid.

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
    return fn(resized)


def pyramid_collapse_max(pyr: Pyramid) -> np.ndarray:
    return pyramid_collapse(pyr, lambda x: np.max(x, axis=0))


def pyramid_collapse_mean(pyr: Pyramid) -> np.ndarray:
    return pyramid_collapse(pyr, lambda x: np.mean(x, axis=0))


"""
- Operations on Feature Maps
"""


class MapCombine(Protocol):
    def __call__(self, maps: dict[str, np.ndarray]) -> np.ndarray: ...


def map_max(maps: dict[int | str, np.ndarray]) -> np.ndarray:
    np.max(list(maps.values()), axis=0)


def map_sum(maps: dict[int | str, np.ndarray]) -> np.ndarray:
    return np.sum(list(maps.values()), axis=0)


def map_weighted_sum(
    maps: dict[int | str, np.ndarray],
    weights: dict[int | str, float],
) -> np.ndarray:
    return np.sum([weights[key] * img for key, img in maps.items()], axis=0)


def map_mean(maps: dict[int | str, np.ndarray]) -> np.ndarray:
    return np.mean(list(maps.values()), axis=0)


class WeightedMean(MapCombine):
    def __init__(self, weights: dict[str, float]):
        self._weights = dict(weights)

    def __call__(self, maps: dict[str, np.ndarray]) -> np.ndarray:
        weights = {key: self._weights[key] for key in maps}
        total_weight = sum(abs(weight) for weight in weights.values())
        normed_weights = {key: weight / total_weight for key, weight in weights.items()}
        return np.sum([normed_weights[key] * img for key, img in maps.items()], axis=0)


@dataclass
class ColorChannelSalienceResult:
    salience_map: np.ndarray
    salience_pyramid: Pyramid
    center: Pyramid
    surround: Pyramid
    on: Pyramid
    off: Pyramid


class ColorChannelSalience:
    def __init__(
        self,
        center_sigma: float,
        surround_sigma: float,
        n_scales: int,
        max_levels: int | None = None,
        min_size: int | None = None,
        combine: PyramidCombine = pyramid_combine_mean,
        collapse: PyramidCollapse = pyramid_collapse_mean,
    ):
        self._center_sigma = center_sigma
        self._surround_sigma = surround_sigma
        self._n_scales = n_scales
        self._max_levels = max_levels
        self._min_size = min_size
        self._combine = combine
        self._collapse = collapse

    def process(self, image: npt.NDArray[np.float32]) -> ColorChannelSalienceResult:
        """Compute salience for a single color channel.

        Args:
            image: Must be float32 and in the range [0, 1].

        Returns:
            A ColorChannelSalienceResult object.
        """
        # Build center/surround and on/off pyramids.
        center, surround = center_surround_pyramids(
            image,
            center_sigma=self._center_sigma,
            surround_sigma=self._surround_sigma,
            n_scales=self._n_scales,
            max_levels=self._max_levels,
            min_size=self._min_size,
        )

        # Build on/off pyramids.
        diff: Pyramid = center - surround
        on: Pyramid = diff.apply(lambda img: np.maximum(img, 0))
        off: Pyramid = diff.apply(lambda img: np.maximum(-img, 0))

        # Combine on/off pyramids, and collapse the result.
        salience_pyramid = self._combine([on, off])
        salience_map = self._collapse(salience_pyramid)

        return ColorChannelSalienceResult(
            salience_map=salience_map,
            salience_pyramid=salience_pyramid,
            center=center,
            surround=surround,
            on=on,
            off=off,
        )


@dataclass
class DepthSalienceResult:
    salience_map: np.ndarray
    salience_pyramid: Pyramid
    center: Pyramid
    surround: Pyramid


class DepthSalience:
    def __init__(
        self,
        center_sigma: float,
        surround_sigma: float,
        n_scales: int,
        max_levels: int | None = None,
        min_size: int | None = None,
        collapse: PyramidCollapse = pyramid_collapse_mean,
    ):
        self._center_sigma = center_sigma
        self._surround_sigma = surround_sigma
        self._n_scales = n_scales
        self._max_levels = max_levels
        self._min_size = min_size
        self._collapse = collapse

    def process(self, image: npt.NDArray[np.float32]) -> DepthSalienceResult:
        """Compute salience for a depth channel.

        Args:
            image: Must be float32.

        Returns:
            A DepthSalienceResult object.
        """
        image = -np.log(image).astype(np.float32)

        # Build center/surround and on/off pyramids.
        center, surround = center_surround_pyramids(
            image,
            center_sigma=self._center_sigma,
            surround_sigma=self._surround_sigma,
            n_scales=self._n_scales,
            max_levels=self._max_levels,
            min_size=self._min_size,
        )

        # Build the on pyramid, and collapse the result.
        diff: Pyramid = center - surround
        salience_pyramid = diff.apply(lambda img: np.maximum(img, 0))
        salience_map = self._collapse(salience_pyramid)

        return DepthSalienceResult(
            salience_map=salience_map,
            salience_pyramid=salience_pyramid,
            center=center,
            surround=surround,
        )


class OrientationSalienceResult:
    salience_map: np.ndarray
    feature_maps: dict[str, np.ndarray]
    feature_pyramids: dict[str, Pyramid]


class OrientationSalience:
    def __init__(
        self,
        period: float,
        sigma: float | None = None,
        phase: float = np.pi / 2,
        gamma: float = 0.75,
        n_orientations: int = 4,
        combine: MapCombine | None = map_mean,
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
    ) -> dict[str, np.ndarray]:
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

    def process(self, pyr: Pyramid) -> OrientationSalienceResult:
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

        salience_map = self._combine(feature_maps)

        return OrientationSalienceResult(
            salience_map=salience_map,
            feature_maps=feature_maps,
            feature_pyramids=feature_pyramids,
        )


@dataclass
class SalienceResult:
    salience_map: np.ndarray
    feature_maps: dict[str, np.ndarray]
    results: dict[
        str,
        ColorChannelSalienceResult | DepthSalienceResult | OrientationSalienceResult,
    ]


class Vocus2(SalienceStrategy):
    def __init__(
        self,
        color_space: str | ColorSpace = ColorSpace.OPPONENT,
        center_sigma: float = 3.0,
        surround_sigma: float = 5.0,
        n_scales: int = 2,
        max_levels: int = 5,
        min_size: int = 16,
        depth: bool = False,
        orientation: bool = False,
        combine: MapCombine | None = None,
        normalize: bool = True,
    ):
        if not isinstance(color_space, ColorSpace):
            color_space = ColorSpace(color_space)

        self._color_space = color_space
        self._color_space_converter = _COLOR_SPACE_CONVERTERS[self._color_space]

        self._center_sigma = center_sigma
        self._surround_sigma = surround_sigma
        self._n_scales = n_scales
        self._max_levels = max_levels
        self._min_size = min_size
        self._normalize = normalize

        # construct salience computers
        self._color = ColorChannelSalience(
            center_sigma=self._center_sigma,
            surround_sigma=self._surround_sigma,
            n_scales=self._n_scales,
            max_levels=self._max_levels,
            min_size=self._min_size,
        )

        if depth:
            self._depth = DepthSalience(
                center_sigma=self._center_sigma,
                surround_sigma=self._surround_sigma,
                n_scales=self._n_scales,
                max_levels=self._max_levels,
                min_size=self._min_size,
            )
        else:
            self._depth = None

        if orientation:
            self._orientation = OrientationSalience(
                period=2 * self._center_sigma,
            )
        else:
            self._orientation = None

        if combine is None:
            weights = {
                "L": 1,
                "a": 1,
                "b": 1,
                "depth": 0.1,
                "orientation": 1,
            }
            self._combine = WeightedMean(weights)
        else:
            self._combine = combine

    def __call__(
        self,
        rgba: npt.NDArray[np.int_],
        depth: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        # Get color and depth data into open-cv compatible formats.
        rgb = rgba[:, :, :3]
        depth = depth.astype(np.float32)

        results = {}
        feature_maps = {}

        Lab = self._color_space_converter(rgb)
        L, a, b = cv2.split(Lab)  # noqa: N806
        for channel, plane in zip(("L", "a", "b"), (L, a, b)):
            results[channel] = self._color.process(plane)

        if self._depth:
            results["depth"] = self._depth.process(depth)

        if self._orientation:
            results["orientation"] = self._orientation.process(results["L"].center)

        feature_maps = {result.salience_map for result in results.values()}
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
