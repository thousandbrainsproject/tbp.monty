# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Protocol, Sequence

import cv2
import numpy as np
import numpy.typing as npt

from tbp.monty.frameworks.models.salience.strategies.vocus2.images import (
    gaussian_blur,
    resize,
)
from tbp.monty.frameworks.sensors import Resolution2D


@dataclass(frozen=True)
class Pyramid:
    data: npt.NDArray[np.object_]

    def __post_init__(self):
        assert self.data.ndim == 2
        assert self.data.size > 0

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


def pyramid_octave_shapes(
    image_shape: tuple[int, int],
    max_octaves: int | None = None,
) -> list[Resolution2D]:
    """Compute the shapes of the pyramid levels.

    Args:
        image_shape: The shape of the plane from which the pyramid will be built.
        max_octaves: The maximum number of levels in the pyramid.

    Returns:
        A list of resolutions, each being the image shape for a pyramid's octave.
    """
    max_possible_octaves = int(np.log2(min(image_shape))) + 1
    if max_octaves is not None:
        max_possible_octaves = min(max_octaves, max_possible_octaves)

    cur_shape = image_shape
    shapes: list[Resolution2D] = []
    while len(shapes) < max_possible_octaves and min(cur_shape) >= 1:
        shapes.append(cur_shape)
        cur_shape = (cur_shape[0] // 2, cur_shape[1] // 2)

    return shapes


def gaussian_pyramid(
    image: npt.NDArray[np.float32],
    sigma: float,
    n_scales: int,
    max_octaves: int | None = None,
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

    Returns:
        2D object-type array with shape (n_octaves, n_scales)

    Raises:
        ValueError: If image has size 0 or is not 2D.

    Note that sigmas = [sigma * (2.0 ** (s / n_scales)) for s in range(pyr.size)]
    """
    if image.size == 0:
        raise ValueError(f"Image must have a non-zero size. Has shape {image.shape}.")
    if image.ndim != 2:
        raise ValueError(f"Image must be 2D. Has shape {image.shape}.")

    # Calculate maximum number of octaves
    shapes: list[tuple[int, int]] = pyramid_octave_shapes(
        image.shape, max_octaves=max_octaves
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
                dst = resize(src, shapes[octave], interpolation=cv2.INTER_AREA)

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
) -> tuple[Pyramid, Pyramid]:
    """Build center and surround pyramids.

    Args:
        image: The image to build the pyramids from.
        center_sigma: The sigma for the center pyramid.
        surround_sigma: The sigma for the surround pyramid.
        n_scales: The number of scales in each pyramid.
        max_octaves: An optional maximum number of levels in the pyramids.

    Returns:
        A tuple of center and surround pyramids.

    Raises:
        ValueError: If center sigma is greater than or equal to surround sigma.
    """
    if center_sigma >= surround_sigma:
        raise ValueError("Center sigma must be strictly less than surround sigma")

    center: Pyramid = gaussian_pyramid(
        image,
        sigma=center_sigma,
        n_scales=n_scales,
        max_octaves=max_octaves,
    )

    n_octaves, n_scales = center.shape

    # Use adapted surround sigma, a la VOCUS2.
    adapted_sigma = np.sqrt(surround_sigma**2 - center_sigma**2)
    surround_data = np.zeros((n_octaves, n_scales), dtype=object)
    for octave in range(n_octaves):
        for scale in range(n_scales):
            scaled_sigma = adapted_sigma * (2.0 ** (scale / n_scales))
            center_img = center.data[octave, scale]
            surround_data[octave, scale] = gaussian_blur(center_img, scaled_sigma)

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
                pyr.data[octave + 1, scale],
                center.shape,
                interpolation=cv2.INTER_CUBIC,
            )
            lap[octave, scale] = center - surround

    return Pyramid(lap)


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
