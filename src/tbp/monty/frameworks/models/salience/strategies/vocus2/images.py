# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np
import numpy.typing as npt


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
