# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy
from typing_extensions import Self

from tbp.monty.sensor_modules.sensor_module import Payload, Transform, TransformContext
from tbp.monty.sensor_modules.transforms.transform import (
    NoDepthSensorPresent,
)

__all__ = [
    "GaussianSmoothing",
]


class GaussianSmoothing(Transform):
    """Deals with Gaussian noise on the raw depth image.

    This transform is designed to deal with Gaussian noise on the raw depth
    image. It remains to be tested whether it will also help with real-world
    depth-camera noise.
    """

    _kernel_width: int
    _sigma: float
    _pad_size: int
    _kernel: npt.NDArray[np.float64]

    def __init__(self: Self, sigma: float, kernel_width: int) -> None:
        """Initialize the transform.

        Args:
            sigma: The standard deviation of the noise.
            kernel_width: The width of the kernel.
        """
        self._sigma = sigma
        self._kernel_width = kernel_width
        self._pad_size = kernel_width // 2
        self._kernel = self._create_kernel()

    def _create_kernel(self: Self) -> npt.NDArray[np.float64]:
        """Create a normalized Gaussian kernel.

        Returns:
            Normalized Gaussian kernel. Array of size (kernel_width, kernel_width).
        """
        x = np.linspace(-self._pad_size, self._pad_size, self._kernel_width)
        kernel_1d = (
            1.0
            / (np.sqrt(2 * np.pi) * self._sigma)
            * np.exp(-np.square(x) / (2 * self._sigma**2))
        )
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        return kernel_2d / np.sum(kernel_2d)

    def _pad(self: Self, img: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Pad an image.

        Args:
            img: The image to pad.

        Returns:
            Padded image.
        """
        return np.pad(img.astype(float), pad_width=self._pad_size, mode="edge")

    def __call__(
        self: Self,
        ctx: TransformContext,  # noqa: ARG002
        payload: Payload,
    ) -> Payload:
        """Apply Gaussian smoothing to the raw depth image.

        Args:
            ctx: The transform context.
            payload: The payload to transform. Must contain a depth sensor.

        Returns:
            A payload with the transformed observation.

        Raises:
            NoDepthSensorPresent: If no depth sensor is present.
        """
        if "depth" in payload.observation:
            depth_img = payload.observation["depth"].copy()
            padded_img = self._pad(depth_img)
            filtered_img = scipy.signal.convolve(padded_img, self._kernel, mode="valid")
            payload.observation["depth"] = filtered_img
        else:
            raise NoDepthSensorPresent("Don't use this transform.")
        return payload
