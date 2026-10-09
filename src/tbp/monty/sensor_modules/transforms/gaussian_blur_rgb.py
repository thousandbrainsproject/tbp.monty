# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import cv2
import numpy as np
from typing_extensions import Self

from tbp.monty.sensor_modules.sensor_module import Payload, Transform, TransformContext
from tbp.monty.sensor_modules.transforms.transform import (
    NoRGBASensorPresent,
)

__all__ = [
    "GaussianBlurRGB",
]


class GaussianBlurRGB(Transform):
    """Apply Gaussian blur to RGB image."""

    _sigma: float
    _kernel_size: int

    def __init__(
        self: Self,
        sigma: float = 1.0,
        kernel_size: int = 0,
    ):
        """Initialize the transform.

        Args:
            sigma: Standard deviation for Gaussian blur. Default is 1.0.
            kernel_size: Kernel size for blur. If 0 (default), OpenCV auto-computes
                from sigma using `6*sigma + 1` rounded to nearest odd. If specified,
                must be odd.

        Raises:
            ValueError: If kernel_size is even (when not 0).
        """
        if kernel_size < 0:
            raise ValueError(
                f"The kernel_size must be non-negative, got {kernel_size}."
            )
        if kernel_size != 0 and kernel_size % 2 == 0:
            raise ValueError(
                f"The kernel_size must be odd or 0 (for auto-compute), "
                f"got {kernel_size}."
            )
        if kernel_size == 0 and sigma <= 0:
            raise ValueError(
                f"The sigma must be positive when kernel_size is 0, got {sigma}."
            )

        self._sigma = sigma
        self._kernel_size = kernel_size

    def __call__(
        self: Self,
        ctx: TransformContext,  # noqa: ARG002
        payload: Payload,
    ) -> Payload:
        """Apply Gaussian blur to RGB image.

        Args:
            ctx: The transform context.
            payload: The payload to transform. Must contain an observation with "rgba"
                modality.

        Returns:
            A payload with the transformed observation.

        Raises:
            NoRGBASensorPresent: If no RGBA sensor is present.
        """
        if "rgba" not in payload.observation:
            raise NoRGBASensorPresent("Don't use this transform.")

        rgba = payload.observation["rgba"]
        rgb_image = rgba[:, :, :3]
        alpha_channel = rgba[:, :, 3:4]

        blurred_rgb = cv2.GaussianBlur(
            rgb_image, (self._kernel_size, self._kernel_size), self._sigma
        )

        payload.observation["rgba"] = np.concatenate(
            [blurred_rgb, alpha_channel], axis=2
        )

        return payload
