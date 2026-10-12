# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing_extensions import Self

from tbp.monty.sensor_modules.sensor_module import Payload, Transform, TransformContext
from tbp.monty.sensor_modules.transforms.transform import (
    NoDepthSensorPresent,
)

__all__ = [
    "AddNoiseToRawDepthImage",
]


class AddNoiseToRawDepthImage(Transform):
    """Add Gaussian noise to raw sensory input."""

    _sigma: float

    def __init__(self: Self, sigma: float) -> None:
        """Initialize the transform.

        Args:
            sigma: The standard deviation of the noise.
        """
        self._sigma = sigma

    def __call__(self: Self, ctx: TransformContext, payload: Payload) -> Payload:
        """Add Gaussian noise to raw sensory input.

        Args:
            ctx: The transform context.
            payload: The payload to transform. Must contain a depth sensor.

        Returns:
            A payload with the transformed observation.

        Raises:
            NoDepthSensorPresent: If no depth sensor is present.
        """
        if "depth" in payload.observation:
            noise = ctx.rng.normal(0, self._sigma, payload.observation["depth"].shape)
            payload.observation["depth"] += noise
        else:
            raise NoDepthSensorPresent("Don't use this transform.")
        return payload
