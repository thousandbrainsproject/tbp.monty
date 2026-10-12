# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from tbp.monty.geometry import Rotation
from tbp.monty.sensor_modules.sensor_module import Payload, Transform, TransformContext

__all__ = [
    "Message",
]


class Message(Transform):
    """Add noise to features specified in noise_params.

    Noise params should have structure {"features":
                                            {"feature_keys": noise_amount, ...},
                                        "locations": noise_amount}
    noise_amount specifies the standard deviation of the gaussian noise sampled
    for real valued features. For boolean features it specifies the probability
    that the boolean flips.
    If we are dealing with normed vectors (surface_normal or curvature_directions),
    the noise is applied by rotating the vector given a sampled rotation. Otherwise
    noise is just added onto the perceived feature value.
    """

    _noise_params: dict[str, Any]

    def __init__(self: Self, noise_params: dict[str, Any]) -> None:
        self._noise_params = noise_params

    def __call__(self: Self, ctx: TransformContext, payload: Payload) -> Payload:
        percept = payload.percept
        if percept is None:
            return payload

        if "features" in self._noise_params:
            for key in self._noise_params["features"]:
                if key in percept.morphological_features:
                    if key == "pose_vectors":
                        # apply randomly sampled rotation to xyz axes with standard
                        # deviation specified in noise_params
                        # TODO: apply same rotation to both to make sure they stay
                        # orthogonal?
                        noise_angles = ctx.rng.normal(
                            0, self._noise_params["features"][key], 3
                        )
                        noise_rotation = Rotation.from_euler(
                            "xyz", noise_angles, degrees=True
                        )
                        percept.morphological_features[key] = noise_rotation.apply(
                            percept.morphological_features[key]
                        )
                    else:
                        percept.morphological_features[key] = (
                            self._add_noise_to_feat_value(
                                ctx=ctx,
                                feat_name=key,
                                feat_val=percept.morphological_features[key],
                            )
                        )
                elif key in percept.non_morphological_features:
                    percept.non_morphological_features[key] = (
                        self._add_noise_to_feat_value(
                            ctx=ctx,
                            feat_name=key,
                            feat_val=percept.non_morphological_features[key],
                        )
                    )
        if "location" in self._noise_params:
            noise = ctx.rng.normal(0, self._noise_params["location"], 3)
            percept.location = percept.location + noise

        payload.percept = percept

        return payload

    def _add_noise_to_feat_value(
        self: Self,
        ctx: TransformContext,
        feat_name: str,
        feat_val: bool | npt.NDArray[np.float64],
    ) -> bool | npt.NDArray[np.float64]:
        if isinstance(feat_val, bool):
            # Flip boolean variable with probability specified in
            # noise_params
            if ctx.rng.random() < self._noise_params["features"][feat_name]:
                return not (feat_val)

            return feat_val

        # Add gaussian noise with standard deviation specified in
        # noise_params
        shape = feat_val.shape
        noise = ctx.rng.normal(0, self._noise_params["features"][feat_name], shape)
        new_feat_val = feat_val + noise
        if feat_name == "hsv":  # make sure hue stays in 0-1 range
            new_feat_val[0] = np.clip(new_feat_val[0], 0, 1)
        return new_feat_val
