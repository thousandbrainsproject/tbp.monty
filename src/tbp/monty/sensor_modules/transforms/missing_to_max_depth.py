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
from typing_extensions import Self

from tbp.monty.sensor_modules.sensor_module import (
    Payload,
    Transform,
    TransformContext,
)

__all__ = [
    "MissingToMaxDepth",
]


class MissingToMaxDepth(Transform):
    """Return max depth when no mesh is present at a location.

    Habitat depth sensors return 0 when no mesh is present at a location. Instead,
    return max_depth. See:
    https://github.com/facebookresearch/habitat-sim/issues/1157 for discussion.
    """

    _max_depth: float
    _threshold: float

    def __init__(
        self: Self,
        max_depth: float,
        threshold: float = 0.0,
    ):
        """Initialize the transform.

        Args:
            max_depth: numeric that will replace missing
            threshold: (optional) numeric, anything less than this is counted as
                missing. Defaults to 0.0.
        """
        self._max_depth = max_depth
        self._threshold = threshold

    def __call__(
        self: Self,
        ctx: TransformContext,  # noqa: ARG002
        payload: Payload,
    ) -> Payload:
        m = np.where(payload.observation["depth"] <= self._threshold)
        payload.observation["depth"][m] = self._max_depth
        return payload
