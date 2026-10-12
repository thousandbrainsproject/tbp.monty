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
from typing import Any

import numpy as np
from typing_extensions import Self

from tbp.monty.cmp import Message
from tbp.monty.frameworks.utils.spatial_arithmetics import get_angle
from tbp.monty.sensor_modules.sensor_module import (
    Payload,
    Transform,
    TransformContext,
)

logger = logging.getLogger(__name__)

__all__ = [
    "FeatureChange",
]


class FeatureChange(Transform):
    _delta_thresholds: dict[str, Any]
    _last_percept: Message | None
    _last_sent_n_steps_ago: int

    def __init__(self, delta_thresholds: dict[str, Any]):
        self._delta_thresholds = delta_thresholds
        self._last_percept = None
        self._last_sent_n_steps_ago = 0

    def _check_feature_change(self, percept: Message) -> bool:
        """Check feature change between last transmitted observation.

        Args:
            percept: Percept to check for feature change.

        Returns:
            True if the features have changed significantly.
        """
        if not percept.get_on_object():
            # Even for the surface-agent sensor, do not return a feature for LM
            # processing that is not on the object
            logger.debug("No new point because not on object")
            return False

        for feature in self._delta_thresholds:
            if feature not in ["n_steps", "distance"]:
                last_feat = self._last_percept.get_feature_by_name(feature)
                current_feat = percept.get_feature_by_name(feature)

            if feature == "n_steps":
                if self._last_sent_n_steps_ago >= self._delta_thresholds[feature]:
                    logger.debug(f"new point because of {feature}")
                    return True
            elif feature == "distance":
                distance = np.linalg.norm(
                    np.array(self._last_percept.location) - np.array(percept.location)
                )

                if distance > self._delta_thresholds[feature]:
                    logger.debug(f"new point because of {feature}")
                    return True

            elif feature == "hsv":
                last_hue = last_feat[0]
                current_hue = current_feat[0]
                hue_d = min(
                    abs(current_hue - last_hue), 1 - abs(current_hue - last_hue)
                )
                if hue_d > self._delta_thresholds[feature][0]:
                    return True
                delta_change_sv = np.abs(last_feat[1:] - current_feat[1:])
                for i, dc in enumerate(delta_change_sv):
                    if dc > self._delta_thresholds[feature][i + 1]:
                        logger.debug(f"new point because of {feature} - {i + 1}")
                        return True

            elif feature == "pose_vectors":
                angle_between = get_angle(
                    last_feat[0],
                    current_feat[0],
                )
                if angle_between >= self._delta_thresholds[feature][0]:
                    logger.debug(
                        f"new point because of {feature} angle : {angle_between}"
                    )
                    return True

            else:
                delta_change = np.abs(last_feat - current_feat)
                if len(delta_change.shape) > 0:
                    for i, dc in enumerate(delta_change):
                        if dc > self._delta_thresholds[feature][i]:
                            logger.debug(f"new point because of {feature} - {dc}")
                            return True
                elif delta_change > self._delta_thresholds[feature]:
                    logger.debug(f"new point because of {feature}")
                    return True
        return False

    def __call__(
        self: Self,
        ctx: TransformContext,  # noqa: ARG002
        payload: Payload,
    ) -> Payload:
        payload.percept = self._filter(payload.percept)
        return payload

    def _filter(self: Self, percept: Message) -> Message:
        """Sets `percept.pass_message` to False if no significant feature change.

        Args:
            percept: Percept to check for significant feature change.

        Returns:
            Percept with `percept.pass_message` ste to False if features haven't
            changed significantly.
        """
        if not percept.pass_message:
            return percept

        if self._last_percept is None:  # first step
            self._last_percept = percept
            self._last_sent_n_steps_ago = 0
            return percept

        significant_feature_change = self._check_feature_change(percept)

        percept.pass_message = significant_feature_change

        if significant_feature_change:
            self._last_percept = percept
            self._last_sent_n_steps_ago = 0
        else:
            self._last_sent_n_steps_ago += 1

        return percept
