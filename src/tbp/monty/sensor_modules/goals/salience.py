# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import cast

import numpy as np
from typing_extensions import Self

from tbp.monty.cmp import Goal
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.salience.on_object_observation import (
    on_object_observation,
)
from tbp.monty.frameworks.models.salience.return_inhibitor import ReturnInhibitor
from tbp.monty.frameworks.models.salience.strategies.protocol import SalienceStrategy
from tbp.monty.frameworks.models.salience.strategies.uniform import Uniform
from tbp.monty.sensor_modules.sensor_module import Payload, Transform, TransformContext


class Salience(Transform):
    _return_inhibitor: ReturnInhibitor
    _salience_strategy: SalienceStrategy
    _sensor_module_id: str

    def __init__(
        self: Self,
        sensor_module_id: str,
        salience_strategy: SalienceStrategy | None = None,
        return_inhibitor: ReturnInhibitor | None = None,
    ) -> None:
        self._sensor_module_id = sensor_module_id
        self._salience_strategy = (
            Uniform() if salience_strategy is None else salience_strategy
        )
        self._return_inhibitor = (
            ReturnInhibitor() if return_inhibitor is None else return_inhibitor
        )

    def __call__(self: Self, ctx: TransformContext, payload: Payload) -> Payload:
        salience_map = self._salience_strategy(
            # TransformContext is a superset of RuntimeContext
            ctx=cast("RuntimeContext", ctx),
            rgba=payload.observation["rgba"],
            depth=payload.observation["depth"],
        )

        on_object = on_object_observation(payload.observation, salience_map)
        ior_weights = self._return_inhibitor(
            on_object.center_location, on_object.locations
        )
        salience = self._weight_salience(ctx, on_object.salience, ior_weights)

        payload.goals.extend(
            [
                Goal(
                    location=on_object.locations[i],
                    morphological_features=None,
                    non_morphological_features=None,
                    confidence=salience[i],
                    # SalienceSM goals are intended for the motor system
                    pass_message=False,
                    sender_id=self._sensor_module_id,
                    sender_type="SM",
                    process_features_in_lm=False,
                    goal_tolerances=None,
                )
                for i in range(len(on_object.locations))
            ]
        )

        return payload

    def _weight_salience(
        self,
        ctx: TransformContext,
        salience: np.ndarray,
        ior_weights: np.ndarray,
    ) -> np.ndarray:
        weighted_salience = self._decay_salience(salience, ior_weights)

        weighted_salience = self._randomize_salience(ctx, weighted_salience)

        return self._normalize_salience(weighted_salience)

    def _decay_salience(
        self, salience: np.ndarray, ior_weights: np.ndarray
    ) -> np.ndarray:
        decay_factor = 0.75
        return salience - decay_factor * ior_weights

    def _randomize_salience(
        self: Self, ctx: TransformContext, weighted_salience: np.ndarray
    ) -> np.ndarray:
        randomness_factor = 0.05
        weighted_salience += ctx.rng.normal(
            loc=0, scale=randomness_factor, size=weighted_salience.shape[0]
        )
        return weighted_salience

    def _normalize_salience(self, weighted_salience: np.ndarray) -> np.ndarray:
        if weighted_salience.size == 0:
            return weighted_salience

        min_ = weighted_salience.min()
        max_ = weighted_salience.max()
        scale = max_ - min_
        if np.isclose(scale, 0):
            return np.clip(weighted_salience, 0, 1)

        return (weighted_salience - min_) / scale
