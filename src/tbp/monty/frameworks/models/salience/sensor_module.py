# Copyright 2025 Thousand Brains Project
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

from tbp.monty.frameworks.models.abstract_monty_classes import SensorModule
from tbp.monty.frameworks.models.salience.on_object_observation import (
    on_object_observation,
)
from tbp.monty.frameworks.models.salience.return_inhibitor import ReturnInhibitor
from tbp.monty.frameworks.models.salience.strategies import (
    RGBADepthObservation,
    SalienceStrategy,
    UniformSalienceStrategy,
)
from tbp.monty.frameworks.models.states import GoalState


class HabitatSalienceSM(SensorModule):
    def __init__(
        self,
        rng,
        sensor_module_id: str,
        salience_strategy_class: type[SalienceStrategy] = UniformSalienceStrategy,
        salience_strategy_args: dict[str, Any] | None = None,
        return_inhibitor_class: type[ReturnInhibitor] = ReturnInhibitor,
        return_inhibitor_args: dict[str, Any] | None = None,
    ) -> None:
        self._rng = rng
        self._sensor_module_id = sensor_module_id

        salience_strategy_args = (
            dict(salience_strategy_args) if salience_strategy_args else {}
        )
        self._salience_strategy = salience_strategy_class(**salience_strategy_args)

        return_inhibitor_args = (
            dict(return_inhibitor_args) if return_inhibitor_args else {}
        )
        self._return_inhibitor = return_inhibitor_class(**return_inhibitor_args)
        self._goals: list[GoalState] = []

    def state_dict(self):
        """Return a serializable dict with this sensor module's state.

        Includes everything needed to save/load this sensor module.
        """
        pass

    def update_state(self, state):
        pass

    def step(self, data) -> None:
        """Generate goal states for the current step.

        Args:
            data: Raw sensor observations

        """
        salience_map = self._salience_strategy(
            RGBADepthObservation(rgba=data["rgba"], depth=data["depth"])
        )

        on_object = on_object_observation(data, salience_map)
        ior_weights = self._return_inhibitor(
            on_object.center_location, on_object.locations
        )
        salience = self._weight_salience(on_object.salience, ior_weights)

        self._goals = [
            GoalState(
                location=on_object.locations[i],
                morphological_features=None,
                non_morphological_features=None,
                confidence=salience[i],
                use_state=True,
                sender_id=self._sensor_module_id,
                sender_type="SM",
                goal_tolerances=None,
            )
            for i in range(len(on_object.locations))
        ]

    def _weight_salience(
        self,
        salience: np.ndarray,
        ior_weights: np.ndarray,
    ) -> np.ndarray:
        decay_factor = 0.75

        weighted_salience = salience - decay_factor * ior_weights

        randomness_factor = 0.05
        weighted_salience += self._rng.normal(
            loc=0, scale=randomness_factor, size=weighted_salience.shape[0]
        )

        # normalize confidence values
        weighted_salience = (weighted_salience - weighted_salience.min()) / (
            weighted_salience.max() - weighted_salience.min()
        )
        return weighted_salience

    def pre_episode(self):
        """This method is called before each episode."""
        self._goals.clear()
        self._return_inhibitor.reset()

    def propose_goal_states(self) -> list[GoalState]:
        return self._goals
