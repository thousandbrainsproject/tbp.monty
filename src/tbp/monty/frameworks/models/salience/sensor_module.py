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

from tbp.monty.frameworks.models.abstract_monty_classes import SensorModule
from tbp.monty.frameworks.models.salience.goal_generator import HabitatGoalGenerator
from tbp.monty.frameworks.models.states import GoalState, State


class HabitatSalienceSM(SensorModule):
    def __init__(
        self,
        rng,
        sensor_module_id: str,
        goal_generator_class: type[HabitatGoalGenerator] = HabitatGoalGenerator,
        goal_generator_args: dict[str, Any] | None = None,
    ) -> None:
        self._rng = rng
        self._sensor_module_id = sensor_module_id

        goal_generator_args = dict(goal_generator_args) if goal_generator_args else {}
        self._goal_generator = goal_generator_class(
            rng=self._rng, **goal_generator_args
        )

        self._goals: list[GoalState] = []

    def state_dict(self):
        """Return a serializable dict with this sensor module's state.

        Includes everything needed to save/load this sensor module.
        """
        pass

    def update_state(self, state):
        pass

    # TODO: Need to update all sensor modules to return State | None
    #       and ensure the framework handles it appropriately.
    def step(self, data) -> State | None:
        self._goals = self._goal_generator(self._sensor_module_id, data)
        return None

    def pre_episode(self):
        """This method is called before each episode."""
        self._goal_generator.reset()
        self._goals.clear()

    def propose_goal_states(self) -> list[GoalState]:
        return self._goals
