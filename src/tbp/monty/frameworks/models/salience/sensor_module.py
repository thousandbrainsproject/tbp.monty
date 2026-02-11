# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import numpy as np
import quaternion as qt

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.abstract_monty_classes import (
    SensorModule,
)
from tbp.monty.frameworks.models.motor_system_state import AgentState, SensorState
from tbp.monty.frameworks.models.salience.on_object_observation import (
    on_object_observation,
)
from tbp.monty.frameworks.models.salience.return_inhibitor import ReturnInhibitor
from tbp.monty.frameworks.models.salience.strategies import (
    SalienceStrategy,
    UniformSalienceStrategy,
)
from tbp.monty.frameworks.models.sensor_modules import SnapshotTelemetry
from tbp.monty.frameworks.models.states import GoalState, State
from tbp.monty.frameworks.sensors import SensorID

__all__ = ["HabitatSalienceSM"]


class HabitatSalienceSM(SensorModule):
    def __init__(
        self,
        rng: np.random.RandomState,
        sensor_module_id: str,
        save_raw_obs: bool = False,
        salience_strategy: SalienceStrategy | None = None,
        return_inhibitor: ReturnInhibitor | None = None,
        snapshot_telemetry: SnapshotTelemetry | None = None,
    ) -> None:
        self._rng = rng
        self._sensor_module_id = sensor_module_id
        self._save_raw_obs = save_raw_obs
        self._salience_strategy = (
            UniformSalienceStrategy()
            if salience_strategy is None
            else salience_strategy
        )
        self._return_inhibitor = (
            ReturnInhibitor() if return_inhibitor is None else return_inhibitor
        )
        self._snapshot_telemetry = (
            SnapshotTelemetry() if snapshot_telemetry is None else snapshot_telemetry
        )

        self._goals: list[GoalState] = []
        # TODO: Goes away once experiment code is extracted
        self.is_exploring = False

    @property
    def sensor_module_id(self) -> str:
        return self._sensor_module_id

    def state_dict(self):
        return self._snapshot_telemetry.state_dict()

    def update_state(self, agent: AgentState):
        """Update information about the sensor's location and rotation."""
        sensor = agent.sensors[SensorID(self.sensor_module_id + ".rgba")]
        self.state = SensorState(
            position=agent.position
            + qt.rotate_vectors(agent.rotation, sensor.position),
            rotation=agent.rotation * sensor.rotation,
        )
        self.motor_only_step = agent.motor_only_step

    def step(self, ctx: RuntimeContext, data) -> State | None:
        """Generate goal states for the current step.

        Args:
            ctx: The runtime context.
            data: Raw sensor observations

        Returns:
            A Percept, if one is generated.
        """
        if self._save_raw_obs and not self.is_exploring:
            self._snapshot_telemetry.raw_observation(
                data, self.state.rotation, self.state.position
            )

        salience_map = self._salience_strategy(rgba=data["rgba"], depth=data["depth"])

        on_object = on_object_observation(data, salience_map)
        ior_weights = self._return_inhibitor(
            on_object.center_location, on_object.locations
        )
        salience = self._weight_salience(ctx, on_object.salience, ior_weights)

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

        return None

    def _weight_salience(
        self,
        ctx: RuntimeContext,
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
        self, ctx: RuntimeContext, weighted_salience: np.ndarray
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

    def pre_episode(self) -> None:
        """This method is called before each episode."""
        self._goals.clear()
        self._return_inhibitor.reset()
        self._snapshot_telemetry.reset()
        self.is_exploring = False

    def propose_goal_states(self) -> list[GoalState]:
        return self._goals
