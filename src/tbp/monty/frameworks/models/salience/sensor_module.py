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

from tbp.monty.cmp import AttentionWeight, Goal
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.abstract_monty_classes import (
    SensorModule,
    SensorObservation,
)
from tbp.monty.frameworks.models.motor_system_state import AgentState, SensorState
from tbp.monty.frameworks.models.salience.on_object_observation import (
    OnObjectObservation,
    on_object_observation,
)
from tbp.monty.frameworks.models.salience.return_inhibitor import ReturnInhibitor
from tbp.monty.frameworks.models.salience.segmentation.protocol import (
    SegmentationStrategy,
)
from tbp.monty.frameworks.models.salience.strategies import (
    SalienceStrategy,
    Uniform,
)
from tbp.monty.frameworks.models.salience.telemetry import SalienceSMTelemetry
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.memento import Memento

__all__ = ["SalienceSM"]


class SalienceSM(SensorModule):
    def __init__(
        self,
        sensor_module_id: str,
        save_raw_obs: bool = False,
        salience_strategy: SalienceStrategy | None = None,
        return_inhibitor: ReturnInhibitor | None = None,
        snapshot_telemetry: SalienceSMTelemetry | None = None,
        segmentation_strategy: SegmentationStrategy | None = None,
    ) -> None:
        self._sensor_module_id = sensor_module_id
        self._save_raw_obs = save_raw_obs
        self._salience_strategy = (
            Uniform() if salience_strategy is None else salience_strategy
        )
        self._return_inhibitor = (
            ReturnInhibitor() if return_inhibitor is None else return_inhibitor
        )
        self._snapshot_telemetry = (
            SalienceSMTelemetry() if snapshot_telemetry is None else snapshot_telemetry
        )

        self._segmentation_strategy = segmentation_strategy

        self._goals: list[Goal] = []
        self._region: list[AttentionWeight] = []
        # TODO: Goes away once experiment code is extracted
        self.is_exploring = False

    @property
    def sensor_module_id(self) -> str:
        return self._sensor_module_id

    def state_dict(self) -> Memento:
        return self._snapshot_telemetry.state_dict()

    def update_state(self, agent: AgentState) -> None:
        """Update information about the sensor's location and rotation."""
        sensor = agent.sensors[SensorID(self.sensor_module_id)]
        self.state = SensorState(
            position=agent.position
            + qt.rotate_vectors(agent.rotation, sensor.position),
            rotation=agent.rotation * sensor.rotation,
        )

    def propose_region(self) -> list[AttentionWeight]:
        return self._region

    def step(
        self,
        ctx: RuntimeContext,
        observation: SensorObservation,
        motor_only_step: bool = False,
    ) -> None:
        """Generate goal for the current step.

        If `motor_only_step` is True, this method will return without using the
        salience strategy, stepping the return inhibitor, or modifying `self._goals`
        in any way.

        Args:
            ctx: The runtime context.
            observation: Sensor observation.
            motor_only_step: Whether the current step is a motor-only step.

        """
        if motor_only_step:
            return

        salience_map = self._salience_strategy(
            ctx=ctx, rgba=observation["rgba"], depth=observation["depth"]
        )

        on_object = on_object_observation(observation, salience_map)
        ior_weights = self._return_inhibitor(
            on_object.center_location, on_object.locations
        )
        salience = self._weight_salience(ctx, on_object.salience, ior_weights)

        self._goals = [
            Goal(
                location=on_object.locations[i],
                morphological_features=None,
                non_morphological_features=None,
                confidence=salience[i],
                use_state=False,  # SalienceSM goals are intended for the motor system
                sender_id=self._sensor_module_id,
                sender_type="SM",
                goal_tolerances=None,
            )
            for i in range(len(on_object.locations))
        ]

        segmentation_map, self._region = self._segment_region(
            ctx, observation, on_object, salience
        )

        if not self.is_exploring:
            if self._save_raw_obs:
                self._snapshot_telemetry.raw_observation(
                    observation, self.state.rotation, self.state.position
                )
            self._snapshot_telemetry.record(segmentation_map, self._region)

    def _segment_region(
        self,
        ctx: RuntimeContext,
        observation: SensorObservation,
        on_object: OnObjectObservation,
        salience: np.ndarray,
    ) -> tuple[np.ndarray | None, list[AttentionWeight]]:
        """Segment the surface under fixation into a region proposal.

        The region is the set of on-object locations inside the segmented
        surface, expressed as goals so it can travel to the attention system via
        ``propose_region``.

        Args:
            ctx: The runtime context.
            observation: Sensor observation.
            on_object: The on-object view of the observation.
            salience: Weighted salience, aligned with ``on_object.locations``.

        Returns:
            The segmentation mask and the region's goals; None and an empty
            list without a segmentation strategy.
        """
        if self._segmentation_strategy is None:
            return None, []

        segmentation_map = self._segmentation_strategy(
            ctx=ctx, rgba=observation["rgba"], depth=observation["depth"]
        )

        # Restore the weighted salience to image shape; boolean-mask indexing and
        # np.where enumerate pixels in the same row-major order.
        salience_map = np.zeros(on_object.on_object_mask.shape)
        salience_map[on_object.on_object_mask] = salience

        surface_mask = segmentation_map.astype(bool) & on_object.on_object_mask
        surface_locations = on_object.locations_map[surface_mask]
        surface_salience = salience_map[surface_mask]

        return segmentation_map, [
            AttentionWeight(
                location=surface_locations[i],
                weight=1,
                sender_id=self._sensor_module_id,
                sender_type="SM",
            )
            for i in range(len(surface_locations))
        ]

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

    def reset(self) -> None:
        self._goals.clear()
        self._region.clear()
        self._return_inhibitor.reset()
        self._snapshot_telemetry.reset()
        self.is_exploring = False

    def propose_goals(self) -> list[Goal]:
        return self._goals
