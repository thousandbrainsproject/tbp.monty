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
import numpy.typing as npt
import quaternion as qt

from tbp.monty.cmp import AttentionRegion, Goal
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.abstract_monty_classes import (
    SensorModule,
    SensorObservation,
)
from tbp.monty.frameworks.models.motor_system_state import AgentState, SensorState
from tbp.monty.frameworks.models.salience.on_object_observation import (
    on_object_observation,
)
from tbp.monty.frameworks.models.salience.return_inhibitor import ReturnInhibitor
from tbp.monty.frameworks.models.salience.segmentation.strategy import (
    SegmentationStrategy,
)
from tbp.monty.frameworks.models.salience.strategies import (
    SalienceStrategy,
    Uniform,
)
from tbp.monty.frameworks.models.salience.telemetry import (
    NoopSalienceSMTelemetry,
    SalienceSMTelemetry,
)
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.memento import Memento

__all__ = ["SalienceSM"]


class SalienceSM(SensorModule):
    _sensor_module_id: str
    _salience_strategy: SalienceStrategy
    _return_inhibitor: ReturnInhibitor
    _snapshot_telemetry: SalienceSMTelemetry
    _goals: list[Goal]
    is_exploring: bool
    _segmentation_strategy: SegmentationStrategy | None
    _region: AttentionRegion

    def __init__(
        self,
        sensor_module_id: str,
        salience_strategy: SalienceStrategy | None = None,
        return_inhibitor: ReturnInhibitor | None = None,
        snapshot_telemetry: SalienceSMTelemetry | None = None,
        segmentation_strategy: SegmentationStrategy | None = None,
    ) -> None:
        self._sensor_module_id = sensor_module_id
        self._salience_strategy = (
            Uniform() if salience_strategy is None else salience_strategy
        )
        self._return_inhibitor = (
            ReturnInhibitor() if return_inhibitor is None else return_inhibitor
        )
        self._snapshot_telemetry = (
            NoopSalienceSMTelemetry()
            if snapshot_telemetry is None
            else snapshot_telemetry
        )

        self._goals = []
        # TODO: Goes away once experiment code is extracted
        self.is_exploring = False
        self._segmentation_strategy = segmentation_strategy
        self._region = AttentionRegion.empty()

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
                # SalienceSM goals are intended for the motor system
                pass_message=False,
                sender_id=self._sensor_module_id,
                sender_type="SM",
                process_features_in_lm=False,
                goal_tolerances=None,
            )
            for i in range(len(on_object.locations))
        ]

        self._region = self._segment_region(
            ctx=ctx,
            rgba=observation["rgba"],
            on_object_map=on_object.on_object_map,
            location_map=on_object.location_map,
        )

        if not self.is_exploring:
            self._snapshot_telemetry.raw_observation(
                observation, self.state.rotation, self.state.position
            )

    def _segment_region(
        self,
        ctx: RuntimeContext,
        rgba: npt.NDArray[np.uint8],
        on_object_map: npt.NDArray[np.bool_],
        location_map: npt.NDArray[np.float64],
    ) -> AttentionRegion:
        """Segment the surface under fixation into a region proposal.

        The region is the set of on-object locations inside the segmented
        surface, expressed as attention weights so it can travel to the
        attention system via ``propose_region``.

        Args:
            ctx: The runtime context.
            rgba: The RGB image from the sensor.
            on_object_map: The on-object view of the observation as a boolean mask.
            location_map: The corresponding 3D locations for each pixel in the
                observation.

        Returns:
            The region it proposes; an empty region if there is no segmentation
                strategy.
        """
        if self._segmentation_strategy is None:
            return AttentionRegion.empty()

        segmentation_map = self._segmentation_strategy(ctx=ctx, rgba=rgba)

        region_on_object_map = segmentation_map.astype(bool) & on_object_map
        region_locations_on_object = location_map[region_on_object_map]

        region = AttentionRegion.uniform(
            region_locations_on_object, AttentionRegion.MAX_WEIGHT
        )

        if not self.is_exploring:
            self._snapshot_telemetry.segmentation_map(segmentation_map)
            self._snapshot_telemetry.attention_region(region)

        return region

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
        self._return_inhibitor.reset()
        self._snapshot_telemetry.reset()
        self.is_exploring = False

    def propose_goals(self) -> list[Goal]:
        return self._goals
