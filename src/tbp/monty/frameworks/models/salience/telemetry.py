# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
import quaternion as qt

from tbp.monty.cmp import goals_to_columns
from tbp.monty.frameworks.models.abstract_monty_classes import SensorObservation
from tbp.monty.memento import Memento

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tbp.monty.cmp import AttentionRegion, Goal

__all__ = [
    "NoopSalienceSMTelemetry",
    "SalienceSMTelemetry",
    "SalienceSMTelemetryProtocol",
]


class SalienceSMTelemetryProtocol(Protocol):
    def reset(self) -> None: ...

    def raw_observation(
        self,
        raw_observation: SensorObservation,
        rotation: qt.quaternion,
        position: np.ndarray,
    ) -> None: ...

    def salience_map(self, salience_map: np.ndarray) -> None: ...

    def segmentation_map(self, segmentation_map: np.ndarray | None) -> None: ...

    def goals(self, goals: Sequence[Goal]) -> None: ...

    def attention_region(self, region: AttentionRegion | None) -> None: ...

    def state_dict(self) -> Memento: ...


class NoopSalienceSMTelemetry(SalienceSMTelemetryProtocol):
    def reset(self) -> None:
        pass

    def raw_observation(
        self,
        raw_observation: SensorObservation,
        rotation: qt.quaternion,
        position: np.ndarray,
    ) -> None:
        pass

    def salience_map(self, salience_map: np.ndarray) -> None:
        pass

    def segmentation_map(self, segmentation_map: np.ndarray | None) -> None:
        pass

    def goals(self, goals: Sequence[Goal]) -> None:
        pass

    def attention_region(self, region: AttentionRegion | None) -> None:
        pass

    def state_dict(self) -> Memento:
        # The empty schema, so consumers indexing these keys stay simple.
        return dict(
            raw_observations=[],
            sm_properties=[],
            salience_maps=[],
            segmentation_maps=[],
            goals=[],
            attention_regions=[],
        )


class SalienceSMTelemetry(SalienceSMTelemetryProtocol):
    """Keeps track of all of SalienceSM's telemetry.

    Records per step: raw observation snapshots with their poses, the 2D
    salience map, the 2D segmentation mask, the goals proposed (as columns, see
    :func:`~tbp.monty.cmp.goals_to_columns`) and the attention region
    proposed from the mask. Whether anything is recorded at all is the sensor module's
    decision (its `save_raw_obs` switch).
    Everything stored here is JSON-encodable by BufferEncoder, so the state
    dict rides into the detailed logging stream with no special handling.
    """

    def __init__(self) -> None:
        self.raw_observations: list[SensorObservation] = []
        self.poses: list[dict[str, np.ndarray]] = []
        self.salience_maps: list[np.ndarray] = []
        self.segmentation_maps: list[np.ndarray | None] = []
        self._goals: list[dict[str, np.ndarray]] = []
        self._attention_regions: list[AttentionRegion | None] = []

    def reset(self) -> None:
        """Reset the telemetry."""
        self.raw_observations = []
        self.poses = []
        self.salience_maps = []
        self.segmentation_maps = []
        self._goals = []
        self._attention_regions = []

    def raw_observation(
        self,
        raw_observation: SensorObservation,
        rotation: qt.quaternion,
        position: np.ndarray,
    ):
        """Record a snapshot of a raw observation and its pose information.

        Args:
            raw_observation: Raw observation.
            rotation: Rotation of the sensor.
            position: Position of the sensor.
        """
        self.raw_observations.append(raw_observation)
        self.poses.append(
            dict(
                sm_rotation=qt.as_float_array(rotation),
                sm_location=np.array(position),
            )
        )

    def salience_map(self, salience_map: np.ndarray) -> None:
        """Record one step's salience map.

        Args:
            salience_map: The 2D salience map.
        """
        self.salience_maps.append(salience_map)

    def segmentation_map(self, segmentation_map: np.ndarray | None) -> None:
        """Record one step's segmentation mask.

        Args:
            segmentation_map: The 2D segmentation mask, or None if no
                segmentation strategy ran this step.
        """
        self.segmentation_maps.append(segmentation_map)

    def goals(self, goals: Sequence[Goal]) -> None:
        """Record one step's proposed goals as columns.

        Args:
            goals: The goals the sensor module proposed this step.
        """
        self._goals.append(goals_to_columns(goals))

    def attention_region(self, region: AttentionRegion | None) -> None:
        """Record the attention region proposed this step.

        Args:
            region: The region, or None when the sensor module proposed none.
        """
        self._attention_regions.append(region)

    def state_dict(self) -> Memento:
        """Return all recorded telemetry.

        Every stream's key is always present, empty when nothing was
        recorded, so consumers (the detailed logger most of all) can index
        them without special-casing the configuration.

        Returns:
            Raw observations in `raw_observations` with poses in
            `sm_properties`, salience maps in `salience_maps`, segmentation
            masks in `segmentation_maps`, goal columns in `goals`, and the
            proposed regions in `attention_regions`.
        """
        return dict(
            goals=self._goals,
            attention_regions=self._attention_regions,
            raw_observations=self.raw_observations,
            sm_properties=self.poses,
            salience_maps=self.salience_maps,
            segmentation_maps=self.segmentation_maps,
        )
