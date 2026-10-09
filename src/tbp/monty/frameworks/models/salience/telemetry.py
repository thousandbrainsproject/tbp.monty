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
import numpy.typing as npt
import quaternion as qt

from tbp.monty.frameworks.models.abstract_monty_classes import SensorObservation
from tbp.monty.memento import Memento

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tbp.monty.cmp import AttentionRegion, Goal

__all__ = [
    "DetailedSalienceSMTelemetry",
    "NoopSalienceSMTelemetry",
    "SalienceSMTelemetry",
]


class SalienceSMTelemetry(Protocol):
    def reset(self) -> None: ...

    def raw_observation(
        self,
        raw_observation: SensorObservation,
        rotation: qt.quaternion,
        position: npt.NDArray[np.float64],
    ) -> None: ...

    def salience_map(self, salience_map: npt.NDArray[np.float64]) -> None: ...

    def segmentation_map(self, segmentation_map: npt.NDArray[np.uint8]) -> None: ...

    def goals(self, goals: Sequence[Goal]) -> None: ...

    def attention_region(self, region: AttentionRegion) -> None: ...

    def state_dict(self) -> Memento: ...


class NoopSalienceSMTelemetry(SalienceSMTelemetry):
    def reset(self) -> None:
        pass

    def raw_observation(
        self,
        raw_observation: SensorObservation,
        rotation: qt.quaternion,
        position: npt.NDArray[np.float64],
    ) -> None:
        pass

    def salience_map(self, salience_map: npt.NDArray[np.float64]) -> None:
        pass

    def segmentation_map(self, segmentation_map: npt.NDArray[np.uint8]) -> None:
        pass

    def goals(self, goals: Sequence[Goal]) -> None:
        pass

    def attention_region(self, region: AttentionRegion) -> None:
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


class DetailedSalienceSMTelemetry(SalienceSMTelemetry):
    """Keeps track of all of SalienceSM's telemetry.

    Records per step: raw observation snapshots with their poses, the 2D
    salience map, the 2D segmentation mask, the goals proposed and the attention region
    proposed from the mask.

    Everything stored here is JSON-encodable by BufferEncoder, so the state
    dict rides into the detailed logging stream with no special handling.
    """

    _raw_observations: list[SensorObservation]
    _poses: list[dict[str, npt.NDArray[np.float64]]]
    _salience_maps: list[npt.NDArray[np.float64]]
    _segmentation_maps: list[npt.NDArray[np.uint8] | None]
    _goals: list[Sequence[Goal]]
    _attention_regions: list[AttentionRegion]

    def __init__(self) -> None:
        self._raw_observations = []
        self._poses = []
        self._salience_maps = []
        self._segmentation_maps = []
        self._goals = []
        self._attention_regions = []

    def reset(self) -> None:
        """Reset the telemetry."""
        self._raw_observations = []
        self._poses = []
        self._salience_maps = []
        self._segmentation_maps = []
        self._goals = []
        self._attention_regions = []

    def raw_observation(
        self,
        raw_observation: SensorObservation,
        rotation: qt.quaternion,
        position: npt.NDArray[np.float64],
    ) -> None:
        """Record a snapshot of a raw observation and its pose information.

        Args:
            raw_observation: Raw observation.
            rotation: Rotation of the sensor.
            position: Position of the sensor.
        """
        self._raw_observations.append(raw_observation)
        self._poses.append(
            dict(
                sm_rotation=qt.as_float_array(rotation),
                sm_location=np.array(position),
            )
        )

    def salience_map(self, salience_map: npt.NDArray[np.float64]) -> None:
        """Record one step's salience map.

        Args:
            salience_map: The 2D salience map.
        """
        self._salience_maps.append(salience_map)

    def segmentation_map(self, segmentation_map: npt.NDArray[np.uint8]) -> None:
        """Record one step's segmentation mask.

        Args:
            segmentation_map: The 2D segmentation mask.
        """
        self._segmentation_maps.append(segmentation_map)

    def goals(self, goals: Sequence[Goal]) -> None:
        """Record one step's proposed goals.

        Args:
            goals: The goals the sensor module proposed this step.
        """
        self._goals.append(goals)

    def attention_region(self, region: AttentionRegion) -> None:
        """Record the attention region proposed this step.

        Args:
            region: The proposed region.
        """
        self._attention_regions.append(region)

    def state_dict(self) -> Memento:
        """Return all recorded telemetry.

        Returns:
            Raw observations in `raw_observations` with poses in
            `sm_properties`, salience maps in `salience_maps`, segmentation
            masks in `segmentation_maps`, goal columns in `goals`, and the
            proposed regions in `attention_regions`.
        """
        return dict(
            goals=self._goals,
            attention_regions=self._attention_regions,
            raw_observations=self._raw_observations,
            sm_properties=self._poses,
            salience_maps=self._salience_maps,
            segmentation_maps=self._segmentation_maps,
        )
