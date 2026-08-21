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
import quaternion as qt

from tbp.monty.cmp import AttentionWeight
from tbp.monty.frameworks.models.abstract_monty_classes import SensorObservation
from tbp.monty.memento import Memento

__all__ = ["SalienceSMTelemetry"]


class SalienceSMTelemetry:
    """Keeps track of all of SalienceSM's telemetry.

    Records per step: raw observation snapshots with their poses, the 2D
    salience map, and the 2D segmentation mask with the region proposed from
    it. Whether anything is recorded at all is the sensor module's decision
    (its `save_raw_obs` switch).
    Everything stored here is JSON-encodable by BufferEncoder, so the state
    dict rides into the detailed logging stream with no special handling.
    """

    def __init__(self) -> None:
        self.raw_observations: list[SensorObservation] = []
        self.poses: list[dict[str, np.ndarray]] = []
        self.salience_maps: list[np.ndarray] = []
        self.segmentation_maps: list[np.ndarray | None] = []
        self.regions: list[list[AttentionWeight]] = []

    def reset(self) -> None:
        """Reset the telemetry."""
        self.raw_observations = []
        self.poses = []
        self.salience_maps = []
        self.segmentation_maps = []
        self.regions = []

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

    def salience(self, salience_map: np.ndarray) -> None:
        """Record one step's salience map.

        Args:
            salience_map: The 2D salience map.
        """
        self.salience_maps.append(salience_map)

    def segmentation(
        self,
        segmentation_map: np.ndarray | None,
        region: list[AttentionWeight],
    ) -> None:
        """Record one step's segmentation mask and region proposal.

        Args:
            segmentation_map: The 2D segmentation mask, or None if no
                segmentation strategy ran this step.
            region: The region proposed from the segmentation.
        """
        self.segmentation_maps.append(segmentation_map)
        self.regions.append(list(region))

    def state_dict(self) -> Memento:
        """Return all recorded telemetry.

        Every stream's key is always present, empty when nothing was
        recorded, so consumers (the detailed logger most of all) can index
        them without special-casing the configuration.

        Returns:
            Raw observations in `raw_observations` with poses in
            `sm_properties`, salience maps in `salience_maps`, and segmentation
            masks in `segmentation_maps` with proposed regions in `regions`.
        """
        return dict(
            raw_observations=self.raw_observations,
            sm_properties=self.poses,
            salience_maps=self.salience_maps,
            segmentation_maps=self.segmentation_maps,
            regions=self.regions,
        )
