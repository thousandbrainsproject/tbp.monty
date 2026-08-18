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

from tbp.monty.cmp import Goal
from tbp.monty.frameworks.models.sensor_modules import SnapshotTelemetry
from tbp.monty.memento import Memento

__all__ = ["SalienceSMTelemetry"]


class SalienceSMTelemetry(SnapshotTelemetry):
    """Keeps track of all of SalienceSM's telemetry.

    On top of the raw observation snapshots, optionally records, per step, the
    2D segmentation mask and the region the sensor module proposed from it.
    Everything stored here is JSON-encodable by BufferEncoder (ndarrays and
    Goals), so the state dict rides into the detailed logging stream with no
    special handling.
    """

    def __init__(self, save_segmentation: bool = False) -> None:
        """Initialize the telemetry.

        Args:
            save_segmentation: Whether to record segmentation masks and regions.
        """
        super().__init__()
        self._save_segmentation = save_segmentation
        self.segmentation_maps: list[np.ndarray | None] = []
        self.regions: list[list[Goal]] = []

    def reset(self) -> None:
        """Reset the telemetry."""
        super().reset()
        self.segmentation_maps = []
        self.regions = []

    def record(
        self,
        segmentation_map: np.ndarray | None,
        region: list[Goal],
    ) -> None:
        """Record one step's segmentation mask and region proposal.

        Does nothing unless the telemetry was created with `save_segmentation`.

        Args:
            segmentation_map: The 2D segmentation mask, or None if no
                segmentation strategy ran this step.
            region: The region proposed from the segmentation.
        """
        if not self._save_segmentation:
            return
        self.segmentation_maps.append(segmentation_map)
        self.regions.append(list(region))

    def state_dict(self) -> Memento:
        """Return all recorded telemetry.

        Returns:
            The raw observation snapshots (see `SnapshotTelemetry.state_dict`),
            plus a list of segmentation masks in `segmentation_maps` and a list
            of proposed regions in `regions`, one entry per recorded step.
        """
        return dict(
            **super().state_dict(),
            segmentation_maps=self.segmentation_maps,
            regions=self.regions,
        )
