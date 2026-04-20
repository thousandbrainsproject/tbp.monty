# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, cast

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.utils.logging_utils import (
    compute_pose_error,
    compute_pose_errors,
)


@dataclass
class HypothesesUpdaterGraphTelemetry:
    """Telemetry from HypothesesUpdater for a single graph."""

    hypotheses_updater: dict[str, Any]
    """Any telemetry from the hypotheses updater."""

    evidence: npt.NDArray[np.float64]
    """The hypotheses evidence scores."""

    rotations: npt.NDArray[np.float64]
    """Rotations of the hypotheses."""

    locations: npt.NDArray[np.float64]
    """Locations of the hypotheses."""

    pose_errors: npt.NDArray[np.float64]
    """Rotation errors relative to the target pose."""


HypothesesUpdaterTelemetry = Dict[str, HypothesesUpdaterGraphTelemetry]
"""HypothesesUpdaterGraphTelemetry indexed by graph ID."""


class TheoreticalLimitLMLoggingMixin:
    """Mixin that adds theoretical limit and pose error logging for learning modules.

    This mixin augments the learning module with methods to compute and log:
      - The maximum evidence score for each object.
      - The theoretical lower bound of pose error on the target object, assuming
        Monty had selected the best possible hypothesis (oracle performance).
      - The actual pose error of the most likely hypothesis (MLH) on the target object.

    These metrics are useful for analyzing the performance gap between the model's
    current inference and its best achievable potential given its internal hypotheses.

    Compatible with:
        - EvidenceGraphLM
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Ensure the mixin is used only with compatible learning modules.

        Raises:
            TypeError: If the mixin is used with a non-compatible learning module.
        """
        super().__init_subclass__(**kwargs)
        if not any(issubclass(b, (EvidenceGraphLM)) for b in cls.__bases__):
            raise TypeError(
                "TheoreticalLimitLMLoggingMixin must be mixed in with a subclass of "
                f"EvidenceGraphLM, got {cls.__bases__}"
            )

    def _add_detailed_stats(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Add detailed statistics to the logging dictionary.

        This includes metrics like the max evidence score per object, the theoretical
        limit of Monty (i.e., pose error of Monty's best potential hypothesis on the
        target object), and the pose error of the MLH hypothesis on the target object.

        Args:
            stats: The existing statistics dictionary to augment.

        Returns:
            Updated statistics dictionary.
        """
        assert isinstance(self, EvidenceGraphLM)

        stats["max_evidence"] = {k: max(v) for k, v in self.evidence.items() if len(v)}
        stats["target_object_theoretical_limit"] = (
            self._theoretical_limit_target_object_pose_error()
        )
        stats["target_object_pose_error"] = self._mlh_target_object_pose_error()
        hypotheses_updater_telemetry = self._hypotheses_updater_telemetry()
        if hypotheses_updater_telemetry:
            stats["hypotheses_updater_telemetry"] = hypotheses_updater_telemetry
        return stats

    def _hypotheses_updater_telemetry(self) -> HypothesesUpdaterTelemetry:
        """Returns HypothesesUpdaterTelemetry for all objects."""
        assert isinstance(self, EvidenceGraphLM)

        stats: HypothesesUpdaterTelemetry = {}
        for graph_id, graph_telemetry in self.hypotheses_updater_telemetry.items():
            stats[graph_id] = self._graph_telemetry(graph_id, graph_telemetry)
        return stats

    def _graph_telemetry(
        self, graph_id: str, graph_telemetry: dict[str, Any]
    ) -> HypothesesUpdaterGraphTelemetry:
        """Assemble telemetry for a specific graph ID.

        Args:
            graph_id: The graph ID.
            graph_telemetry: Telemetry from the hypotheses updater for this graph.

        Returns:
            HypothesesUpdaterGraphTelemetry for the given graph ID.
        """
        assert isinstance(self, EvidenceGraphLM)

        evidence = self.evidence[graph_id]
        locations = self.possible_locations[graph_id]
        poses = self.possible_poses[graph_id]

        if len(evidence) == 0:
            return HypothesesUpdaterGraphTelemetry(
                hypotheses_updater=graph_telemetry.copy(),
                evidence=np.empty(shape=(0,), dtype=np.float64),
                rotations=np.empty(shape=(0, 3), dtype=np.float64),
                locations=np.empty(shape=(0, 3), dtype=np.float64),
                pose_errors=np.empty(shape=(0,), dtype=np.float64),
            )

        rotations_inv = Rotation.from_matrix(poses).inv()

        return HypothesesUpdaterGraphTelemetry(
            hypotheses_updater=graph_telemetry.copy(),
            evidence=evidence,
            rotations=rotations_inv.as_euler("xyz", degrees=True),
            locations=locations,
            pose_errors=cast(
                "npt.NDArray[np.float64]",
                compute_pose_errors(
                    rotations_inv,
                    Rotation.from_quat(self.primary_target_rotation_quat),
                ),
            ),
        )

    def _theoretical_limit_target_object_pose_error(self) -> float:
        """Compute the theoretical minimum rotation error on the target object.

        This considers all possible hypotheses rotations on the target object
        and compares them to the target's rotation. The theoretical limit conveys the
        best achievable performance if Monty selects the best hypothesis as its most
        likely hypothesis (MLH).

        Note that having a low pose error for the theoretical limit may not be
        sufficient to decide on the quality of the hypothesis. Although good
        hypotheses generally correlate with a good theoretical limit, the rotation
        error can be small (i.e., low geodesic distance to the ground-truth
        rotation) while the hypothesis is at a different location of the object.

        Returns:
            The minimum achievable rotation error (in radians).
        """
        assert isinstance(self, EvidenceGraphLM)

        object_possible_poses = self.possible_poses[self.primary_target]
        if not len(object_possible_poses):
            return -1

        hyp_rotations = Rotation.from_matrix(object_possible_poses).inv()

        target_rotation = Rotation.from_quat(self.primary_target_rotation_quat)
        return compute_pose_error(hyp_rotations, target_rotation)

    def _mlh_target_object_pose_error(self) -> float:
        """Compute the actual rotation error between predicted and target pose.

        This compares the most likely hypothesis pose (based on evidence) on the target
        object with the ground truth rotation of the target object.

        Returns:
            The rotation error (in radians).
        """
        assert isinstance(self, EvidenceGraphLM)

        obj_rotation = self.get_mlh_for_object(self.primary_target)["rotation"].inv()
        target_rotation = Rotation.from_quat(self.primary_target_rotation_quat)
        return compute_pose_error(obj_rotation, target_rotation)
