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

from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.evidence_matching.resampling_hypotheses_updater import (  # noqa: E501
    ChannelResamplingStats,
)
from tbp.monty.frameworks.utils.logging_utils import compute_pose_error


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

        This includes metrics like the resampling stats, max evidence score per object,
        the theoretical limit of Monty (i.e., pose error of Monty's best potential
        hypothesis on the target object) , and the pose error of the MLH hypothesis
        on the target object.

        Args:
            stats: The existing statistics dictionary to augment.

        Returns:
            Updated statistics dictionary.
        """
        stats["max_evidence"] = {k: max(v) for k, v in self.evidence.items()}
        stats["target_object_theoretical_limit"] = (
            self._theoretical_limit_target_object_pose_error()
        )
        stats["target_object_pose_error"] = self._mlh_target_object_pose_error()

        if (
            hasattr(self.hypotheses_updater, "save_stats")
            and self.hypotheses_updater.save_stats
        ):
            stats["resampling_stats"] = self._resampling_stats()

        return stats

    def _resampling_stats(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Compile resampling statistics across all objects and input channels.

        Iterates over each graph and associated channels, and gathers detailed
        resampling statistics by calling `_channel_resampling_stats`.

        Returns:
            A nested dictionary mapping with keys:
                graph_id -> input_channel -> resampling_statistic.
        """
        stats = {}
        for graph_id, graph_stats in self.hypotheses_updater.resampling_stats.items():
            stats[graph_id] = {
                channel_stats.input_channel: self._channel_resampling_stats(
                    graph_id, channel_stats
                )
                for channel_stats in graph_stats
            }
        return stats

    def _channel_resampling_stats(
        self, graph_id: str, channel_stats: ChannelResamplingStats
    ) -> dict[str, Any]:
        """Compute resampling statistics for a specific input channel.

        This includes:
            - IDs of removed and added hypotheses.
            - Evidence, evidence slopes and ages of hypotheses in the channel.
            - Hypotheses rotations and pose errors to the target rotations.

        Args:
            graph_id: The object ID for the graph in reference.
            channel_stats: Resampling Stats for the specific input channel.

        Returns:
            A dictionary containing:
                - remove_ids: Hypotheses removed during resampling. Note that these
                    IDs can only be used to index hypotheses from the previous timestep.
                - add_ids: Hypotheses added during resampling at the current timestep.
                - evidence: The hypotheses evidence scores.
                - evidence_slopes: The slopes extracted from the `EvidenceSlopeTracker`
                - ages: The ages of the hypotheses as tracked by the
                    `EvidenceSlopeTracker`.
                - rotations: Rotations of the hypotheses. Note that the buffer encoder
                    will encode those as euler "xyz" rotations in degrees.
                - pose_errors: Rotation errors relative to the target pose.
        """
        tracker = self.hypotheses_updater.evidence_slope_trackers[graph_id]
        mapper = self.channel_hypothesis_mapping[graph_id]
        channel_rotations = mapper.extract(
            self.possible_poses[graph_id], channel_stats.input_channel
        )
        channel_rotations_inv = Rotation.from_matrix(channel_rotations).inv()
        channel_evidence = mapper.extract(
            self.evidence[graph_id], channel_stats.input_channel
        )

        stats = {}
        stats["remove_ids"] = channel_stats.removed_hypotheses_ids
        stats["add_ids"] = channel_stats.added_hypotheses_ids
        stats["evidence"] = channel_evidence
        stats["evidence_slopes"] = tracker._calculate_slopes(
            channel_stats.input_channel
        )
        stats["ages"] = tracker.hyp_age[channel_stats.input_channel]
        stats["rotations"] = channel_rotations_inv
        stats["pose_errors"] = compute_pose_error(
            channel_rotations_inv,
            Rotation.from_quat(self.primary_target_rotation_quat),
            return_min=False,
        )
        return stats

    def _theoretical_limit_target_object_pose_error(self) -> float:
        """Compute the theoretical minimum rotation error on the target object.

        This considers all possible hypotheses rotations on the target object
        and compares them to the target's rotation. The theoretical limit conveys the
        best achievable performance if Monty selects the best hypothesis as its most
        likely hypothesis (MLH).

        Note that having a low pose error for the theoretical limit may not be
        sufficient for deciding on the quality of the hypothesis. Despite good
        hypotheses being generally correlated with good theoretical limit, it is
        possible for rotation error to be small (i.e., low geodesic distance to
        ground-truth rotation), while the hypothesis is on a different location
        of the object.

        Returns:
            The minimum achievable rotation error (in radians).
        """
        hyp_rotations = Rotation.from_matrix(
            self.possible_poses[self.primary_target]
        ).inv()
        target_rotation = Rotation.from_quat(self.primary_target_rotation_quat)
        error = compute_pose_error(hyp_rotations, target_rotation, return_min=True)
        return error

    def _mlh_target_object_pose_error(self) -> float:
        """Compute the actual rotation error between predicted and target pose.

        This compares the most likely hypothesis pose (based on evidence) on the target
        object with the ground truth rotation of the target object.

        Returns:
            The rotation error (in radians).
        """
        obj_rotation = self.get_mlh_for_object(self.primary_target)["rotation"].inv()
        target_rotation = Rotation.from_quat(self.primary_target_rotation_quat)
        error = compute_pose_error(obj_rotation, target_rotation)
        return error
