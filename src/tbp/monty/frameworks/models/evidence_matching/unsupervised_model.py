# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Monty model with unsupervised object ID association capabilities.

This module provides a Monty model that supports learning associations between
object IDs across different learning modules without requiring predefined labels.
"""

import logging
from typing import Any, Dict, List

from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching,
)

logger = logging.getLogger(__name__)


def _pairwise_mean_distance(locations: List[Any]) -> float:
    import numpy as np

    if len(locations) < 2:
        return 0.0
    locations = np.array(locations)
    total_distance = 0.0
    num_pairs = 0
    for i in range(len(locations)):
        for j in range(i + 1, len(locations)):
            distance = np.linalg.norm(locations[i] - locations[j])
            total_distance += distance
            num_pairs += 1
    if num_pairs == 0:
        return 0.0
    return total_distance / num_pairs


def _extract_locations_from_object_votes(object_votes):
    _locations = []
    if isinstance(object_votes, list):
        for state in object_votes:
            if hasattr(state, "location") and state.location is not None:
                _locations.append(state.location)
    return _locations


def _extract_confidences_from_object_votes(object_votes):
    _confidences = []
    if isinstance(object_votes, list):
        for state in object_votes:
            if hasattr(state, "confidence"):
                _confidences.append(state.confidence)
    return _confidences


def _extract_locations(votes_per_lm: List[Any], active_lms: List[int]) -> List[Any]:
    """Helper to extract all locations from votes.

    Returns:
        List[Any]: List of location arrays extracted from votes.
    """

    def extract_from_vote(_vote):
        _locations = []
        if isinstance(_vote, dict):
            for _object_votes in _vote.values():
                _locations.extend(_extract_locations_from_object_votes(_object_votes))
        return _locations

    locations = []
    for lm_idx in active_lms:
        vote = votes_per_lm[lm_idx]
        locations.extend(extract_from_vote(vote))
    return locations


def _calculate_spatial_consistency(
    votes_per_lm: List[Any], active_lms: List[int]
) -> float:
    """Calculate spatial consistency across learning modules.

    Returns:
        float: Spatial consistency in [0, 1].
    """
    locations = _extract_locations(votes_per_lm, active_lms)
    mean_distance = _pairwise_mean_distance(locations)
    consistency = 1.0 / (1.0 + mean_distance)
    return float(consistency)


def _extract_max_confidences(
    votes_per_lm: List[Any], active_lms: List[int]
) -> List[float]:
    """Helper to extract max confidence per LM.

    Returns:
        List[float]: Maximum confidence per active LM.
    """

    def extract_confidences_from_vote(_vote):
        _lm_confidences = []
        if isinstance(_vote, dict):
            for _object_votes in _vote.values():
                _lm_confidences.extend(
                    _extract_confidences_from_object_votes(_object_votes)
                )
        return _lm_confidences

    confidences = []
    for lm_idx in active_lms:
        vote = votes_per_lm[lm_idx]
        lm_confidences = extract_confidences_from_vote(vote)
        if lm_confidences:
            confidences.append(max(lm_confidences))
    return confidences


def _calculate_confidence_correlation(
    votes_per_lm: List[Any], active_lms: List[int]
) -> float:
    """Calculate confidence correlation across learning modules.

    Returns:
        float: Confidence correlation proxy in [0, 1].
    """
    confidences = _extract_max_confidences(votes_per_lm, active_lms)
    if len(confidences) < 2:
        return 0.0
    import numpy as np

    if np.std(confidences) == 0:
        return 1.0
    variance = float(np.var(confidences))
    correlation = 1.0 / (1.0 + variance)
    return float(correlation)


class MontyForUnsupervisedAssociation(MontyForEvidenceGraphMatching):
    """Monty model with enhanced voting for unsupervised object ID association.

    This class extends MontyForEvidenceGraphMatching to enable learning modules
    to discover correspondences between their object representations without
    requiring predefined object labels.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the unsupervised association Monty model."""
        # Extract association-specific parameters
        self.enable_association_analysis = kwargs.pop(
            "enable_association_analysis",
            True,
        )
        self.log_association_details = kwargs.pop(
            "log_association_details",
            False,
        )

        super().__init__(*args, **kwargs)

        # Track cross-LM association statistics
        self.association_history = []

        logger.info(
            f"Initialized MontyForUnsupervisedAssociation with "
            f"{len(self.learning_modules)} learning modules"
        )

    def _vote(self):
        """Enhanced voting mechanism that supports unsupervised object ID association.

        This method extends the parent's voting to collect and analyze association
        information across learning modules.
        """
        if self.lm_to_lm_vote_matrix is None:
            return

        # Collect votes from all learning modules
        votes_per_lm = []
        for i in range(len(self.learning_modules)):
            vote = self.learning_modules[i].send_out_vote()
            votes_per_lm.append(vote)

        # Analyze cross-LM associations before distributing votes
        if self.enable_association_analysis:
            self._analyze_cross_lm_associations(votes_per_lm)

        # Distribute votes to learning modules
        for i in range(len(self.learning_modules)):
            # Collect votes from other LMs that this LM should receive
            voting_data = {}
            for j in self.lm_to_lm_vote_matrix[i]:
                if votes_per_lm[j] is not None:
                    voting_data[f"lm_{j}"] = votes_per_lm[j]

            # Send votes to the learning module
            if voting_data:
                self.learning_modules[i].receive_votes(voting_data)

        # Log association details if enabled
        if self.log_association_details:
            self._log_association_details()

    def _analyze_cross_lm_associations(self, votes_per_lm: List[Any]):
        """Analyze associations between learning modules based on their votes.

        This method collects statistics about how well learning modules are
        associating their object representations.
        """
        if not votes_per_lm:
            return

        # Count active LMs (those with votes)
        active_lms = []
        for i, vote in enumerate(votes_per_lm):
            if vote is not None and vote:
                active_lms.append(i)

        if len(active_lms) < 2:
            return  # Need at least 2 LMs for association analysis

        # Analyze spatial consistency across LMs
        spatial_consistency = _calculate_spatial_consistency(votes_per_lm, active_lms)

        # Analyze confidence correlation across LMs
        confidence_correlation = _calculate_confidence_correlation(
            votes_per_lm,
            active_lms,
        )

        # Store association analysis results
        association_data = {
            "step": self.episode_steps,
            "active_lms": active_lms,
            "spatial_consistency": spatial_consistency,
            "confidence_correlation": confidence_correlation,
            "num_associations": self._count_total_associations(),
        }

        self.association_history.append(association_data)

        # Keep only recent history to manage memory
        if len(self.association_history) > 1000:
            self.association_history = self.association_history[-1000:]

    def _count_total_associations(self) -> int:
        """Count the total number of learned associations across all LMs.

        Returns:
            int: Total number of associations across all LMs.
        """
        total_associations = 0

        for lm in self.learning_modules:
            if hasattr(lm, "get_association_statistics"):
                stats = lm.get_association_statistics()
                total_associations += stats.get("total_associations", 0)

        return total_associations

    def _log_association_details(self):
        """Log detailed association information for debugging."""
        if not logger.isEnabledFor(logging.DEBUG):
            return

        logger.debug("=== Association Details (Step %d) ===", self.episode_steps)

        for i, lm in enumerate(self.learning_modules):
            if hasattr(lm, "get_association_statistics"):
                stats = lm.get_association_statistics()
                logger.debug("LM %d: %s", i, stats)

        if self.association_history:
            latest = self.association_history[-1]
            logger.debug(
                "Cross-LM analysis: spatial_consistency=%.3f,\n"
                "confidence_correlation=%.3f",
                latest["spatial_consistency"],
                latest["confidence_correlation"],
            )

    def get_association_summary(self) -> Dict:
        """Get a summary of association learning across all LMs.

        Returns:
            Dict: Summary containing LM stats and cross-LM analysis.
        """
        summary = {
            "total_lms": len(self.learning_modules),
            "lm_statistics": [],
            "cross_lm_analysis": {
                "history_length": len(self.association_history),
                "recent_spatial_consistency": 0.0,
                "recent_confidence_correlation": 0.0,
            },
        }

        # Collect statistics from each LM
        for i, lm in enumerate(self.learning_modules):
            if hasattr(lm, "get_association_statistics"):
                lm_stats = lm.get_association_statistics()
                lm_stats["lm_id"] = i
                summary["lm_statistics"].append(lm_stats)

        # Add recent cross-LM analysis
        if self.association_history:
            recent_data = self.association_history[-10:]  # Last 10 steps
            if recent_data:
                summary["cross_lm_analysis"]["recent_spatial_consistency"] = sum(
                    d["spatial_consistency"] for d in recent_data
                ) / len(recent_data)
                summary["cross_lm_analysis"]["recent_confidence_correlation"] = sum(
                    d["confidence_correlation"] for d in recent_data
                ) / len(recent_data)

        return summary

    def reset_association_history(self):
        """Reset association history (useful for testing)."""
        self.association_history = []

        # Also reset associations in individual LMs
        for lm in self.learning_modules:
            if hasattr(lm, "reset_associations"):
                lm.reset_associations()

        logger.info("Reset association history for all learning modules")
