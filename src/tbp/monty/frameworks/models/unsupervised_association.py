# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""
Unsupervised object ID association mechanisms for cross-modal learning.

This module implements the core functionality for learning associations between
object IDs across different learning modules without requiring predefined labels.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING, Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.utils.logging_utils import compute_pose_error

logger = logging.getLogger(__name__)


class LearningModuleProtocol(Protocol):
    """Protocol defining the interface expected by UnsupervisedAssociationMixin."""

    evidence: Dict[str, Any]
    object_evidence_threshold: float
    current_mlh: Dict[str, Any]

    def get_all_known_object_ids(self) -> List[str]:
        """Return all known object IDs."""
        ...


@dataclass
class AssociationData:
    """Data structure for storing object ID association information."""
    co_occurrence_count: int = 0
    spatial_consistency_score: float = 0.0
    temporal_context: List[int] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    last_updated_step: int = 0

    def update_confidence(self, new_confidence: float, current_step: int):
        """Update confidence history with temporal decay."""
        self.confidence_history.append(new_confidence)
        self.last_updated_step = current_step

        # Keep only recent history (last 100 observations)
        if len(self.confidence_history) > 100:
            self.confidence_history = self.confidence_history[-100:]

    def get_average_confidence(self, decay_factor: float = 0.95) -> float:
        """Calculate time-weighted average confidence."""
        if not self.confidence_history:
            return 0.0

        weights = [decay_factor ** i for i in range(len(self.confidence_history))]
        weights.reverse()  # Most recent gets the highest weight

        weighted_sum = sum(conf * weight for conf, weight in
                           zip(self.confidence_history, weights))
        weight_sum = sum(weights)

        return weighted_sum / weight_sum if weight_sum > 0 else 0.0


def _extract_evidence_from_vote(vote_info: Any) -> float:
    """Extract evidence value from vote information (CMP-compliant)."""
    if isinstance(vote_info, dict):
        return vote_info.get('confidence', vote_info.get('evidence', 0.0))
    elif isinstance(vote_info, (list, tuple)) and len(vote_info) > 0:
        # Handle a list of vote objects
        return max(getattr(vote, 'confidence', 0.0) for vote in vote_info)
    elif hasattr(vote_info, 'non_morphological_features') and vote_info.non_morphological_features:
        # Extract from CMP-compliant vote - prioritize evidence_strength
        return vote_info.non_morphological_features.get('evidence_strength', vote_info.confidence)
    elif hasattr(vote_info, 'confidence'):
        return vote_info.confidence
    else:
        return 0.0


def _create_weighted_vote(vote_info: Any, weight: float) -> Any:
    """Create a weighted version of a vote."""
    if isinstance(vote_info, dict):
        weighted_vote = vote_info.copy()
        if 'confidence' in weighted_vote:
            weighted_vote['confidence'] *= weight
        elif 'evidence' in weighted_vote:
            weighted_vote['evidence'] *= weight
        return weighted_vote
    elif hasattr(vote_info, 'confidence'):
        # Create a copy and modify confidence
        import copy
        weighted_vote = copy.deepcopy(vote_info)
        weighted_vote.confidence *= weight
        return weighted_vote
    else:
        return vote_info


def _extract_spatial_info_from_vote(vote_info: Any) -> Tuple[Any, Any]:
    """Extract location and pose information from vote data (CMP-compliant)."""
    if isinstance(vote_info, dict):
        return vote_info.get('location'), vote_info.get('pose_vectors')
    elif hasattr(vote_info, 'location'):
        # Extract pose from morphological_features (CMP standard)
        other_pose = None
        if hasattr(vote_info, 'morphological_features') and vote_info.morphological_features:
            other_pose = vote_info.morphological_features.get('pose_vectors')
        return vote_info.location, other_pose
    return None, None


def _calculate_location_similarity(my_location: Any, other_location: Any) -> float:
    """Calculate similarity between two locations (legacy method)."""
    # Handle both numpy arrays and lists
    my_loc = list(my_location) if hasattr(my_location, '__iter__') else [my_location]
    other_loc = list(other_location) if hasattr(other_location, '__iter__') else [other_location]

    # Simple distance calculation
    distance = sum((a - b) ** 2 for a, b in zip(my_loc, other_loc)) ** 0.5
    return 1.0 / (1.0 + distance)


def _calculate_pose_similarity(my_pose: Any, other_pose: Any) -> float:
    """Calculate similarity between two poses using actual rotation comparison."""
    if my_pose is None or other_pose is None:
        return 0.5  # Neutral similarity when pose unavailable

    try:
        # Ensure both poses are scipy Rotation objects
        if hasattr(my_pose, 'as_matrix') and hasattr(other_pose, 'as_matrix'):
            # Use existing pose error calculation from logging_utils
            angular_error = compute_pose_error(my_pose, other_pose)
            # Convert angular error to similarity score (0 error = 1.0 similarity)
            # Normalize by π radians (maximum possible angular difference)
            similarity = np.exp(-angular_error / np.pi)
            return float(similarity)
        elif hasattr(my_pose, 'as_matrix'):
            # Try to convert other_pose to Rotation if it's a matrix
            if hasattr(other_pose, 'shape') and other_pose.shape == (3, 3):
                other_rotation = Rotation.from_matrix(other_pose)
                angular_error = compute_pose_error(my_pose, other_rotation)
                similarity = np.exp(-angular_error / np.pi)
                return float(similarity)
        elif hasattr(other_pose, 'as_matrix'):
            # Try to convert my_pose to Rotation if it's a matrix
            if hasattr(my_pose, 'shape') and my_pose.shape == (3, 3):
                my_rotation = Rotation.from_matrix(my_pose)
                angular_error = compute_pose_error(my_rotation, other_pose)
                similarity = np.exp(-angular_error / np.pi)
                return float(similarity)
    except (AttributeError, ValueError, TypeError) as e:
        # Handle any conversion or calculation errors gracefully
        logger.debug(f"Error calculating pose similarity: {e}")
        return 0.5  # Return neutral similarity on error

    # Fallback for unsupported pose formats
    return 0.5


def _detect_periodicity(temporal_context: Any) -> float:
    """Detect periodic patterns in temporal context."""
    if len(temporal_context) < 3:
        return 0.0

    try:
        # Calculate intervals between consecutive associations
        intervals = np.diff(temporal_context)
        if len(intervals) < 2:
            return 0.0
        interval_std = float(np.std(intervals))
        interval_mean = float(np.mean(intervals))

        if interval_mean == 0:
            return 0.0

        # Lower coefficient of variation indicates more regular pattern
        coefficient_of_variation = interval_std / interval_mean
        import math
        periodicity_score = math.exp(-float(coefficient_of_variation))

        return float(periodicity_score)

    except Exception as e:
        logger.debug(f"Error detecting periodicity: {e}")
        return 0.0


def _calculate_temporal_clustering(temporal_context: Any) -> float:
    """Calculate how clustered the temporal associations are."""
    if len(temporal_context) < 2:
        return 0.0

    try:
        # Calculate the span of temporal context
        time_span = temporal_context[-1] - temporal_context[0]

        if time_span == 0:
            return 1.0  # All associations at the same time = perfect clustering

        # Calculate the density of associations
        num_associations = len(temporal_context)
        density = num_associations / time_span

        # Normalize density score (higher density = better clustering)
        # Use sigmoid-like function to map to [0, 1]
        clustering_score = 2 / (1 + np.exp(-density)) - 1

        return float(clustering_score)

    except Exception as e:
        logger.debug(f"Error calculating temporal clustering: {e}")
        return 0.0


class UnsupervisedAssociator:
    """
    Handles unsupervised object ID association learning for learning modules.

    This class enables learning modules to discover correspondences between
    their internal object representations and those of other learning modules
    without requiring predefined object labels.

    The learning module should create an instance of this class and use it
    via composition rather than inheritance.

    Example usage:
        class MyLearningModule(EvidenceGraphLM):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.associator = UnsupervisedAssociator(
                    association_threshold=0.1,
                    learning_module_id=self.learning_module_id
                )
    """

    def __init__(self,
                 learning_module_id: str,
                 association_threshold: float = 0.1,
                 min_association_threshold: float = 0.3,
                 spatial_consistency_weight: float = 0.3,
                 temporal_consistency_weight: float = 0.2,
                 co_occurrence_weight: float = 0.5,
                 max_association_memory_size: int = 1000,
                 location_weight: float = 0.7,
                 pose_weight: float = 0.3,
                 temporal_recency_weight: float = 0.1,
                 distance_tolerance: float = 1.0,
                 sensor_scale_estimate: float = 1.0,
                 temporal_decay_factor: float = 0.99,
                 periodicity_weight: float = 0.3,
                 clustering_weight: float = 0.2,
                 recency_weight: float = 0.5,
                 association_learning_enabled: bool = True):
        """Initialize association capabilities.

        Args:
            learning_module_id: Identifier for the learning module using this associator
            association_threshold: Minimum evidence threshold for recording associations
            min_association_threshold: Minimum strength threshold for using associations
            spatial_consistency_weight: Weight for spatial consistency in association strength
            temporal_consistency_weight: Weight for temporal consistency in association strength
            co_occurrence_weight: Weight for co-occurrence count in association strength
            max_association_memory_size: Maximum number of associations to keep in memory
            location_weight: Weight for location similarity in spatial consistency
            pose_weight: Weight for pose similarity in spatial consistency
            temporal_recency_weight: Weight for temporal recency in association strength
            distance_tolerance: Tolerance for location similarity calculations
            sensor_scale_estimate: Scale estimate for sensor measurements
            temporal_decay_factor: Decay factor for temporal analysis
            periodicity_weight: Weight for periodicity in temporal analysis
            clustering_weight: Weight for clustering in temporal analysis
            recency_weight: Weight for recency in temporal analysis
            association_learning_enabled: Whether association learning is enabled
        """
        self.learning_module_id = learning_module_id

        # Association memory: {my_object_id: {other_lm_id: {other_object_id: AssociationData}}}
        self.association_memory = defaultdict(
            lambda: defaultdict(lambda: defaultdict(AssociationData))
        )

        # Configuration parameters
        self.association_threshold = association_threshold
        self.min_association_threshold = min_association_threshold
        self.spatial_consistency_weight = spatial_consistency_weight
        self.temporal_consistency_weight = temporal_consistency_weight
        self.co_occurrence_weight = co_occurrence_weight
        self.max_association_memory_size = max_association_memory_size

        # Spatial consistency parameters
        self.location_weight = location_weight
        self.pose_weight = pose_weight
        self.temporal_recency_weight = temporal_recency_weight

        # Location similarity parameters
        self.distance_tolerance = distance_tolerance
        self.sensor_scale_estimate = sensor_scale_estimate

        # Temporal pattern analysis parameters
        self.temporal_decay_factor = temporal_decay_factor
        self.periodicity_weight = periodicity_weight
        self.clustering_weight = clustering_weight
        self.recency_weight = recency_weight

        # Tracking variables
        self.current_step = 0
        self.association_learning_enabled = association_learning_enabled

        logger.info(f"Initialized UnsupervisedAssociator for LM {learning_module_id}")

    def update_associations(self, vote_data: Dict, current_step: int,
                           learning_module_evidence: Dict[str, Any],
                           learning_module_threshold: float,
                           current_mlh: Dict[str, Any]):
        """
        Update object ID associations based on co-occurrence with other LMs.

        Args:
            vote_data: Dictionary of votes from other learning modules
            current_step: Current step number for temporal tracking
            learning_module_evidence: Evidence dictionary from the learning module
            learning_module_threshold: Evidence threshold from the learning module
            current_mlh: Current most likely hypothesis from the learning module
        """
        if not self.association_learning_enabled:
            return

        self.current_step = current_step
        my_current_hypotheses = self._get_current_high_evidence_hypotheses(
            learning_module_evidence, learning_module_threshold
        )

        if not my_current_hypotheses:
            return

        for other_lm_id, other_votes in vote_data.items():
            if not other_votes or not hasattr(other_votes, 'items'):
                continue

            for other_object_id, vote_info in other_votes.items():
                other_evidence = _extract_evidence_from_vote(vote_info)

                if other_evidence > self.association_threshold:
                    self._record_co_occurrence(
                        my_current_hypotheses,
                        other_lm_id,
                        other_object_id,
                        other_evidence,
                        vote_info,
                        current_mlh
                    )

        # Prune old associations to manage memory
        self._prune_association_memory()

    def _get_current_high_evidence_hypotheses(self, evidence: Dict[str, Any],
                                             threshold: float) -> List[str]:
        """Get object IDs with evidence above threshold.

        Args:
            evidence: Evidence dictionary from the learning module
            threshold: Evidence threshold from the learning module

        Returns:
            List of object IDs with evidence above threshold
        """
        high_evidence_objects = []

        try:
            for object_id, evidence_values in evidence.items():
                # Handle both numpy arrays and lists
                if hasattr(evidence_values, '__iter__') and not isinstance(evidence_values, str):
                    max_evidence = float(max(evidence_values))
                else:
                    max_evidence = float(evidence_values)

                if max_evidence > threshold:
                    high_evidence_objects.append(object_id)

        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"Error getting high evidence hypotheses: {e}")

        return high_evidence_objects



    def _record_co_occurrence(self, my_objects: List[str], other_lm_id: str,
                              other_object_id: str, other_evidence: float, vote_info: Any,
                              current_mlh: Dict[str, Any]):
        """Record co-occurrence between my objects and another LM's object.

        Args:
            my_objects: List of my object IDs with high evidence
            other_lm_id: ID of the other learning module
            other_object_id: Object ID from the other learning module
            other_evidence: Evidence strength from the other learning module
            vote_info: Vote information from the other learning module
            current_mlh: Current most likely hypothesis from the learning module
        """
        for my_object_id in my_objects:
            association_data = self.association_memory[my_object_id][other_lm_id][other_object_id]

            # Update co-occurrence count
            association_data.co_occurrence_count += 1

            # Update confidence
            association_data.update_confidence(other_evidence, self.current_step)

            # Update spatial consistency if spatial information is available
            spatial_score = self._calculate_spatial_consistency(vote_info, my_object_id, current_mlh)
            if spatial_score is not None:
                association_data.spatial_consistency_score = (
                        0.9 * association_data.spatial_consistency_score + 0.1 * spatial_score
                )

            # Update temporal context
            association_data.temporal_context.append(self.current_step)
            if len(association_data.temporal_context) > 50:  # Keep recent history
                association_data.temporal_context = association_data.temporal_context[-50:]

            logger.debug(f"Recorded co-occurrence: {my_object_id} <-> {other_lm_id}:{other_object_id} "
                         f"(count: {association_data.co_occurrence_count}, "
                         f"confidence: {association_data.get_average_confidence():.3f})")

    def _get_my_spatial_info(self, my_object_id: str, current_mlh: Dict[str, Any]) -> Tuple[Any, Any]:
        """Get my object's current spatial information.

        Args:
            my_object_id: The object ID to get spatial info for
            current_mlh: Current most likely hypothesis from the learning module

        Returns:
            Tuple of (location, rotation) or (None, None) if not available
        """
        if not current_mlh or current_mlh.get('graph_id') != my_object_id:
            return None, None

        return current_mlh.get('location'), current_mlh.get('rotation')

    def _calculate_spatial_consistency(self, vote_info: Any, my_object_id: str,
                                     current_mlh: Dict[str, Any]) -> Optional[float]:
        """Calculate spatial consistency between my object and another LM's vote.

        Args:
            vote_info: Vote information from another learning module
            my_object_id: My object ID
            current_mlh: Current most likely hypothesis from the learning module

        Returns:
            Spatial consistency score or None if spatial info is not available
        """
        try:
            # Extract spatial information from a vote
            other_location, other_pose = _extract_spatial_info_from_vote(vote_info)
            if other_location is None:
                return None

            # Get my object's current spatial information
            my_location, my_pose = self._get_my_spatial_info(my_object_id, current_mlh)
            if my_location is None:
                return None

            # Calculate location similarity with improved method
            location_similarity = self._calculate_location_similarity_improved(my_location, other_location)

            # Calculate pose similarity
            pose_similarity = _calculate_pose_similarity(my_pose, other_pose)

            # Combine location and pose similarities using configurable weights
            return self.location_weight * location_similarity + self.pose_weight * pose_similarity

        except Exception as e:
            logger.debug(f"Error calculating spatial consistency: {e}")
            return None

    def _calculate_location_similarity_improved(self, my_location: Any, other_location: Any) -> float:
        """Calculate improved location similarity with better normalization and scaling."""
        try:
            # Convert to numpy arrays for consistent handling
            my_loc = np.array(my_location) if hasattr(my_location, '__iter__') else np.array([my_location])
            other_loc = np.array(other_location) if hasattr(other_location, '__iter__') else np.array([other_location])

            # Ensure same dimensionality
            if len(my_loc) != len(other_loc):
                logger.debug(f"Location dimension mismatch: {len(my_loc)} vs {len(other_loc)}")
                return 0.5  # Neutral similarity for dimension mismatch

            # Normalize by sensor scale estimates to handle different sensor ranges
            my_loc_normalized = my_loc / self.sensor_scale_estimate
            other_loc_normalized = other_loc / self.sensor_scale_estimate

            # Check if we have previous locations for movement pattern analysis
            if (hasattr(self, '_previous_my_locations') and
                    hasattr(self, '_previous_other_locations') and
                    len(self._previous_my_locations) > 0 and
                    len(self._previous_other_locations) > 0):

                # Calculate movement deltas
                my_delta = my_loc_normalized - self._previous_my_locations[-1]
                other_delta = other_loc_normalized - self._previous_other_locations[-1]

                # Compare movement patterns (more robust than absolute positions)
                movement_distance = np.linalg.norm(my_delta - other_delta)
                movement_similarity = float(np.exp(-movement_distance / self.distance_tolerance))

                # Store current locations for next comparison
                self._previous_my_locations.append(my_loc_normalized)
                self._previous_other_locations.append(other_loc_normalized)

                # Keep only recent history
                if len(self._previous_my_locations) > 10:
                    self._previous_my_locations = self._previous_my_locations[-10:]
                    self._previous_other_locations = self._previous_other_locations[-10:]

                return float(movement_similarity)
            else:
                # Initialize movement tracking
                if not hasattr(self, '_previous_my_locations'):
                    self._previous_my_locations = []
                    self._previous_other_locations = []

                self._previous_my_locations.append(my_loc_normalized)
                self._previous_other_locations.append(other_loc_normalized)

                # Fallback to improved distance calculation with better normalization
                distance = np.linalg.norm(my_loc_normalized - other_loc_normalized)
                similarity = float(np.exp(-distance / self.distance_tolerance))
                return similarity

        except Exception as e:
            logger.debug(f"Error in improved location similarity calculation: {e}")
            # Fallback to original method
            return _calculate_location_similarity(my_location, other_location)

    def get_association_strength(self, my_object_id: str, other_lm_id: str,
                                 other_object_id: str) -> float:
        """
        Calculate the strength of association between my object ID and another LM's object ID.

        Args:
            my_object_id: My object ID
            other_lm_id: Other learning module ID
            other_object_id: Other LM's object ID

        Returns:
            Association strength between 0.0 and 1.0
        """
        if (my_object_id not in self.association_memory or
                other_lm_id not in self.association_memory[my_object_id] or
                other_object_id not in self.association_memory[my_object_id][other_lm_id]):
            return 0.0

        association_data = self.association_memory[my_object_id][other_lm_id][other_object_id]

        # Calculate components of association strength

        # 1. Co-occurrence frequency
        total_observations = self._get_total_observations(my_object_id)
        co_occurrence_strength = (association_data.co_occurrence_count /
                                  max(total_observations, 1)) if total_observations > 0 else 0.0

        # 2. Average confidence
        confidence_strength = association_data.get_average_confidence()

        # 3. Spatial consistency
        spatial_strength = association_data.spatial_consistency_score

        # 4. Temporal recency (decay factor for old associations)
        temporal_strength = self._calculate_temporal_strength_improved(association_data)

        # Combine all factors using configurable weights
        total_strength = (
                self.co_occurrence_weight * co_occurrence_strength +
                self.temporal_consistency_weight * confidence_strength +
                self.spatial_consistency_weight * spatial_strength +
                self.temporal_recency_weight * temporal_strength
        )

        return min(total_strength, 1.0)

    def _get_total_observations(self, object_id: str) -> int:
        """Get the total number of observations for an object."""
        if not hasattr(self, 'evidence'):
            return 1  # Default to avoid division by zero

        evidence = getattr(self, 'evidence', {})
        if object_id not in evidence:
            return 1

        evidence_values = evidence[object_id]
        if hasattr(evidence_values, '__len__'):
            return len(evidence_values)
        return 1

    def _calculate_temporal_strength(self, association_data: AssociationData) -> float:
        """Calculate temporal strength based on recency of associations (legacy method)."""
        if not association_data.temporal_context:
            return 0.0

        # Calculate how recent the last association was
        steps_since_last = self.current_step - association_data.last_updated_step
        decay_factor = self.temporal_decay_factor

        return decay_factor ** steps_since_last

    def _calculate_temporal_strength_improved(self, association_data: AssociationData) -> float:
        """Calculate enhanced temporal strength with multiple factors."""
        if not association_data.temporal_context:
            return 0.0

        try:
            temporal_context = np.array(association_data.temporal_context)

            # 1. Recency score - how recent the last association was
            recency_score = self._calculate_recency_score(association_data)

            # 2. Periodicity detection - look for regular patterns
            periodicity_score = _detect_periodicity(temporal_context)

            # 3. Temporal clustering - how clustered the associations are
            clustering_score = _calculate_temporal_clustering(temporal_context)

            # Combine all temporal factors using configurable weights
            total_temporal_strength = (
                    self.recency_weight * recency_score +
                    self.periodicity_weight * periodicity_score +
                    self.clustering_weight * clustering_score
            )

            return min(total_temporal_strength, 1.0)

        except Exception as e:
            logger.debug(f"Error in improved temporal strength calculation: {e}")
            # Fallback to original method
            return self._calculate_temporal_strength(association_data)

    def _calculate_recency_score(self, association_data: AssociationData) -> float:
        """Calculate recency score with adaptive decay."""
        steps_since_last = self.current_step - association_data.last_updated_step
        return self.temporal_decay_factor ** steps_since_last

    def get_associated_object_ids(self, my_object_id: str, other_lm_id: str,
                                  min_strength: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Get object IDs from another LM that are associated with my object ID.

        Args:
            my_object_id: My object ID
            other_lm_id: Other learning module ID
            min_strength: Minimum association strength threshold

        Returns:
            List of (other_object_id, association_strength) tuples
        """
        if min_strength is None:
            min_strength = self.min_association_threshold

        associated_objects = []

        if (my_object_id in self.association_memory and
                other_lm_id in self.association_memory[my_object_id]):

            for other_object_id in self.association_memory[my_object_id][other_lm_id]:
                strength = self.get_association_strength(my_object_id, other_lm_id, other_object_id)
                if strength >= min_strength:
                    associated_objects.append((other_object_id, strength))

        # Sort by association strength (descending)
        associated_objects.sort(key=lambda x: x[1], reverse=True)
        return associated_objects

    def _map_votes_for_object(self, my_object_id, vote_data):
        """
        Helper to map votes for a single object ID.
        """
        mapped = []
        for other_lm_id, other_votes in vote_data.items():
            if not other_votes or not hasattr(other_votes, 'items'):
                continue
            for other_object_id, vote_info in other_votes.items():
                association_strength = self.get_association_strength(
                    my_object_id, other_lm_id, other_object_id
                )
                if association_strength > self.min_association_threshold:
                    weighted_vote = _create_weighted_vote(vote_info, association_strength)
                    mapped.append(weighted_vote)
                    logger.debug(f"Mapped vote: {other_lm_id}:{other_object_id} -> {my_object_id} "
                                 f"(strength: {association_strength:.3f})")
        return mapped

    def map_votes_to_my_objects(self, vote_data: Dict, known_object_ids: List[str]) -> Dict:
        """
        Map incoming votes to my object IDs using learned associations.

        Args:
            vote_data: Dictionary of votes from other learning modules
            known_object_ids: List of known object IDs from the learning module

        Returns:
            Dictionary mapping my object IDs to weighted votes
        """
        mapped_votes = {}

        if not self.association_learning_enabled:
            # Fall back to original behavior if association learning is disabled
            return vote_data

        for my_object_id in known_object_ids:
            mapped_votes[my_object_id] = self._map_votes_for_object(my_object_id, vote_data)

        return mapped_votes

    def _prune_association_memory(self):
        """Prune old or weak associations to manage memory usage."""
        # Count total associations
        total_associations = sum(
            len(self.association_memory[my_obj][other_lm])
            for my_obj in self.association_memory
            for other_lm in self.association_memory[my_obj]
        )

        if total_associations <= self.max_association_memory_size:
            return

        # Collect all associations with their strengths
        all_associations = []
        for my_obj_id in self.association_memory:
            for other_lm_id in self.association_memory[my_obj_id]:
                for other_obj_id in self.association_memory[my_obj_id][other_lm_id]:
                    strength = self.get_association_strength(my_obj_id, other_lm_id, other_obj_id)
                    all_associations.append((my_obj_id, other_lm_id, other_obj_id, strength))

        # Sort by strength and keep only the strongest associations
        all_associations.sort(key=lambda x: x[3], reverse=True)
        associations_to_keep = all_associations[:self.max_association_memory_size]

        # Rebuild association memory with only strong associations
        new_memory = defaultdict(lambda: defaultdict(lambda: defaultdict(AssociationData)))
        for my_obj_id, other_lm_id, other_obj_id, _ in associations_to_keep:
            new_memory[my_obj_id][other_lm_id][other_obj_id] = (
                self.association_memory[my_obj_id][other_lm_id][other_obj_id]
            )

        self.association_memory = new_memory
        logger.info(f"Pruned association memory to {len(associations_to_keep)} associations")

    def get_association_statistics(self) -> Dict:
        """Get statistics about current associations for debugging/analysis."""
        stats = {
            'total_associations': 0,
            'associations_by_lm': defaultdict(int),
            'average_strength': 0.0,
            'strong_associations': 0,  # Above min_association_threshold
        }

        total_strength = 0.0
        for my_obj_id in self.association_memory:
            for other_lm_id in self.association_memory[my_obj_id]:
                for other_obj_id in self.association_memory[my_obj_id][other_lm_id]:
                    stats['total_associations'] += 1
                    stats['associations_by_lm'][other_lm_id] += 1

                    strength = self.get_association_strength(my_obj_id, other_lm_id, other_obj_id)
                    total_strength += strength

                    if strength > self.min_association_threshold:
                        stats['strong_associations'] += 1

        if stats['total_associations'] > 0:
            stats['average_strength'] = total_strength / stats['total_associations']

        return dict(stats)

    def reset_associations(self):
        """Reset all learned associations (useful for testing)."""
        self.association_memory = defaultdict(
            lambda: defaultdict(lambda: defaultdict(AssociationData))
        )
        logger.info("Reset all object ID associations")
