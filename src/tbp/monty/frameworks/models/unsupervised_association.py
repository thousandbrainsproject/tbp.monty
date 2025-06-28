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
from typing import Dict, List, Optional, Tuple, Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    # This helps IDEs understand the expected interface without runtime overhead
    pass

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


    # Create a minimal numpy-like interface for basic operations
    class np:
        @staticmethod
        def array(data):
            return data

        @staticmethod
        def linalg_norm(data):
            if isinstance(data, (list, tuple)):
                return sum(x ** 2 for x in data) ** 0.5
            return abs(data)

        linalg = type('linalg', (), {'norm': linalg_norm.__func__})()

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
        weights.reverse()  # Most recent gets highest weight

        weighted_sum = sum(conf * weight for conf, weight in
                           zip(self.confidence_history, weights))
        weight_sum = sum(weights)

        return weighted_sum / weight_sum if weight_sum > 0 else 0.0


def _extract_evidence_from_vote(vote_info: Any) -> float:
    """Extract evidence value from vote information."""
    if isinstance(vote_info, dict):
        return vote_info.get('confidence', vote_info.get('evidence', 0.0))
    elif isinstance(vote_info, (list, tuple)) and len(vote_info) > 0:
        # Handle a list of vote objects
        return max(getattr(vote, 'confidence', 0.0) for vote in vote_info)
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
    """Extract location and pose information from vote data."""
    if isinstance(vote_info, dict):
        return vote_info.get('location'), vote_info.get('pose_vectors')
    elif hasattr(vote_info, 'location'):
        other_pose = getattr(vote_info, 'morphological_features', {}).get('pose_vectors')
        return vote_info.location, other_pose
    return None, None


def _calculate_location_similarity(my_location: Any, other_location: Any) -> float:
    """Calculate similarity between two locations."""
    # Handle both numpy arrays and lists
    my_loc = list(my_location) if hasattr(my_location, '__iter__') else [my_location]
    other_loc = list(other_location) if hasattr(other_location, '__iter__') else [other_location]

    # Simple distance calculation
    distance = sum((a - b) ** 2 for a, b in zip(my_loc, other_loc)) ** 0.5
    return 1.0 / (1.0 + distance)


def _calculate_pose_similarity(my_pose: Any, other_pose: Any) -> float:
    """Calculate similarity between two poses."""
    if my_pose is None or other_pose is None:
        return 1.0  # Default if pose not available

    # Simple pose similarity based on rotation matrix similarity
    if hasattr(my_pose, 'as_matrix'):
        try:
            my_pose.as_matrix()
            # Simple similarity check - this is a placeholder
            return 0.8  # Default reasonable similarity
        except (AttributeError, ValueError, TypeError):
            # Handle specific exceptions that might occur during matrix operations
            return 1.0
    return 1.0


class UnsupervisedAssociationMixin:
    """
    Mixin class that adds unsupervised object ID association capabilities
    to learning modules.

    This mixin enables learning modules to discover correspondences between
    their internal object representations and those of other learning modules
    without requiring predefined object labels.

    This mixin expects to be combined with a class that implements the
    LearningModuleProtocol interface (provides evidence, object_evidence_threshold,
    current_mlh attributes and get_all_known_object_ids method).

    Example usage:
        class MyLearningModule(UnsupervisedAssociationMixin, EvidenceGraphLM):
            pass
    """

    # Type hints for IDE support - these will be provided by the mixed-in class
    if TYPE_CHECKING:
        evidence: Dict[str, Any]
        object_evidence_threshold: float
        current_mlh: Dict[str, Any]

        def get_all_known_object_ids(self) -> List[str]:
            """Return all known object IDs."""
            ...

    def __init__(self, *args, **kwargs):
        """Initialize association capabilities."""
        super().__init__(*args, **kwargs)

        # Association memory: {my_object_id: {other_lm_id: {other_object_id: AssociationData}}}
        self.association_memory = defaultdict(
            lambda: defaultdict(lambda: defaultdict(AssociationData))
        )

        # Configuration parameters
        self.association_threshold = kwargs.get('association_threshold', 0.1)
        self.min_association_threshold = kwargs.get('min_association_threshold', 0.3)
        self.spatial_consistency_weight = kwargs.get('spatial_consistency_weight', 0.3)
        self.temporal_consistency_weight = kwargs.get('temporal_consistency_weight', 0.2)
        self.co_occurrence_weight = kwargs.get('co_occurrence_weight', 0.5)
        self.max_association_memory_size = kwargs.get('max_association_memory_size', 1000)

        # Tracking variables
        self.current_step = 0
        self.association_learning_enabled = kwargs.get('association_learning_enabled', True)

        logger.info(f"Initialized UnsupervisedAssociationMixin for LM {getattr(self, 'learning_module_id', 'unknown')}")

    def update_associations(self, vote_data: Dict, current_step: int):
        """
        Update object ID associations based on co-occurrence with other LMs.
        
        Args:
            vote_data: Dictionary of votes from other learning modules
            current_step: Current step number for temporal tracking
        """
        if not self.association_learning_enabled:
            return

        self.current_step = current_step
        my_current_hypotheses = self._get_current_high_evidence_hypotheses()

        if not my_current_hypotheses:
            return

        for other_lm_id, other_votes in vote_data.items():
            if not isinstance(other_votes, dict):
                continue

            for other_object_id, vote_info in other_votes.items():
                other_evidence = _extract_evidence_from_vote(vote_info)

                if other_evidence > self.association_threshold:
                    self._record_co_occurrence(
                        my_current_hypotheses,
                        other_lm_id,
                        other_object_id,
                        other_evidence,
                        vote_info
                    )

        # Prune old associations to manage memory
        self._prune_association_memory()

    def _get_current_high_evidence_hypotheses(self) -> List[str]:
        """Get object IDs with evidence above threshold."""
        high_evidence_objects = []

        # Ensure this mixin is used with a compatible class
        if not self._check_interface_compatibility():
            return high_evidence_objects

        # Access attributes through the interface
        threshold = getattr(self, 'object_evidence_threshold', 1.0)

        try:
            # These calls are safe because we've checked interface compatibility
            known_object_ids = self.get_all_known_object_ids()
            evidence = self.evidence
        except (AttributeError, TypeError) as e:
            logger.warning(f"Error accessing required attributes: {e}")
            return high_evidence_objects

        for object_id in known_object_ids:
            if object_id in evidence:
                evidence_values = evidence[object_id]
                # Handle both numpy arrays and lists
                if hasattr(evidence_values, '__iter__'):
                    max_evidence = max(evidence_values)
                else:
                    max_evidence = evidence_values

                if max_evidence > threshold:
                    high_evidence_objects.append(object_id)

        return high_evidence_objects

    def _check_interface_compatibility(self) -> bool:
        """
        Check if this mixin is being used with a compatible class.

        Returns:
            True if the class provides the required interface, False otherwise.
        """
        required_attributes = ['evidence', 'object_evidence_threshold', 'current_mlh']
        required_methods = ['get_all_known_object_ids']

        # Check for required attributes
        for attr in required_attributes:
            if not hasattr(self, attr):
                logger.warning(f"Required attribute '{attr}' not found. "
                               f"UnsupervisedAssociationMixin should be used with a class "
                               f"that implements LearningModuleProtocol.")
                return False

        # Check for required methods
        for method in required_methods:
            if not hasattr(self, method) or not callable(getattr(self, method)):
                logger.warning(f"Required method '{method}' not found. "
                               f"UnsupervisedAssociationMixin should be used with a class "
                               f"that implements LearningModuleProtocol.")
                return False

        return True

    def _record_co_occurrence(self, my_objects: List[str], other_lm_id: str,
                              other_object_id: str, other_evidence: float, vote_info: Any):
        """Record co-occurrence between my objects and another LM's object."""
        for my_object_id in my_objects:
            association_data = self.association_memory[my_object_id][other_lm_id][other_object_id]

            # Update co-occurrence count
            association_data.co_occurrence_count += 1

            # Update confidence
            association_data.update_confidence(other_evidence, self.current_step)

            # Update spatial consistency if spatial information is available
            spatial_score = self._calculate_spatial_consistency(vote_info, my_object_id)
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

    def _get_my_spatial_info(self, my_object_id: str) -> Tuple[Any, Any]:
        """Get my object's current spatial information."""
        if not hasattr(self, 'current_mlh'):
            return None, None

        current_mlh = getattr(self, 'current_mlh', {})
        if current_mlh.get('graph_id') != my_object_id:
            return None, None

        return current_mlh.get('location'), current_mlh.get('rotation')

    def _calculate_spatial_consistency(self, vote_info: Any, my_object_id: str) -> Optional[float]:
        """Calculate spatial consistency between my object and other LM's vote."""
        try:
            # Extract spatial information from vote
            other_location, other_pose = _extract_spatial_info_from_vote(vote_info)
            if other_location is None:
                return None

            # Get my object's current spatial information
            my_location, my_pose = self._get_my_spatial_info(my_object_id)
            if my_location is None:
                return None

            # Calculate location similarity
            location_similarity = _calculate_location_similarity(my_location, other_location)

            # Calculate pose similarity
            pose_similarity = _calculate_pose_similarity(my_pose, other_pose)

            # Combine location and pose similarities
            return 0.7 * location_similarity + 0.3 * pose_similarity

        except Exception as e:
            logger.debug(f"Error calculating spatial consistency: {e}")
            return None

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
        temporal_strength = self._calculate_temporal_strength(association_data)

        # Combine all factors
        total_strength = (
                self.co_occurrence_weight * co_occurrence_strength +
                self.temporal_consistency_weight * confidence_strength +
                self.spatial_consistency_weight * spatial_strength +
                0.1 * temporal_strength  # Small weight for recency
        )

        return min(total_strength, 1.0)

    def _get_total_observations(self, object_id: str) -> int:
        """Get total number of observations for an object."""
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
        """Calculate temporal strength based on recency of associations."""
        if not association_data.temporal_context:
            return 0.0

        # Calculate how recent the last association was
        steps_since_last = self.current_step - association_data.last_updated_step
        decay_factor = 0.99  # Decay rate per step

        return decay_factor ** steps_since_last

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
            if not isinstance(other_votes, dict):
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

    def map_votes_to_my_objects(self, vote_data: Dict) -> Dict:
        """
        Map incoming votes to my object IDs using learned associations.

        Args:
            vote_data: Dictionary of votes from other learning modules

        Returns:
            Dictionary mapping my object IDs to weighted votes
        """
        mapped_votes = {}

        if not self.association_learning_enabled:
            # Fall back to original behavior if association learning is disabled
            return vote_data

        # Ensure this mixin is used with a compatible class
        if not self._check_interface_compatibility():
            logger.warning("Interface compatibility check failed, returning original votes")
            return vote_data

        try:
            # This call is safe because we've checked interface compatibility
            known_object_ids = self.get_all_known_object_ids()
        except (AttributeError, TypeError) as e:
            logger.warning(f"Error calling get_all_known_object_ids: {e}, returning original votes")
            return vote_data

        for my_object_id in known_object_ids:
            mapped_votes[my_object_id] = self._map_votes_for_object(my_object_id, vote_data)

        return mapped_votes

    def _prune_association_memory(self):
        """Prune old or weak associations to manage memory usage."""
        if len(self.association_memory) <= self.max_association_memory_size:
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
