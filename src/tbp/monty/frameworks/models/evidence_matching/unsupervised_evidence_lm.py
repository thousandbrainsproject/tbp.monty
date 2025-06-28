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
Enhanced Evidence Learning Module with unsupervised object ID association capabilities.

This module extends the standard EvidenceGraphLM to support learning associations
between object IDs across different learning modules without requiring predefined labels.
"""

import logging
from typing import Dict

import numpy as np

from tbp.monty.frameworks.models.evidence_matching.learning_module import EvidenceGraphLM
from tbp.monty.frameworks.models.states import State
from tbp.monty.frameworks.models.unsupervised_association import UnsupervisedAssociationMixin

logger = logging.getLogger(__name__)


class UnsupervisedEvidenceGraphLM(UnsupervisedAssociationMixin, EvidenceGraphLM):
    """
    Evidence-based learning module with unsupervised object ID association capabilities.
    
    This class combines the evidence-based learning and matching capabilities of
    EvidenceGraphLM with the unsupervised association learning from UnsupervisedAssociationMixin.
    It enables cross-modal learning without requiring predefined object labels.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the unsupervised evidence learning module."""
        # Extract association-specific parameters
        association_params = {
            'association_threshold': kwargs.pop('association_threshold', 0.1),
            'min_association_threshold': kwargs.pop('min_association_threshold', 0.3),
            'spatial_consistency_weight': kwargs.pop('spatial_consistency_weight', 0.3),
            'temporal_consistency_weight': kwargs.pop('temporal_consistency_weight', 0.2),
            'co_occurrence_weight': kwargs.pop('co_occurrence_weight', 0.5),
            'max_association_memory_size': kwargs.pop('max_association_memory_size', 1000),
            'association_learning_enabled': kwargs.pop('association_learning_enabled', True),
        }

        # Initialize parent classes
        super().__init__(*args, **kwargs, **association_params)

        # Track episode steps for association learning
        self.episode_step = 0

        # Set learning_module_id if provided in kwargs
        if 'learning_module_id' in kwargs:
            self.learning_module_id = kwargs['learning_module_id']

        lm_id = getattr(self, 'learning_module_id', 'unknown')
        logger.info(f"Initialized UnsupervisedEvidenceGraphLM {lm_id} "
                    f"with association learning {'enabled' if self.association_learning_enabled else 'disabled'}")

    def receive_votes(self, vote_data):
        """
        Enhanced vote receiving with association learning.
        
        This method first learns/updates associations based on co-occurrence,
        then maps votes to local object IDs using learned associations,
        and finally updates evidence as in the parent class.
        """
        if vote_data is None:
            return

        # Increment step counter for temporal tracking
        self.episode_step += 1

        # Learn associations from co-occurrence patterns
        if self.association_learning_enabled:
            self.update_associations(vote_data, self.episode_step)

            # Map votes to my object IDs using learned associations
            mapped_votes = self.map_votes_to_my_objects(vote_data)

            # Log association mapping for debugging
            if logger.isEnabledFor(logging.DEBUG):
                self._log_vote_mapping(vote_data, mapped_votes)

            # Use mapped votes for evidence updating
            super().receive_votes(mapped_votes)
        else:
            # Fall back to original behavior
            super().receive_votes(vote_data)

    def send_out_vote(self):
        """
        Enhanced vote sending with association metadata.
        
        Extends the parent's vote with additional metadata needed for
        association learning.
        """
        # Get the standard vote from parent class
        vote = super().send_out_vote()

        if vote is None:
            return None

        # Add association metadata to enable better association learning
        enhanced_vote = {}

        for object_id, vote_states in vote.items():
            enhanced_vote[object_id] = []

            for state in vote_states:
                # Create enhanced state with additional metadata
                enhanced_state = self._create_enhanced_vote_state(state, object_id)
                enhanced_vote[object_id].append(enhanced_state)

        return enhanced_vote

    def _create_enhanced_vote_state(self, state: State, object_id: str) -> State:
        """Create an enhanced vote state with association metadata."""
        # Create a copy of the original state
        enhanced_state = State(
            location=state.location,
            morphological_features=state.morphological_features.copy() if state.morphological_features else {},
            non_morphological_features=state.non_morphological_features.copy() if state.non_morphological_features else {},
            confidence=state.confidence,
            use_state=state.use_state,
            sender_id=state.sender_id,
            sender_type=state.sender_type
        )

        # Add association metadata
        if enhanced_state.non_morphological_features is None:
            enhanced_state.non_morphological_features = {}

        enhanced_state.non_morphological_features.update({
            'association_metadata': {
                'sender_lm_id': self.learning_module_id,
                'object_id': object_id,
                'episode_step': self.episode_step,
                'total_evidence': self._get_total_evidence_for_object(object_id),
                'spatial_context': self._get_spatial_context(object_id),
            }
        })

        return enhanced_state

    def _get_total_evidence_for_object(self, object_id: str) -> float:
        """Get total evidence for an object across all hypotheses."""
        if hasattr(self, 'evidence') and object_id in self.evidence:
            return float(np.sum(self.evidence[object_id]))
        return 0.0

    def _get_spatial_context(self, object_id: str) -> Dict:
        """Get spatial context information for an object."""
        context = {}

        if hasattr(self, 'current_mlh') and self.current_mlh.get('graph_id') == object_id:
            context.update({
                'current_location': self.current_mlh.get('location'),
                'current_rotation': self.current_mlh.get('rotation'),
                'current_scale': self.current_mlh.get('scale', 1.0),
            })

        return context

    def _log_vote_mapping(self, original_votes: Dict, mapped_votes: Dict):
        """Log vote mapping for debugging purposes."""
        logger.debug(f"LM {self.learning_module_id} vote mapping:")

        for my_object_id, mapped_vote_list in mapped_votes.items():
            if not mapped_vote_list:
                continue
            logger.debug(f"  {my_object_id}: received {len(mapped_vote_list)} mapped votes")
            self._log_association_strengths(my_object_id, original_votes)

    def _log_association_strengths(self, my_object_id, original_votes):
        """Helper to log association strengths for a given object ID."""
        for other_lm_id, other_objects in original_votes.items():
            if not isinstance(other_objects, dict):
                continue
            for other_object_id in other_objects:
                strength = self.get_association_strength(my_object_id, other_lm_id, other_object_id)
                if strength > 0:
                    logger.debug(f"    {other_lm_id}:{other_object_id} -> strength: {strength:.3f}")

    def pre_episode(self, primary_target=None, semantic_id_to_label=None):
        """Reset episode-specific tracking variables."""
        super().pre_episode(primary_target)
        self.episode_step = 0

        # Log association statistics at the beginning of each episode
        if self.association_learning_enabled and logger.isEnabledFor(logging.INFO):
            stats = self.get_association_statistics()
            logger.info(f"LM {self.learning_module_id} association stats: "
                        f"{stats['total_associations']} total, "
                        f"{stats['strong_associations']} strong, "
                        f"avg strength: {stats['average_strength']:.3f}")

    def post_episode(self, terminal_state=None):
        """Post-episode processing with association analysis."""
        super().post_episode()

        # Log final association statistics for this episode
        if self.association_learning_enabled and logger.isEnabledFor(logging.DEBUG):
            stats = self.get_association_statistics()
            logger.debug(f"LM {self.learning_module_id} end-of-episode association stats: {stats}")

    def get_output(self):
        """Get output with association information."""
        output = super().get_output()

        if output is not None and self.association_learning_enabled:
            # Add association statistics to output for analysis
            if hasattr(output, 'non_morphological_features'):
                if output.non_morphological_features is None:
                    output.non_morphological_features = {}
                output.non_morphological_features['association_stats'] = self.get_association_statistics()

        return output
