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
from tbp.monty.frameworks.models.unsupervised_association import UnsupervisedAssociator
from tbp.monty.frameworks.utils.graph_matching_utils import get_scaled_evidences

logger = logging.getLogger(__name__)


def _convert_to_parent_vote_format(vote_data):
    """
    Convert association vote data back to the format expected by parent class.

    The parent EvidenceGraphLM expects votes in a specific format for evidence updating.

    Args:
        vote_data: Vote data in association format or CMP format

    Returns:
        Vote data in format expected by parent class
    """
    if not vote_data or isinstance(vote_data, list):
        return vote_data
    if isinstance(vote_data, dict):
        converted_votes = []
        stack = [vote_data]
        while stack:
            obj = stack.pop()
            if isinstance(obj, dict):
                stack.extend(obj.values())
            elif isinstance(obj, list):
                stack.extend(obj)
            else:
                converted_votes.append(obj)
        return converted_votes
    return vote_data


class UnsupervisedEvidenceGraphLM(EvidenceGraphLM):
    """
    Evidence-based learning module with unsupervised object ID association capabilities.

    This class combines the evidence-based learning and matching capabilities of
    EvidenceGraphLM with the unsupervised association learning from UnsupervisedAssociator.
    It enables cross-modal learning without requiring predefined object labels.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the unsupervised evidence learning module."""
        # Extract association-specific parameters before passing to parent
        association_params = {
            'association_threshold': kwargs.pop('association_threshold', 0.1),
            'min_association_threshold': kwargs.pop('min_association_threshold', 0.3),
            'spatial_consistency_weight': kwargs.pop('spatial_consistency_weight', 0.3),
            'temporal_consistency_weight': kwargs.pop('temporal_consistency_weight', 0.2),
            'co_occurrence_weight': kwargs.pop('co_occurrence_weight', 0.5),
            'max_association_memory_size': kwargs.pop('max_association_memory_size', 1000),
            'association_learning_enabled': kwargs.pop('association_learning_enabled', True),
        }

        # Extract learning_module_id before passing to parent
        learning_module_id = kwargs.pop('learning_module_id', None)

        # Initialize parent class
        super().__init__(*args, **kwargs)

        # Set learning_module_id if provided
        if learning_module_id is not None:
            self.learning_module_id = learning_module_id

        # Create associator using composition
        lm_id = getattr(self, 'learning_module_id', 'unknown')
        self.associator = UnsupervisedAssociator(
            learning_module_id=lm_id,
            **association_params
        )

        # Track episode steps for association learning
        self.episode_step = 0

        logger.info(f"Initialized UnsupervisedEvidenceGraphLM {lm_id} "
                    f"with association learning {'enabled' if self.associator.association_learning_enabled else 'disabled'}")

    def receive_votes(self, vote_data):
        """
        Enhanced vote receiving with association learning and CMP compliance.

        This method processes CMP-compliant votes that include object IDs and
        association metadata in the non_morphological_features field.
        """
        if vote_data is None:
            return

        # Increment step counter for temporal tracking
        self.episode_step += 1

        # Extract association information from CMP-compliant votes
        if self.associator.association_learning_enabled:
            association_vote_data = self._extract_association_data_from_votes(vote_data)

            # Learn associations from co-occurrence patterns
            self.associator.update_associations(
                association_vote_data,
                self.episode_step,
                self.evidence,
                self.object_evidence_threshold,
                self.current_mlh
            )

            # Map votes to my object IDs using learned associations
            mapped_votes = self.associator.map_votes_to_my_objects(
                association_vote_data,
                self.get_all_known_object_ids()
            )

            # Convert back to format expected by parent class
            parent_vote_data = _convert_to_parent_vote_format(mapped_votes)

            # Log association mapping for debugging
            if logger.isEnabledFor(logging.DEBUG):
                self._log_vote_mapping(association_vote_data, mapped_votes)

            # Use mapped votes for evidence updating
            super().receive_votes(parent_vote_data)
        else:
            # Convert to parent format and fall back to original behavior
            parent_vote_data = _convert_to_parent_vote_format(vote_data)
            super().receive_votes(parent_vote_data)

    def send_out_vote(self):
        """
        Enhanced vote sending with proper CMP compliance and association metadata.

        This method creates votes that properly include object IDs and association
        metadata in the non_morphological_features field, following the CMP protocol.
        """
        if self.buffer.get_num_observations_on_object() == 0:
            return None

        possible_states = {}
        evidences = get_scaled_evidences(self.get_all_evidences())

        for graph_id in evidences.keys():
            interesting_hyp = np.nonzero(
                evidences[graph_id] > self.vote_evidence_threshold
            )
            if len(interesting_hyp[0]) > 0:
                possible_states[graph_id] = []
                for hyp_id in interesting_hyp[0]:
                    # Create CMP-compliant vote with object ID and association metadata
                    vote = State(
                        location=self.possible_locations[graph_id][hyp_id],
                        morphological_features={
                            "pose_vectors": self.possible_poses[graph_id][hyp_id].T,
                            "pose_fully_defined": True,
                        },
                        # FIX: Include object ID and association metadata in CMP
                        non_morphological_features={
                            "object_id": graph_id,  # Essential for association learning!
                            "sender_lm_id": self.learning_module_id,
                            "evidence_strength": evidences[graph_id][hyp_id],
                            "association_metadata": self._get_association_metadata(),
                        },
                        confidence=evidences[graph_id][hyp_id],
                        use_state=True,
                        sender_id=self.learning_module_id,
                        sender_type="LM",
                    )
                    possible_states[graph_id].append(vote)

        return possible_states if possible_states else None

    def _get_association_metadata(self) -> Dict:
        """
        Get association metadata to include in vote messages.

        Returns:
            Dictionary containing metadata useful for association learning
        """
        metadata = {
            "temporal_context": self.episode_step,
            "num_observations": self.buffer.get_num_observations_on_object(),
        }

        # Add spatial context if available
        if hasattr(self, 'current_mlh') and self.current_mlh:
            if 'location' in self.current_mlh:
                metadata["current_location"] = self.current_mlh['location']
            if 'rotation' in self.current_mlh:
                metadata["current_rotation"] = self.current_mlh['rotation']

        # Add association statistics if available
        if hasattr(self, 'associator') and self.associator:
            metadata["total_associations"] = self._count_total_associations()
            metadata["association_learning_enabled"] = self.associator.association_learning_enabled

        return metadata

    def _count_total_associations(self) -> int:
        """
        Count the total number of associations in the association memory.
        Returns:
            int: Total number of associations.
        """
        if hasattr(self, 'associator') and self.associator.association_memory:
            return sum(
                len(other_lm_dict)
                for my_obj_dict in self.associator.association_memory.values()
                for other_lm_dict in my_obj_dict.values()
            )
        return 0

    def _extract_association_data_from_votes(self, vote_data):
        """
        Extract association data from CMP-compliant votes.

        Converts from the new CMP format (with object IDs in non_morphological_features)
        to the format expected by the association learning system.

        Args:
            vote_data: List of votes from other learning modules (CMP format)

        Returns:
            Dictionary in format expected by association learning:
            {sender_lm_id: {object_id: vote_info}}
        """
        if not isinstance(vote_data, list):
            return {}
        return self._flatten_votes_for_association(vote_data)

    @staticmethod
    def _flatten_votes_for_association(vote_data):
        association_data = {}
        stack = [(vote, None) for vote in vote_data]
        while stack:
            item, object_id = stack.pop()
            if isinstance(item, list):
                stack.extend((subitem, object_id) for subitem in item)
            elif isinstance(item, dict) and not hasattr(item, 'non_morphological_features'):
                stack.extend((v, k) for k, v in item.items())
            elif hasattr(item, 'non_morphological_features') and isinstance(item.non_morphological_features, dict):
                nmf = item.non_morphological_features
                sender_lm_id = nmf.get('sender_lm_id', getattr(item, 'sender_id', None))
                actual_object_id = nmf.get('object_id', object_id)
                if sender_lm_id and actual_object_id:
                    association_data.setdefault(sender_lm_id, {})[actual_object_id] = item
        return association_data

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
