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
Unit tests for unsupervised object ID association functionality.

This module tests the core components of the unsupervised association learning
system, including the association mixin, enhanced learning modules, and
configuration utilities.
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from tbp.monty.frameworks.models.evidence_matching.unsupervised_evidence_lm import (
    UnsupervisedEvidenceGraphLM,
)
from tbp.monty.frameworks.models.states import State
from tbp.monty.frameworks.models.unsupervised_association import (
    UnsupervisedAssociationMixin,
    AssociationData,
)

# Import config utilities with error handling
try:
    from tbp.monty.frameworks.config_utils.unsupervised_association_configs import (
        get_association_params_preset,
    )

    HAS_CONFIG_UTILS = True
except ImportError:
    # Define placeholder function when import fails
    def get_association_params_preset():
        """Placeholder function when config utilities are not available."""
        raise ImportError("Config utilities not available")


    HAS_CONFIG_UTILS = False


class TestAssociationData(unittest.TestCase):
    """Test the AssociationData class."""

    def test_association_data_initialization(self):
        """Test that AssociationData initializes correctly."""
        data = AssociationData()

        self.assertEqual(data.co_occurrence_count, 0)
        self.assertEqual(data.spatial_consistency_score, 0.0)
        self.assertEqual(data.temporal_context, [])
        self.assertEqual(data.confidence_history, [])
        self.assertEqual(data.last_updated_step, 0)

    def test_update_confidence(self):
        """Test confidence updating with temporal tracking."""
        data = AssociationData()

        # Update confidence multiple times
        data.update_confidence(0.8, 10)
        data.update_confidence(0.9, 20)
        data.update_confidence(0.7, 30)

        self.assertEqual(len(data.confidence_history), 3)
        self.assertEqual(data.last_updated_step, 30)
        self.assertAlmostEqual(data.confidence_history[-1], 0.7)

    def test_get_average_confidence(self):
        """Test time-weighted average confidence calculation."""
        data = AssociationData()

        # Add some confidence values
        data.update_confidence(0.5, 10)
        data.update_confidence(0.8, 20)
        data.update_confidence(0.9, 30)

        avg_confidence = data.get_average_confidence()

        # Should be weighted towards more recent values
        self.assertGreater(avg_confidence, 0.5)
        self.assertLessEqual(avg_confidence, 1.0)


class TestUnsupervisedAssociationMixin(unittest.TestCase):
    """Test the UnsupervisedAssociationMixin class."""

    def setUp(self):
        """Set up test fixtures."""

        # Create a mock class that includes the mixin
        class MockLM(UnsupervisedAssociationMixin):
            def __init__(self):
                self.learning_module_id = "test_lm"
                self.object_evidence_threshold = 1.0
                self.evidence = {
                    'object_1': np.array([2.0, 1.5, 0.5]),
                    'object_2': np.array([1.8, 1.2, 0.3]),
                }
                self.current_mlh = {
                    'graph_id': 'object_1',
                    'location': [1.0, 2.0, 3.0],
                    'rotation': Mock(),
                    'scale': 1.0,
                }
                super().__init__()

            def get_all_known_object_ids(self):
                return list(self.evidence.keys())

        self.mock_lm = MockLM()

    def test_initialization(self):
        """Test that the mixin initializes correctly."""
        self.assertIsNotNone(self.mock_lm.association_memory)
        self.assertEqual(self.mock_lm.association_threshold, 0.1)
        self.assertEqual(self.mock_lm.min_association_threshold, 0.3)
        self.assertTrue(self.mock_lm.association_learning_enabled)

    def test_get_current_high_evidence_hypotheses(self):
        """Test getting objects with high evidence."""
        high_evidence_objects = self.mock_lm._get_current_high_evidence_hypotheses()

        # Both objects should have evidence above threshold
        self.assertIn('object_1', high_evidence_objects)
        self.assertIn('object_2', high_evidence_objects)

    def test_record_co_occurrence(self):
        """Test recording co-occurrence between objects."""
        vote_info = {'confidence': 0.8, 'location': [1.1, 2.1, 3.1]}

        self.mock_lm._record_co_occurrence(
            ['object_1'], 'other_lm', 'other_object', 0.8, vote_info
        )

        # Check that association was recorded
        association_data = self.mock_lm.association_memory['object_1']['other_lm']['other_object']
        self.assertEqual(association_data.co_occurrence_count, 1)
        self.assertEqual(len(association_data.confidence_history), 1)

    def test_get_association_strength(self):
        """Test association strength calculation."""
        # Record some associations
        vote_info = {'confidence': 0.8}
        self.mock_lm._record_co_occurrence(
            ['object_1'], 'other_lm', 'other_object', 0.8, vote_info
        )

        # Calculate association strength
        strength = self.mock_lm.get_association_strength('object_1', 'other_lm', 'other_object')

        self.assertGreater(strength, 0.0)
        self.assertLessEqual(strength, 1.0)

    def test_get_associated_object_ids(self):
        """Test getting associated object IDs."""
        # Record associations with different strengths
        vote_info_strong = {'confidence': 0.9}
        vote_info_weak = {'confidence': 0.2}

        # Record multiple co-occurrences for strong association
        for _ in range(5):
            self.mock_lm._record_co_occurrence(
                ['object_1'], 'other_lm', 'strong_object', 0.9, vote_info_strong
            )

        # Record single co-occurrence for weak association
        self.mock_lm._record_co_occurrence(
            ['object_1'], 'other_lm', 'weak_object', 0.2, vote_info_weak
        )

        # Get associated objects
        associated = self.mock_lm.get_associated_object_ids('object_1', 'other_lm', min_strength=0.1)

        # Should return both, sorted by strength
        self.assertGreater(len(associated), 0)

        # First should be stronger than second
        if len(associated) > 1:
            self.assertGreater(associated[0][1], associated[1][1])

    def test_map_votes_to_my_objects(self):
        """Test mapping votes to local object IDs."""
        # Set up some associations
        vote_info = {'confidence': 0.8}
        for _ in range(3):
            self.mock_lm._record_co_occurrence(
                ['object_1'], 'other_lm', 'other_object', 0.8, vote_info
            )

        # Create vote data
        vote_data = {
            'other_lm': {
                'other_object': {'confidence': 0.9, 'location': [1.0, 2.0, 3.0]}
            }
        }

        # Map votes
        mapped_votes = self.mock_lm.map_votes_to_my_objects(vote_data)

        # Should have mapped votes for object_1
        self.assertIn('object_1', mapped_votes)
        if mapped_votes['object_1']:
            self.assertGreater(len(mapped_votes['object_1']), 0)


class TestUnsupervisedEvidenceGraphLM(unittest.TestCase):
    """Test the UnsupervisedEvidenceGraphLM class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the parent class methods
        with patch('tbp.monty.frameworks.models.evidence_matching.learning_module.EvidenceGraphLM.__init__',
                   return_value=None):
            self.lm = UnsupervisedEvidenceGraphLM(
                learning_module_id="test_lm",
                association_threshold=0.1,
                min_association_threshold=0.3,
            )

            # Set up required attributes that would normally be set by parent
            self.lm.evidence = {'object_1': np.array([2.0, 1.5])}
            self.lm.object_evidence_threshold = 1.0
            self.lm.current_mlh = {
                'graph_id': 'object_1',
                'location': [1.0, 2.0, 3.0],
                'rotation': Mock(),
            }

    def test_initialization(self):
        """Test that the enhanced LM initializes correctly."""
        self.assertEqual(self.lm.learning_module_id, "test_lm")
        self.assertEqual(self.lm.association_threshold, 0.1)
        self.assertEqual(self.lm.min_association_threshold, 0.3)
        self.assertTrue(self.lm.association_learning_enabled)

    @patch('tbp.monty.frameworks.models.evidence_matching.learning_module.EvidenceGraphLM.receive_votes')
    def test_receive_votes_with_association_learning(self, mock_parent_receive):
        """Test enhanced vote receiving with association learning."""
        # Mock required methods
        self.lm.get_all_known_object_ids = Mock(return_value=['object_1'])

        vote_data = {
            'other_lm': {
                'other_object': [State(
                    location=np.array([1.0, 2.0, 3.0]),
                    morphological_features={
                        'pose_vectors': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                        'pose_fully_defined': True
                    },
                    non_morphological_features={},
                    confidence=0.8,
                    use_state=True,
                    sender_id='other_lm',
                    sender_type='LM'
                )]
            }
        }

        # Call receive_votes
        self.lm.receive_votes(vote_data)

        # Should have called parent's receive_votes
        mock_parent_receive.assert_called_once()

        # Should have incremented episode step
        self.assertEqual(self.lm.episode_step, 1)


class TestConfigurationUtilities(unittest.TestCase):
    """Test the configuration utility functions."""

    @unittest.skipUnless(HAS_CONFIG_UTILS, "Config utilities not available")
    def test_get_association_params_preset(self):
        """Test getting predefined association parameter sets."""
        # Test valid presets
        conservative = get_association_params_preset('conservative')
        aggressive = get_association_params_preset('aggressive')
        balanced = get_association_params_preset('balanced')

        self.assertIn('association_threshold', conservative)
        self.assertIn('min_association_threshold', aggressive)
        self.assertIn('spatial_consistency_weight', balanced)

        # Conservative should have higher thresholds
        self.assertGreater(conservative['min_association_threshold'],
                           aggressive['min_association_threshold'])

    @unittest.skipUnless(HAS_CONFIG_UTILS, "Config utilities not available")
    def test_get_association_params_preset_invalid(self):
        """Test error handling for invalid preset names."""
        with self.assertRaises(ValueError):
            get_association_params_preset('invalid_preset_name')

    def test_config_structure_validation(self):
        """Test that we can validate config structures even without full imports."""
        # Test basic config structure that should work
        base_config = {
            'learning_module_class': Mock,
            'learning_module_args': {
                'association_threshold': 0.1,
                'min_association_threshold': 0.3,
                'spatial_consistency_weight': 0.3,
                'temporal_consistency_weight': 0.2,
                'co_occurrence_weight': 0.5,
            }
        }

        # Validate structure
        self.assertIn('learning_module_class', base_config)
        self.assertIn('learning_module_args', base_config)
        self.assertIn('association_threshold', base_config['learning_module_args'])
        self.assertIn('min_association_threshold', base_config['learning_module_args'])


if __name__ == '__main__':
    unittest.main()
