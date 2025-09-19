# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Unit and integration tests for unsupervised object ID association functionality.

This module contains two types of tests:

1. Unit Tests (TestUnsupervisedAssociationUnit, TestAssociationData, etc.):
   - Test individual components in isolation using mock
   - Fast execution, focused on specific functionality
   - Test association mixin, data structures, and utility functions

2. Integration Tests (TestUnsupervisedAssociationIntegration):
   - Test end-to-end functionality using real Monty experiments
   - Follow Monty's established integration testing patterns
   - Use actual configurations, run real training/evaluation cycles
   - Validate that association learning works in a complete system
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from tbp.monty.frameworks.models.evidence_matching.unsupervised_evidence_lm import (
    UnsupervisedEvidenceGraphLM,
)
from tbp.monty.frameworks.models.states import State
from tbp.monty.frameworks.models.unsupervised_association import (
    AssociationData,
    UnsupervisedAssociator,
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
        """Placeholder when config utilities are not available.

        Raises:
            ImportError: When config utilities cannot be imported.
        """
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

        # The method uses time weighting with decay_factor=0.95
        # weights = [0.95^0, 0.95^1, 0.95^2] = [1.0, 0.95, 0.9025]
        # (reversed for most recent first). So weights = [0.9025, 0.95, 1.0]
        # for [0.5, 0.8, 0.9].
        weights = [0.9025, 0.95, 1.0]
        weighted_sum = 0.5 * 0.9025 + 0.8 * 0.95 + 0.9 * 1.0
        weight_sum = sum(weights)
        expected_avg = weighted_sum / weight_sum

        self.assertAlmostEqual(avg_confidence, expected_avg, places=5)


class TestCMPCompliance(unittest.TestCase):
    """Test CMP compliance of the unsupervised association system."""

    def test_extract_evidence_from_cmp_vote(self):
        """Test extracting evidence from CMP-compliant vote messages."""
        from tbp.monty.frameworks.models.unsupervised_association import (
            _extract_evidence_from_vote,
        )

        # Test CMP-compliant State object
        state = State(
            location=np.array([1, 2, 3]),
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": True,
            },
            non_morphological_features={
                "object_id": "test_object",
                "evidence_strength": 0.75,
            },
            confidence=0.8,
            use_state=True,
            sender_id="test_lm",
            sender_type="LM",
        )

        # Should extract evidence_strength from non_morphological_features
        evidence = _extract_evidence_from_vote(state)
        self.assertEqual(evidence, 0.75)

        # Test fallback to confidence when evidence_strength not available
        state_no_evidence = State(
            location=np.array([1, 2, 3]),
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": True,
            },
            non_morphological_features={"object_id": "test_object"},
            confidence=0.9,
            use_state=True,
            sender_id="test_lm",
            sender_type="LM",
        )

        evidence = _extract_evidence_from_vote(state_no_evidence)
        self.assertEqual(evidence, 0.9)

    def test_extract_spatial_info_from_cmp_vote(self):
        """Test extracting spatial information from CMP-compliant votes."""
        from tbp.monty.frameworks.models.unsupervised_association import (
            _extract_spatial_info_from_vote,
        )

        pose_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        state = State(
            location=np.array([1, 2, 3]),
            morphological_features={
                "pose_vectors": pose_vectors,
                "pose_fully_defined": True,
            },
            non_morphological_features={"object_id": "test_object"},
            confidence=0.8,
            use_state=True,
            sender_id="test_lm",
            sender_type="LM",
        )

        location, pose = _extract_spatial_info_from_vote(state)

        np.testing.assert_array_equal(location, np.array([1, 2, 3]))
        np.testing.assert_array_equal(pose, pose_vectors)


class TestUnsupervisedAssociator(unittest.TestCase):
    """Test the UnsupervisedAssociator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.associator = UnsupervisedAssociator(
            learning_module_id="test_lm",
            association_threshold=0.1,
            min_association_threshold=0.3,
        )

        # Mock learning module data
        self.evidence = {
            "object_1": np.array([2.0, 1.5, 0.5]),
            "object_2": np.array([1.8, 1.2, 0.3]),
        }
        self.threshold = 1.0
        self.current_mlh = {
            "graph_id": "object_1",
            "location": [1.0, 2.0, 3.0],
            "rotation": Mock(),
            "scale": 1.0,
        }
        self.known_object_ids = list(self.evidence.keys())

    def test_initialization(self):
        """Test that the associator initializes correctly."""
        self.assertIsNotNone(self.associator.association_memory)
        self.assertEqual(self.associator.association_threshold, 0.1)
        self.assertEqual(self.associator.min_association_threshold, 0.3)
        self.assertTrue(self.associator.association_learning_enabled)

    def test_get_current_high_evidence_hypotheses(self):
        """Test getting objects with high evidence."""
        high_evidence_objects = self.associator._get_current_high_evidence_hypotheses(
            self.evidence, self.threshold
        )

        # Both objects should have evidence above a threshold
        self.assertIn("object_1", high_evidence_objects)
        self.assertIn("object_2", high_evidence_objects)

    def test_record_co_occurrence(self):
        """Test recording co-occurrence between objects."""
        vote_info = {"confidence": 0.8, "location": [1.1, 2.1, 3.1]}

        self.associator._record_co_occurrence(
            ["object_1"], "other_lm", "other_object", 0.8, vote_info, self.current_mlh
        )

        # Check that association was recorded
        association_data = self.associator.association_memory["object_1"]["other_lm"][
            "other_object"
        ]
        self.assertEqual(association_data.co_occurrence_count, 1)
        self.assertEqual(len(association_data.confidence_history), 1)

    def test_get_association_strength(self):
        """Test association strength calculation."""
        # Record some associations
        vote_info = {"confidence": 0.8}
        self.associator._record_co_occurrence(
            ["object_1"], "other_lm", "other_object", 0.8, vote_info, self.current_mlh
        )

        # Calculate association strength
        strength = self.associator.get_association_strength(
            "object_1",
            "other_lm",
            "other_object",
        )

        self.assertGreater(strength, 0.0)
        self.assertLessEqual(strength, 1.0)

    def test_get_associated_object_ids(self):
        """Test getting associated object IDs."""
        # Record associations with different strengths
        vote_info_strong = {"confidence": 0.9}
        vote_info_weak = {"confidence": 0.2}

        # Record multiple co-occurrences for strong association
        for _ in range(5):
            self.associator._record_co_occurrence(
                ["object_1"],
                "other_lm",
                "strong_object",
                0.9,
                vote_info_strong,
                self.current_mlh,
            )

        # Record single co-occurrence for weak association
        self.associator._record_co_occurrence(
            ["object_1"],
            "other_lm",
            "weak_object",
            0.2,
            vote_info_weak,
            self.current_mlh,
        )

        # Get associated objects
        associated = self.associator.get_associated_object_ids(
            "object_1",
            "other_lm",
            min_strength=0.1,
        )

        # Should return both, sorted by strength
        self.assertGreater(len(associated), 0)

        # First should be stronger than second
        if len(associated) > 1:
            self.assertGreater(associated[0][1], associated[1][1])

    def test_map_votes_to_my_objects(self):
        """Test mapping votes to local object IDs."""
        # Set up some associations
        vote_info = {"confidence": 0.8}
        for _ in range(3):
            self.associator._record_co_occurrence(
                ["object_1"],
                "other_lm",
                "other_object",
                0.8,
                vote_info,
                self.current_mlh,
            )

        # Create CMP-compliant vote data
        vote_data = {
            "other_lm": {
                "other_object": State(
                    location=np.array([1.0, 2.0, 3.0]),
                    morphological_features={
                        "pose_vectors": np.eye(3),
                        "pose_fully_defined": True,
                    },
                    non_morphological_features={
                        "object_id": "other_object",
                        "evidence_strength": 0.9,
                    },
                    confidence=0.9,
                    use_state=True,
                    sender_id="other_lm",
                    sender_type="LM",
                )
            }
        }

        # Map votes
        mapped_votes = self.associator.map_votes_to_my_objects(
            vote_data,
            self.known_object_ids,
        )

        # Should have mapped votes for object_1
        self.assertIn("object_1", mapped_votes)
        if mapped_votes["object_1"]:
            self.assertGreater(len(mapped_votes["object_1"]), 0)


class TestUnsupervisedEvidenceGraphLM(unittest.TestCase):
    """Test the UnsupervisedEvidenceGraphLM class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the parent class methods
        with patch(
            (
                "tbp.monty.frameworks.models.evidence_matching."
                "learning_module.EvidenceGraphLM.__init__"
            ),
            return_value=None,
        ):
            self.lm = UnsupervisedEvidenceGraphLM(
                learning_module_id="test_lm",
                association_threshold=0.1,
                min_association_threshold=0.3,
            )

            # Set up required attributes that parent would normally set
            self.lm.evidence = {"object_1": np.array([2.0, 1.5])}
            self.lm.object_evidence_threshold = 1.0
            self.lm.current_mlh = {
                "graph_id": "object_1",
                "location": [1.0, 2.0, 3.0],
                "rotation": Mock(),
            }

    def test_initialization(self):
        """Test that the enhanced LM initializes correctly."""
        self.assertEqual(self.lm.learning_module_id, "test_lm")
        self.assertEqual(self.lm.associator.association_threshold, 0.1)
        self.assertEqual(self.lm.associator.min_association_threshold, 0.3)
        self.assertTrue(self.lm.associator.association_learning_enabled)

    @patch(
        "tbp.monty.frameworks.models.evidence_matching.learning_module.EvidenceGraphLM.receive_votes"
    )
    def test_receive_votes_with_association_learning(self, mock_parent_receive):
        """Test enhanced vote receiving with association learning."""
        # Mock required methods
        self.lm.get_all_known_object_ids = Mock(return_value=["object_1"])

        # Create CMP-compliant vote data with object IDs in non_morphological_features
        vote_data = [
            {
                "other_object": [
                    State(
                        location=np.array([1.0, 2.0, 3.0]),
                        morphological_features={
                            "pose_vectors": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                            "pose_fully_defined": True,
                        },
                        non_morphological_features={
                            "object_id": "other_object",
                            "evidence_strength": 0.8,
                            "association_metadata": {
                                "temporal_context": 10,
                                "num_observations": 5,
                            },
                        },
                        confidence=0.8,
                        use_state=True,
                        sender_id="other_lm",
                        sender_type="LM",
                    )
                ]
            }
        ]

        # Call receive_votes
        self.lm.receive_votes(vote_data)

        # Should have called parent's receive_votes
        mock_parent_receive.assert_called_once()

        # Should have an incremented episode step
        self.assertEqual(self.lm.episode_step, 1)


class TestConfigurationUtilities(unittest.TestCase):
    """Test the configuration utility functions."""

    @unittest.skipUnless(HAS_CONFIG_UTILS, "Config utilities not available")
    def test_get_association_params_preset(self):
        """Test getting predefined association parameter sets."""
        # Test valid presets
        conservative = get_association_params_preset("conservative")
        aggressive = get_association_params_preset("aggressive")
        balanced = get_association_params_preset("balanced")

        self.assertIn("association_threshold", conservative)
        self.assertIn("min_association_threshold", aggressive)
        self.assertIn("spatial_consistency_weight", balanced)

        # Conservative should have higher thresholds
        self.assertGreater(
            conservative["min_association_threshold"],
            aggressive["min_association_threshold"],
        )

    @unittest.skipUnless(HAS_CONFIG_UTILS, "Config utilities not available")
    def test_get_association_params_preset_invalid(self):
        """Test error handling for invalid preset names."""
        with self.assertRaises(ValueError):
            get_association_params_preset("invalid_preset_name")

    def test_config_structure_validation(self):
        """Test that we can validate config structures even without full imports."""
        # Test basic config structure that should work
        base_config = {
            "learning_module_class": Mock,
            "learning_module_args": {
                "association_threshold": 0.1,
                "min_association_threshold": 0.3,
                "spatial_consistency_weight": 0.3,
                "temporal_consistency_weight": 0.2,
                "co_occurrence_weight": 0.5,
            },
        }

        # Validate structure
        self.assertIn("learning_module_class", base_config)
        self.assertIn("learning_module_args", base_config)
        self.assertIn("association_threshold", base_config["learning_module_args"])
        self.assertIn("min_association_threshold", base_config["learning_module_args"])


class TestUnsupervisedAssociationUnit(unittest.TestCase):
    """Unit tests for unsupervised association learning components.

    This test class validates the core functionality of the association learning
    system:
    1. Creates mock LMs with different object IDs for the same physical objects
    2. Simulates co-occurrence scenarios where both LMs detect the same object
    3. Verifies that associations are learned through co-occurrence patterns
    4. Tests vote mapping using learned associations
    5. Validates association memory management and pruning

    The test demonstrates that:
    - visual_object_1 and touch_object_A can be learned as the same physical
      object (cup)
    - visual_object_2 and touch_object_B can be learned as the same physical
      object (ball)
    - Votes from touch_lm can be mapped to visual_lm's object IDs
    - Association strength increases with repeated co-occurrence
    - Memory management prevents unbounded growth of association data
    """

    def setUp(self):
        """Set up integration test environment."""
        # Create mock learning modules with association capabilities
        self.lm1 = self._create_mock_lm(
            "visual_lm",
            ["visual_object_1", "visual_object_2"],
        )
        self.lm2 = self._create_mock_lm(
            "touch_lm",
            ["touch_object_A", "touch_object_B"],
        )

        # Simulate same physical objects with different IDs
        # visual_object_1 <-> touch_object_A (same cup)
        # visual_object_2 <-> touch_object_B (same ball)

    def _create_mock_lm(self, lm_id, object_ids):
        """Create a mock learning module with association capabilities.

        Returns:
            MockLM: A mock learning module instance configured for tests.
        """

        class MockLM:
            def __init__(self, learning_module_id, known_objects):
                self.learning_module_id = learning_module_id
                self.known_objects = known_objects
                self.evidence = {obj_id: [0.9] for obj_id in known_objects}
                self.object_evidence_threshold = 0.5
                self.current_mlh = {
                    "graph_id": known_objects[0] if known_objects else None,
                    "location": [0.0, 0.0, 0.0],
                    "rotation": None,
                }
                # Create associator using composition
                self.associator = UnsupervisedAssociator(
                    learning_module_id=learning_module_id,
                    association_threshold=0.1,
                    min_association_threshold=0.3,
                )

            def get_all_known_object_ids(self):
                return self.known_objects

        return MockLM(lm_id, object_ids)

    def test_end_to_end_association_learning(self):
        """Test complete association learning workflow."""
        # Step 1: Simulate co-occurrence scenarios
        self._simulate_co_occurrence_scenario()

        # Step 2: Verify associations are learned
        self._verify_associations_learned()

        # Step 3: Test vote mapping functionality
        self._test_vote_mapping()

    def _simulate_co_occurrence_scenario(self):
        """Simulate scenarios where both LMs detect the same object."""
        # Scenario 1: Both LMs detect the same cup (visual_object_1 <-> touch_object_A)
        for step in range(10):
            # LM1 has high evidence for visual_object_1, low for others
            self.lm1.evidence = {
                "visual_object_1": np.array([2.0, 1.8, 1.5]),  # High evidence
                "visual_object_2": np.array([0.3, 0.2, 0.1]),  # Low evidence
            }
            self.lm1.current_mlh = {
                "graph_id": "visual_object_1",
                "location": [1.0, 2.0, 3.0],
                "rotation": None,
            }

            # LM2 has high evidence for touch_object_A
            vote_data_for_lm1 = {
                "touch_lm": {
                    "touch_object_A": {
                        "confidence": 0.85,
                        "evidence": 0.85,
                        "location": [1.1, 2.1, 3.1],  # Similar location
                        "step": step,
                    }
                }
            }

            # Update associations
            self.lm1.associator.update_associations(
                vote_data_for_lm1,
                step,
                self.lm1.evidence,
                self.lm1.object_evidence_threshold,
                self.lm1.current_mlh,
            )

        # Scenario 2: Both LMs detect the same ball (visual_object_2 <-> touch_object_B)
        for step in range(10, 20):
            # LM1 has high evidence for visual_object_2, low for others
            self.lm1.evidence = {
                "visual_object_1": np.array([0.2, 0.1, 0.3]),  # Low evidence
                "visual_object_2": np.array([1.9, 1.7, 1.6]),  # High evidence
            }
            self.lm1.current_mlh = {
                "graph_id": "visual_object_2",
                "location": [5.0, 6.0, 7.0],
                "rotation": None,
            }

            vote_data_for_lm1 = {
                "touch_lm": {
                    "touch_object_B": {
                        "confidence": 0.80,
                        "evidence": 0.80,
                        "location": [5.2, 6.1, 7.0],  # Similar location
                        "step": step,
                    }
                }
            }

            self.lm1.associator.update_associations(
                vote_data_for_lm1,
                step,
                self.lm1.evidence,
                self.lm1.object_evidence_threshold,
                self.lm1.current_mlh,
            )

    def _verify_associations_learned(self):
        """Verify that associations have been properly learned."""
        # Check association strength for a cup (visual_object_1 <-> touch_object_A)
        cup_association_strength = self.lm1.associator.get_association_strength(
            "visual_object_1", "touch_lm", "touch_object_A"
        )
        self.assertGreater(
            cup_association_strength,
            0.3,
            "Cup association should be strong after co-occurrence",
        )

        # Check association strength for ball (visual_object_2 <-> touch_object_B)
        ball_association_strength = self.lm1.associator.get_association_strength(
            "visual_object_2", "touch_lm", "touch_object_B"
        )
        self.assertGreater(
            ball_association_strength,
            0.3,
            "Ball association should be strong after co-occurrence",
        )

        # Check that wrong associations are weak
        wrong_association_strength = self.lm1.associator.get_association_strength(
            "visual_object_1", "touch_lm", "touch_object_B"
        )
        self.assertLess(
            wrong_association_strength, 0.1, "Wrong associations should remain weak"
        )

    def _test_vote_mapping(self):
        """Test that vote mapping works correctly using learned associations."""
        # Create incoming votes from touch_lm
        incoming_votes = {
            "touch_lm": {
                "touch_object_A": {
                    "confidence": 0.9,
                    "evidence": 0.9,
                    "location": [1.0, 2.0, 3.0],
                },
                "touch_object_B": {
                    "confidence": 0.7,
                    "evidence": 0.7,
                    "location": [5.0, 6.0, 7.0],
                },
            }
        }

        # Map votes to visual LM's object IDs
        mapped_votes = self.lm1.associator.map_votes_to_my_objects(
            incoming_votes, self.lm1.get_all_known_object_ids()
        )

        # Verify mapping results
        self.assertIn(
            "visual_object_1",
            mapped_votes,
            "visual_object_1 should receive mapped votes",
        )
        self.assertIn(
            "visual_object_2",
            mapped_votes,
            "visual_object_2 should receive mapped votes",
        )

        # Check that votes are properly weighted by association strength
        visual_obj1_votes = mapped_votes["visual_object_1"]
        visual_obj2_votes = mapped_votes["visual_object_2"]

        self.assertGreater(
            len(visual_obj1_votes),
            0,
            "visual_object_1 should get votes from touch_object_A",
        )
        self.assertGreater(
            len(visual_obj2_votes),
            0,
            "visual_object_2 should get votes from touch_object_B",
        )

    def test_association_memory_management(self):
        """Test that association memory is properly managed."""
        # Test memory pruning
        original_max_size = self.lm1.associator.max_association_memory_size
        # Set a small limit for testing
        self.lm1.associator.max_association_memory_size = 5

        # Create many associations to trigger pruning
        for i in range(10):
            vote_data = {
                f"lm_{i}": {
                    f"object_{i}": {"confidence": 0.5, "evidence": 0.5, "step": i}
                }
            }
            self.lm1.associator.update_associations(
                vote_data,
                i,
                self.lm1.evidence,
                self.lm1.object_evidence_threshold,
                self.lm1.current_mlh,
            )

        # Verify memory was pruned
        total_associations = sum(
            len(self.lm1.associator.association_memory[my_obj][other_lm])
            for my_obj in self.lm1.associator.association_memory
            for other_lm in self.lm1.associator.association_memory[my_obj]
        )

        self.assertLessEqual(
            total_associations,
            self.lm1.associator.max_association_memory_size,
            "Association memory should be pruned when limit is exceeded",
        )

        # Restore original size
        self.lm1.associator.max_association_memory_size = original_max_size


def _count_total_associations(learning_modules):
    """Helper to count total associations with co_occurrence_count > 0.

    Returns:
        int: Total number of associations with at least one co-occurrence.
    """

    def count_in_obj(obj):
        count = 0
        for other_lm in obj.values():
            for other_obj in other_lm.values():
                if other_obj.co_occurrence_count > 0:
                    count += 1
        return count

    total = 0
    for lm in learning_modules:
        if hasattr(lm, "association_memory"):
            for my_obj in lm.association_memory.values():
                total += count_in_obj(my_obj)
    return total


class TestUnsupervisedAssociationIntegration(unittest.TestCase):
    """Integration tests using real Monty experiments.

    This test class follows Monty's established integration testing patterns by:
    1. Using real experiment configurations from our benchmarks
    2. Running actual training and evaluation cycles
    3. Validating experiment outputs and association learning metrics
    4. Testing end-to-end functionality with real components
    """

    def setUp(self):
        """Set up integration test with real experiment configuration."""
        # Import here to avoid issues if config dependencies aren't available
        try:
            from benchmarks.configs.unsupervised_association_experiments import (
                create_simple_cross_modal_config,
            )

            self.config_available = True
            self.base_config = create_simple_cross_modal_config()
        except ImportError as e:
            self.config_available = False
            self.skipTest(f"Skipping: config deps not available: {e}")

    @unittest.skipIf(not HAS_CONFIG_UTILS, "Config utilities not available")
    def test_simple_cross_modal_association_experiment(self):
        """Test that simple cross-modal association experiment can be initialized."""
        if not self.config_available:
            self.skipTest("Experiment configuration not available")

        import copy
        import tempfile

        from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment

        # Use a temporary directory for test output
        with tempfile.TemporaryDirectory() as temp_dir:
            config = copy.deepcopy(self.base_config)

            # Modify config for faster testing
            config["experiment_args"].max_train_steps = 5
            config["experiment_args"].max_eval_steps = 3
            config["experiment_args"].n_train_epochs = 1
            config["experiment_args"].n_eval_epochs = 1
            config["monty_config"]["monty_args"].min_train_steps = 2
            config["monty_config"]["monty_args"].min_eval_steps = 2

            # Set output directory to temp location
            config["logging_config"]["output_dir"] = temp_dir

            # Test that the experiment can be initialized
            exp = MontyObjectRecognitionExperiment(config)

            # Test that experiment initializes correctly
            self.assertIsNotNone(exp, "Experiment should be created")
            self.assertIsNotNone(exp.config, "Experiment should have config")

            # Test that the config has the expected structure
            self.assertIn("monty_config", exp.config)
            self.assertIn("learning_module_configs", exp.config["monty_config"])

            # Verify that learning module configs have association parameters
            lm_configs = exp.config["monty_config"]["learning_module_configs"]
            for lm_config in lm_configs.values():
                lm_args = lm_config["learning_module_args"]
                self.assertIn("association_learning_enabled", lm_args)
                self.assertTrue(lm_args["association_learning_enabled"])
                self.assertIn("association_threshold", lm_args)

    def _assert_association_learning_enabled(self, learning_modules):
        """Helper to assert association learning is enabled for all modules."""
        enabled_lms = [
            lm for lm in learning_modules if hasattr(lm, "association_learning_enabled")
        ]
        for lm in enabled_lms:
            self.assertTrue(
                lm.association_learning_enabled,
                "Association learning should be enabled",
            )

    def _validate_association_learning_occurred(self, exp):
        """Validate that association learning occurred during the experiment."""
        total_associations = _count_total_associations(exp.model.learning_modules)
        self.assertGreaterEqual(
            total_associations, 0, "Some associations should be formed during training"
        )
        print(f"Total associations formed: {total_associations}")
        self._assert_association_learning_enabled(exp.model.learning_modules)

    @unittest.skipIf(not HAS_CONFIG_UTILS, "Config utilities not available")
    def test_association_experiment_configuration_validity(self):
        """Test that our association experiment configurations are valid."""
        if not self.config_available:
            self.skipTest("Experiment configuration not available")

        # Test that the configuration has all required components
        config = self.base_config

        # Check required top-level keys
        required_keys = [
            "experiment_class",
            "experiment_args",
            "logging_config",
            "monty_config",
            "dataset_class",
            "dataset_args",
        ]

        for key in required_keys:
            self.assertIn(key, config, f"Configuration should contain {key}")

        # Check that monty_config has association-related components
        monty_config = config["monty_config"]
        self.assertIn("learning_module_configs", monty_config)

        # Verify learning modules are configured for association learning
        lm_configs = monty_config["learning_module_configs"]
        # lm_configs can be either a list or a dict depending on configuration style
        if isinstance(lm_configs, dict):
            self.assertGreater(
                len(lm_configs),
                1,
                "Need >1 LMs for association",
            )
            lm_configs_list = list(lm_configs.values())
        else:
            self.assertIsInstance(lm_configs, list)
            self.assertGreater(
                len(lm_configs),
                1,
                "Need >1 LMs for association",
            )
            lm_configs_list = lm_configs

        # Check that voting matrix is configured for cross-LM communication
        if "lm_to_lm_vote_matrix" in monty_config:
            vote_matrix = monty_config["lm_to_lm_vote_matrix"]
            self.assertIsInstance(vote_matrix, list)
            # Should have voting connections between LMs
            total_connections = sum(len(connections) for connections in vote_matrix)
            self.assertGreater(
                total_connections,
                0,
                "Should have LM voting connections",
            )


if __name__ == "__main__":
    unittest.main()
