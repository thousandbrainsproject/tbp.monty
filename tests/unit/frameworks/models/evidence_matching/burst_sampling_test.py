# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from unittest.mock import Mock, patch

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.ma.testutils import assert_array_equal

from tbp.monty.frameworks.models.evidence_matching.burst_sampling import (
    BurstSamplingHypothesesUpdater,
)
from tbp.monty.frameworks.models.evidence_matching.hypotheses import (
    Hypotheses,
)

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

from unittest import TestCase

from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.models.evidence_matching.evidence_slope_tracker import (
    EvidenceSlopeTracker,
    HypothesesSelection,
)
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    InvalidEvidenceThresholdConfig,
)


class BurstSamplingHypothesesUpdaterTest(TestCase):
    def setUp(self) -> None:
        # We'll add specific mocked functions for the graph memory in
        # individual tests, since they'll change from test to test.
        self.mock_graph_memory = Mock()

        self.updater = BurstSamplingHypothesesUpdater(
            feature_weights={},
            graph_memory=self.mock_graph_memory,
            max_match_distance=0,
            tolerances={},
            evidence_threshold_config="all",
        )

        hypotheses_displacer = Mock()
        hypotheses_displacer.displace_hypotheses_and_compute_evidence = Mock(
            # Have the displacer return the given hypotheses without displacement
            # since we're not testing that.
            side_effect=lambda **kwargs: (kwargs["possible_hypotheses"], Mock()),
        )
        self.updater.hypotheses_displacer = hypotheses_displacer

    def test_init_fails_when_passed_invalid_evidence_threshold_config(self) -> None:
        """Test that the updater only accepts "all" for evidence_threshold_config."""
        with self.assertRaises(InvalidEvidenceThresholdConfig):
            BurstSamplingHypothesesUpdater(
                feature_weights={},
                graph_memory=self.mock_graph_memory,
                max_match_distance=0,
                tolerances={},
                evidence_threshold_config="invalid",  # type: ignore[arg-type]
            )

    def test_update_hypotheses_ids_map_correctly(self) -> None:
        """Test that hypotheses ids map correctly when some are deleted."""
        channel_size = 5

        # Mocked out because it is accessed by the telemetry
        self.updater.max_slope = Mock()

        hypotheses = Hypotheses(
            # Give each evidence a unique value so we can track which values are
            # remaining in the returned hypotheses
            evidence=np.array(range(channel_size)),
            locations=np.zeros((channel_size, 3)),
            poses=np.zeros((channel_size, 3, 3)),
            # We're going to keep the second and third elements, so set
            # them to some values we can test later, True and False, respectively.
            possible=np.array([False, True, False, False, False]),
        )

        # Add graph memory mock methods
        self.mock_graph_memory.get_input_channels_in_graph = Mock(
            return_value=["patch"]
        )
        self.mock_graph_memory.get_locations_in_graph = Mock(
            return_value=np.zeros((channel_size, 3))
        )
        self.mock_graph_memory.get_num_nodes_in_graph = Mock(return_value=channel_size)

        # Mock out the evidence_slope_trackers so we can control which values
        # are removed from the list of hypotheses
        tracker1 = Mock()
        tracker1.total_size = Mock(return_value=channel_size)
        tracker1.select_hypotheses = Mock(
            return_value=HypothesesSelection(
                mask_to_retain=np.array([False, True, True, False, False])
            )
        )
        self.updater.evidence_slope_trackers = {"object1": tracker1}

        result, _ = self.updater.update_hypotheses(
            hypotheses=hypotheses,
            features={"patch": {"pose_fully_defined": True}},
            displacement=np.zeros(3),
            graph_id="object1",
            evidence_update_threshold=0,
        )

        assert_array_equal(result.possible, np.array([True, False]))
        assert_array_equal(result.evidence, np.array([1, 2]))

    def test_burst_triggers_when_max_slope_at_or_below_threshold(self) -> None:
        """Test that burst triggers when max_slope <= burst_trigger_slope.

        When the maximum global slope is at or below the burst trigger threshold
        and we are not already in a burst (sampling_burst_steps == 0), entering
        the context manager should set sampling_burst_steps to sampling_burst_duration.
        """
        self.updater.burst_trigger_slope = 1.0
        self.updater.sampling_burst_duration = 5
        self.updater.sampling_burst_steps = 0

        # Set a low-slope tracker to trigger a burst.
        # max_slope (0.5) <= burst_trigger_slope (1.0)
        tracker = EvidenceSlopeTracker(min_age=0)
        tracker.add_hyp(3)
        tracker.update(np.array([0.0, 0.2, 0.1]))
        tracker.update(np.array([0.25, 0.5, -0.1]))
        self.updater.evidence_slope_trackers = {"object1": tracker}

        # We would have 3 slopes (0.25, 0.3, -0.2), of which the maximum
        # will be 0.3
        expected_max_slope = 0.3

        # The context manager will set the sampling_burst_steps to the
        # sampling_burst_duration when a burst is triggered
        expected_burst_steps = self.updater.sampling_burst_duration

        with self.updater:
            self.assertEqual(self.updater.max_slope, expected_max_slope)
            self.assertEqual(self.updater.sampling_burst_steps, expected_burst_steps)

    def test_burst_does_not_trigger_when_max_slope_above_threshold(self) -> None:
        """Test that burst does NOT trigger when max_slope > burst_trigger_slope.

        When the maximum global slope is above the burst trigger threshold,
        no burst should be triggered even if sampling_burst_steps == 0.
        """
        self.updater.burst_trigger_slope = 1.0
        self.updater.sampling_burst_duration = 5
        self.updater.sampling_burst_steps = 0

        tracker = EvidenceSlopeTracker(min_age=0)
        tracker.add_hyp(3)
        # Initial evidence then high update produces high slope
        tracker.update(np.array([0.0, 0.0, 0.0]))
        tracker.update(np.array([2.0, 2.0, 2.0]))
        self.updater.evidence_slope_trackers = {"object1": tracker}

        # We would have 3 slopes (2.0, 2.0, 2.0), of which the maximum
        # will be 2.0
        expected_max_slope = 2.0

        with self.updater:
            self.assertEqual(self.updater.max_slope, expected_max_slope)
            self.assertEqual(self.updater.sampling_burst_steps, 0)

    def test_burst_does_not_trigger_when_already_in_burst(self) -> None:
        """Test that burst does NOT trigger when already in a burst.

        When sampling_burst_steps > 0 (already in a burst), no new burst
        should be triggered even if max_slope <= burst_trigger_slope.
        """
        self.updater.burst_trigger_slope = 1.0
        self.updater.sampling_burst_duration = 5
        self.updater.sampling_burst_steps = 3  # Already in a burst

        tracker = EvidenceSlopeTracker(min_age=0)
        tracker.add_hyp(3)
        tracker.update(np.array([0.0, 0.0, 0.0]))
        tracker.update(np.array([0.5, 0.5, 0.5]))
        self.updater.evidence_slope_trackers = {"object1": tracker}

        # We would have 3 slopes (0.5, 0.5, 0.5), of which the maximum
        # will be 0.5 (less than burst_trigger_slope)
        expected_max_slope = 0.5

        with self.updater:
            self.assertEqual(self.updater.max_slope, expected_max_slope)
            self.assertEqual(self.updater.sampling_burst_steps, 3)

    def test_sampling_burst_steps_decrements_in_exit(self) -> None:
        """Test that sampling_burst_steps decrements by 1 in __exit__.

        When exiting the context manager with sampling_burst_steps > 0,
        it should be decremented by 1.
        """
        self.updater.sampling_burst_steps = 3

        with self.updater:
            pass

        self.assertEqual(self.updater.sampling_burst_steps, 2)

    def test_sampling_burst_steps_does_not_go_negative(self) -> None:
        """Test that sampling_burst_steps does not go below 0.

        When sampling_burst_steps is already 0 and no burst is triggered,
        exiting should not decrement it below 0.
        """
        self.updater.sampling_burst_steps = 0
        self.updater.burst_trigger_slope = 1.0

        # High-slope tracker to prevent a burst
        tracker = EvidenceSlopeTracker(min_age=0)
        tracker.add_hyp(3)
        tracker.update(np.array([0.0, 0.0, 0.0]))
        tracker.update(np.array([2.0, 2.0, 2.0]))
        self.updater.evidence_slope_trackers = {"object1": tracker}

        with self.updater:
            self.assertEqual(self.updater.sampling_burst_steps, 0)

        self.assertEqual(self.updater.sampling_burst_steps, 0)

    @given(
        sampling_multiplier=st.floats(min_value=0.0, max_value=3.0),
        graph_num_nodes=st.integers(min_value=1, max_value=100),
        pose_fully_defined=st.booleans(),
    )
    def test_sample_count_returns_positive_count_during_burst(
        self, sampling_multiplier, graph_num_nodes, pose_fully_defined
    ) -> None:
        """Test sample count with various burst sampling parameters.

        When sampling_burst_steps > 0, _sample_count should calculate and
        return a positive count based on graph nodes and sampling_multiplier.

        The sampling_multiplier is capped at num_hyps_per_node:
            - 2 for pose_fully_defined=True,
            - umbilical_num_poses for pose_fully_defined=False

        Sample count cannot exceed graph_num_nodes * num_hyps_per_node.
        """
        self.updater.sampling_burst_steps = 3
        self.updater.sampling_multiplier = sampling_multiplier
        channel_features = {"pose_fully_defined": pose_fully_defined}
        num_hyps_per_node = self.updater._num_hyps_per_node(features=channel_features)

        self.mock_graph_memory.get_input_channels_in_graph = Mock(
            return_value=["patch"]
        )
        self.mock_graph_memory.get_locations_in_graph = Mock(
            return_value=np.zeros((graph_num_nodes, 3))
        )

        tracker = EvidenceSlopeTracker(min_age=0)

        _, new_hypotheses_per_channel = self.updater._sample_count(
            features={"patch": channel_features},
            graph_id="object1",
            input_channels=["patch"],
            tracker=tracker,
        )

        sample_count = new_hypotheses_per_channel["patch"]

        # The number of required hypotheses cannot be negative
        self.assertGreaterEqual(sample_count, 0)

        # Divisible by num_hyps_per_node
        self.assertEqual(sample_count % num_hyps_per_node, 0)

        # Cannot exceed the available max number of hypotheses.
        self.assertLessEqual(sample_count, graph_num_nodes * num_hyps_per_node)

    @given(
        sampling_multiplier=st.floats(min_value=0.0, max_value=2.0),
        pose_fully_defined=st.booleans(),
    )
    def test_sample_count_returns_zero_count_when_not_in_burst(
        self,
        sampling_multiplier: float,
        pose_fully_defined: bool,
    ) -> None:
        """Test that _sample_count returns an empty mapping when not in burst.

        When sampling_burst_steps == 0, _sample_count should return an empty
        mapping regardless of other parameters (e.g, sampling_multiplier).
        """
        self.updater.sampling_burst_steps = 0
        self.updater.sampling_multiplier = sampling_multiplier

        tracker = EvidenceSlopeTracker(min_age=0)

        _, new_hypotheses_per_channel = self.updater._sample_count(
            features={"patch": {"pose_fully_defined": pose_fully_defined}},
            graph_id="object1",
            input_channels=["patch"],
            tracker=tracker,
        )

        self.assertEqual(new_hypotheses_per_channel, {})

    def test_burst_lasts_exactly_sampling_burst_duration_steps(self) -> None:
        """Test that burst lasts for exactly sampling_burst_duration steps.

        When a burst is triggered, it should last for exactly sampling_burst_duration
        steps (i.e., sampling_burst_steps should decrement from sampling_burst_duration
        down to 0 over that many context manager cycles). During the burst,
        re-triggering is prevented by the `sampling_burst_steps > 0` condition.
        """
        self.updater.burst_trigger_slope = 1.0
        self.updater.sampling_burst_duration = 5
        self.updater.sampling_burst_steps = 0

        # Low max_slope hypotheses to trigger a burst in the first iteration.
        tracker = EvidenceSlopeTracker(min_age=0)
        tracker.add_hyp(3)
        tracker.update(np.array([0.0, 0.0, 0.0]))
        tracker.update(np.array([0.5, 0.5, 0.5]))
        self.updater.evidence_slope_trackers = {"object1": tracker}

        burst_steps_history = []
        for _ in range(5):
            with self.updater:
                burst_steps_history.append(self.updater.sampling_burst_steps)

        self.assertEqual(burst_steps_history, [5, 4, 3, 2, 1])
        self.assertEqual(self.updater.sampling_burst_steps, 0)

    def test_max_global_slope_returns_inf_when_no_trackers(self) -> None:
        """Test that _max_global_slope returns -inf when no trackers exist.

        When evidence_slope_trackers is empty, _max_global_slope should
        return -inf (which is less than any burst_trigger_slope threshold,
        effectively triggering a sampling burst).
        """
        self.updater.evidence_slope_trackers = {}

        max_slope = self.updater._max_global_slope()

        self.assertEqual(max_slope, float("-inf"))

    @given(
        sampling_burst_duration=st.integers(min_value=1, max_value=10),
    )
    def test_burst_triggers_on_first_step_with_no_trackers(
        self, sampling_burst_duration
    ) -> None:
        """Test that burst triggers on first step when no trackers exist.

        At the start of an episode (no trackers), max_slope is -inf which is
        below any threshold, so a burst should be triggered. At the beginning
        of a sampling burst, the burst steps should be set equal to the
        `sampling_burst_duration`.
        """
        self.updater.burst_trigger_slope = 1.0
        self.updater.sampling_burst_duration = sampling_burst_duration
        self.updater.sampling_burst_steps = 0
        self.updater.evidence_slope_trackers = {}

        with self.updater:
            self.assertEqual(self.updater.sampling_burst_steps, sampling_burst_duration)

    def test_init_fails_when_sampling_multiplier_is_negative(self) -> None:
        with self.assertRaises(ValueError) as context:
            BurstSamplingHypothesesUpdater(
                feature_weights={},
                graph_memory=self.mock_graph_memory,
                max_match_distance=0,
                tolerances={},
                evidence_threshold_config="all",
                sampling_multiplier=-0.1,
            )

        self.assertIn("sampling_multiplier should be >= 0", str(context.exception))

    def test_update_hypotheses_creates_tracker_for_new_graph_id(self) -> None:
        """Test that a new EvidenceSlopeTracker is created for unseen graph_id.

        When update_hypotheses is called with a graph_id that doesn't exist
        in evidence_slope_trackers, a new tracker should be created for that
        graph_id.
        """
        channel_size = 3
        self.updater.max_slope = 0.0
        self.updater.sampling_burst_steps = 0

        hypotheses = Hypotheses(
            evidence=np.array([1.0, 2.0, 3.0]),
            locations=np.zeros((channel_size, 3)),
            poses=np.zeros((channel_size, 3, 3)),
            possible=np.array([True, True, True]),
        )

        self.mock_graph_memory.get_input_channels_in_graph = Mock(
            return_value=["patch"]
        )
        self.mock_graph_memory.get_locations_in_graph = Mock(
            return_value=np.zeros((channel_size, 3))
        )
        self.mock_graph_memory.get_num_nodes_in_graph = Mock(return_value=channel_size)

        self.updater.evidence_slope_trackers = {}

        # Create a pre-initialized tracker
        new_tracker = EvidenceSlopeTracker()
        new_tracker.add_hyp(channel_size)
        new_tracker.update(np.array([1.0, 2.0, 3.0]))

        # Mock the EvidenceSlopeTracker to return our pre-initialized tracker
        # with the correct hypotheses
        with patch(
            "tbp.monty.frameworks.models.evidence_matching."
            "burst_sampling.EvidenceSlopeTracker",
            return_value=new_tracker,
        ):
            self.updater.update_hypotheses(
                hypotheses=hypotheses,
                features={"patch": {"pose_fully_defined": True}},
                displacement=np.zeros(3),
                graph_id="new_object",
                evidence_update_threshold=0,
            )

        # Verify the new tracker was added to evidence_slope_trackers for graph_id
        self.assertIn("new_object", self.updater.evidence_slope_trackers)
        self.assertIs(self.updater.evidence_slope_trackers["new_object"], new_tracker)

    def test_sample_existing_hypotheses_returns_empty_when_no_hypotheses_retained(
        self,
    ) -> None:
        """Test sampling existing hypotheses returns empty arrays when none retained.

        When HypothesesSelection has no hypotheses to retain, sampling
        existing hypotheses should clear the tracker and return empty Hypotheses.
        """
        tracker = EvidenceSlopeTracker(min_age=0)
        tracker.add_hyp(3)
        tracker.update(np.array([1.0, 2.0, 3.0]))

        hypotheses = Hypotheses(
            evidence=np.array([1.0, 2.0, 3.0]),
            locations=np.zeros((3, 3)),
            poses=np.zeros((3, 3, 3)),
            possible=np.array([True, True, True]),
        )

        # All hypotheses should be removed (empty mask_to_retain)
        hypotheses_selection = HypothesesSelection(
            mask_to_retain=np.array([False, False, False])
        )

        result = self.updater._sample_existing_hypotheses_based_on_evidence_slopes(
            hypotheses_selection=hypotheses_selection,
            hypotheses=hypotheses,
            tracker=tracker,
        )

        # Verify empty arrays are returned
        self.assertEqual(result.locations.shape, (0, 3))
        self.assertEqual(result.poses.shape, (0, 3, 3))
        self.assertEqual(result.evidence.shape, (0,))
        self.assertEqual(result.possible.shape, (0,))

        # Verify tracker was cleared
        self.assertEqual(tracker.total_size(), 0)

    def test_max_global_slope_skips_empty_trackers(self) -> None:
        """Test that _max_global_slope skips trackers with zero total_size.

        When a tracker has total_size == 0, it should be skipped and not
        affect the max slope calculation.
        """
        # Tracker with some evidence.
        # Max slope here is 1.0
        tracker_with_data = EvidenceSlopeTracker(min_age=0)
        tracker_with_data.add_hyp(3)
        tracker_with_data.update(np.array([0.0, 0.0, 0.0]))
        tracker_with_data.update(np.array([1.0, 1.0, 1.0]))

        # Empty tracker (simulates cleared hypotheses)
        empty_tracker = EvidenceSlopeTracker(min_age=0)

        self.updater.evidence_slope_trackers = {
            "object1": tracker_with_data,
            "object2": empty_tracker,
        }

        max_slope = self.updater._max_global_slope()

        # Should return 1.0, ignoring the empty tracker.
        self.assertEqual(max_slope, 1.0)

    def test_max_global_slope_skips_trackers_with_nan_slopes(self) -> None:
        """Test that _max_global_slope handles trackers where slopes are all nan.

        When a tracker has hypotheses but calculate_slopes returns an nan
        array (e.g., due to min age requirements), it should be skipped.
        """
        tracker = EvidenceSlopeTracker(min_age=5)  # High min_age

        # Only one update, so slopes will be nan
        tracker.add_hyp(3)
        tracker.update(np.array([1.0, 2.0, 3.0]))

        self.updater.evidence_slope_trackers = {"object1": tracker}

        max_slope = self.updater._max_global_slope()

        # Should return -inf since no valid slopes exist
        self.assertEqual(max_slope, float("-inf"))

    @given(
        pose_fully_defined=st.booleans(),
        num_euler_angles=st.integers(min_value=1, max_value=10),
    )
    def test_num_hyps_per_node_with_initial_possible_poses(
        self, pose_fully_defined, num_euler_angles
    ) -> None:
        """Test _num_hyps_per_node returns length of initial_possible_poses.

        When initial_possible_poses is a list of euler angles, _num_hyps_per_node
        should return the length of that list regardless of pose_fully_defined.
        """
        euler_angles = [[0, 0, i * 30] for i in range(num_euler_angles)]

        updater = BurstSamplingHypothesesUpdater(
            feature_weights={},
            graph_memory=self.mock_graph_memory,
            max_match_distance=0,
            tolerances={},
            evidence_threshold_config="all",
            initial_possible_poses=euler_angles,
        )

        self.assertEqual(
            updater._num_hyps_per_node({"pose_fully_defined": pose_fully_defined}),
            num_euler_angles,
        )

    @given(pose_fully_defined=st.booleans())
    def test_sample_new_hypotheses_returns_empty_when_count_zero(
        self, pose_fully_defined
    ) -> None:
        tracker = EvidenceSlopeTracker()

        result = self.updater._sample_new_hypotheses_based_on_current_observation(
            features={"patch": {"pose_fully_defined": pose_fully_defined}},
            graph_id="object1",
            new_hypotheses_per_channel={"patch": 0},
            tracker=tracker,
        )

        self.assertEqual(result.locations.shape, (0, 3))
        self.assertEqual(result.poses.shape, (0, 3, 3))
        self.assertEqual(result.evidence.shape, (0,))
        self.assertEqual(result.possible.shape, (0,))

    @given(
        num_nodes=st.integers(min_value=1, max_value=10),
        num_hyps_per_node=st.integers(min_value=1, max_value=10),
    )
    def test_sample_new_hypotheses_without_feature_matching(
        self, num_nodes, num_hyps_per_node
    ) -> None:
        """Test sampling new hypotheses when use_features_for_matching is False.

        When feature matching is disabled, hypotheses should be sampled from
        all nodes with zero initial evidence.
        """
        sample_count = num_nodes * num_hyps_per_node

        # Set up graph memory mocks
        self.mock_graph_memory.get_locations_in_graph = Mock(
            return_value=np.random.rand(num_nodes, 3)
        )

        # Set up updater with feature matching disabled
        self.updater.use_features_for_matching = {"patch": False}

        # Use predefined poses to avoid needing rotation features
        euler_angles = [[0, 0, i * 30] for i in range(num_hyps_per_node)]
        self.updater.initial_possible_poses = [
            Rotation.from_euler("xyz", pose, degrees=True).inv()
            for pose in euler_angles
        ]

        tracker = EvidenceSlopeTracker()

        result = self.updater._sample_new_hypotheses_based_on_current_observation(
            features={"patch": {"pose_fully_defined": True}},
            graph_id="object1",
            new_hypotheses_per_channel={"patch": sample_count},
            tracker=tracker,
        )

        self.assertEqual(result.evidence.shape[0], sample_count)
        self.assertEqual(result.locations.shape[0], sample_count)
        self.assertEqual(result.poses.shape[0], sample_count)

        # Evidence should be all zeros when not using feature matching
        assert_array_equal(result.evidence, np.zeros(sample_count))

        # All hypotheses should be marked as not possible (newly sampled)
        assert_array_equal(result.possible, np.zeros(sample_count, dtype=np.bool_))

        # Tracker should have the new hypotheses added
        self.assertEqual(tracker.total_size(), sample_count)

    def test_sample_new_hypotheses_with_feature_matching(self) -> None:
        """Test sampling new hypotheses when use_features_for_matching is True.

        When feature matching is enabled, hypotheses should be sampled from
        top-k nodes based on feature evidence scores.
        """
        num_nodes = 5
        sample_count = 4  # Request 4 hypotheses (2 nodes * 2 hyps/node)

        # Set up graph memory mocks
        mock_graph_memory = Mock()
        mock_graph_memory.get_feature_array = Mock(
            return_value={"patch": np.zeros((num_nodes, 3))}
        )
        mock_graph_memory.get_feature_order = Mock(
            return_value={"patch": ["feature1", "feature2", "feature3"]}
        )
        mock_graph_memory.get_locations_in_graph = Mock(
            return_value=np.random.rand(num_nodes, 3)
        )

        updater = BurstSamplingHypothesesUpdater(
            feature_weights={"patch": {"feature1": 1.0}},
            graph_memory=mock_graph_memory,
            max_match_distance=0,
            tolerances={"patch": {"feature1": 0.1}},
            evidence_threshold_config="all",
            feature_evidence_increment=1,
        )
        updater.use_features_for_matching = {"patch": True}

        # Mock the hypotheses displacer
        hypotheses_displacer = Mock()
        hypotheses_displacer.displace_hypotheses_and_compute_evidence = Mock(
            side_effect=lambda **kwargs: (kwargs["possible_hypotheses"], Mock()),
        )
        updater.hypotheses_displacer = hypotheses_displacer

        # Mock the feature evidence calculator
        mock_calculator = Mock()
        mock_calculator.calculate = Mock(
            return_value=np.array([0.1, 0.5, 0.3, 0.9, 0.2])
        )
        updater.feature_evidence_calculator = mock_calculator

        # Use predefined poses (initial possible poses)
        euler_angles = [[0, 0, 0], [0, 0, 180]]
        updater.initial_possible_poses = [
            Rotation.from_euler("xyz", pose, degrees=True).inv()
            for pose in euler_angles
        ]

        result = updater._sample_new_hypotheses_based_on_current_observation(
            features={"patch": {"pose_fully_defined": True}},
            graph_id="object1",
            new_hypotheses_per_channel={"patch": sample_count},
            tracker=EvidenceSlopeTracker(),
        )

        # Should have 4 hypotheses
        self.assertEqual(result.count, sample_count)

        # Feature calculator should have been called
        mock_calculator.calculate.assert_called_once()

        # Evidence should be from top-k nodes
        # Indices 3 and 1 have highest scores (0.5 and 0.9)
        self.assertTrue(np.all(result.evidence >= 0.5))

    def test_sample_new_hypotheses_with_initial_poses_none(self) -> None:
        """Test sampling new hypotheses when initial_possible_poses is None.

        When initial_possible_poses is None, rotations should be computed using
        the graph rotation features.
        """
        num_nodes = 3
        sample_count = 4

        self.updater.initial_possible_poses = None
        self.updater.use_features_for_matching = {"patch": False}

        # Set up graph memory mocks
        self.mock_graph_memory.get_locations_in_graph = Mock(
            return_value=np.random.rand(num_nodes, 3)
        )

        # Each node has 3 orthonormal rotation vectors (3x3 matrix)
        # We use identity matrices here for simplicity
        self.mock_graph_memory.get_rotation_features_at_all_nodes = Mock(
            return_value=np.tile(np.eye(3), (3, 1, 1)).astype(np.float64)
        )

        result = self.updater._sample_new_hypotheses_based_on_current_observation(
            features={
                "patch": {
                    "pose_fully_defined": True,
                    "pose_vectors": np.eye(3, dtype=np.float64),
                },
            },
            graph_id="object1",
            new_hypotheses_per_channel={"patch": sample_count},
            tracker=EvidenceSlopeTracker(),
        )

        # Verify rotation features were fetched
        self.mock_graph_memory.get_rotation_features_at_all_nodes.assert_called_once()

        # Should have 4 hypotheses, each pose is 3x3 rotation
        self.assertEqual(result.poses.shape[0], sample_count)
        self.assertEqual(result.poses.shape[1:], (3, 3))

    @given(
        num_hyps_per_node=st.integers(min_value=2, max_value=10),
        sampling_multiplier=st.floats(min_value=0.0, max_value=3.0),
    )
    def test_sample_count_proportional_multi_channel(
        self, num_hyps_per_node, sampling_multiplier
    ) -> None:
        """Test proportional sampling across two channels with different node counts.

        When two channels have different node counts, _sample_count should
        allocate hypotheses proportionally based on each channel's node count.

        The sampling_multiplier is capped at num_hyps_per_node.
        """
        # Channel A has 6 nodes, channel B has 4 nodes
        channels = ["channel_a", "channel_b"]
        channel_nodes = [6, 4]
        self.updater.sampling_burst_steps = 3
        self.updater.sampling_multiplier = sampling_multiplier

        euler_angles = [
            [0, 0, i * 360 / num_hyps_per_node] for i in range(num_hyps_per_node)
        ]
        self.updater.initial_possible_poses = [
            Rotation.from_euler("xyz", pose, degrees=True).inv()
            for pose in euler_angles
        ]

        self.mock_graph_memory.get_input_channels_in_graph = Mock(return_value=channels)
        # Use side_effect instead of return_value so that each channel
        # returns a different number of node locations.
        self.mock_graph_memory.get_locations_in_graph = Mock(
            side_effect=lambda _graph_id, channel: (
                np.random.rand(channel_nodes[0], 3)
                if channel == "channel_a"
                else np.random.rand(channel_nodes[1], 3)
            )
        )

        _, new_hypotheses_per_channel = self.updater._sample_count(
            features={
                "channel_a": {"pose_fully_defined": True},
                "channel_b": {"pose_fully_defined": True},
            },
            graph_id="object1",
            input_channels=channels,
            tracker=EvidenceSlopeTracker(),
        )

        for channel, num_nodes in zip(channels, channel_nodes):
            capped = min(sampling_multiplier, num_hyps_per_node)
            expected = round(num_nodes * capped)
            expected -= expected % num_hyps_per_node
            self.assertEqual(new_hypotheses_per_channel[channel], expected)
            self.assertEqual(new_hypotheses_per_channel[channel] % num_hyps_per_node, 0)

    @given(
        num_nodes=st.integers(min_value=2, max_value=10),
        num_rotations=st.integers(min_value=1, max_value=10),
    )
    def test_sample_new_hypotheses_with_initial_poses_set(
        self, num_nodes, num_rotations
    ) -> None:
        """Test sampling new hypotheses when initial_possible_poses is set.

        When initial_possible_poses is a list of rotations, those rotations
        should be tiled across all selected nodes.
        """
        num_selected_nodes = 2
        sample_count = num_selected_nodes * num_rotations

        # Set up graph memory mocks
        self.mock_graph_memory.get_locations_in_graph = Mock(
            return_value=np.random.rand(num_nodes, 3)
        )

        # Set up updater with predefined rotations
        euler_angles = [[0, 0, i * 360 / num_rotations] for i in range(num_rotations)]
        self.updater.initial_possible_poses = [
            Rotation.from_euler("xyz", pose, degrees=True).inv()
            for pose in euler_angles
        ]
        self.updater.use_features_for_matching = {"patch": False}

        result = self.updater._sample_new_hypotheses_based_on_current_observation(
            features={"patch": {"pose_fully_defined": True}},
            graph_id="object1",
            new_hypotheses_per_channel={"patch": sample_count},
            tracker=EvidenceSlopeTracker(),
        )

        self.assertEqual(result.poses.shape[0], sample_count)

        # Verify poses are correctly tiled from initial_possible_poses
        expected_rot_mats = np.array(
            [r.as_matrix() for r in self.updater.initial_possible_poses]
        )
        expected_tiled = np.repeat(expected_rot_mats, num_selected_nodes, axis=0)
        np.testing.assert_array_almost_equal(result.poses, expected_tiled, decimal=5)
