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

from tbp.monty.frameworks.utils.evidence_matching import (
    ChannelMapper,
    EvidenceSlopeTracker,
    HypothesesSelection,
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

        # Mock out the evidence_slope_trackers so we can control which values
        # are removed from the list of hypotheses
        tracker1 = Mock()
        tracker1.select_hypotheses = Mock(
            return_value=HypothesesSelection(
                maintain_mask=np.array([False, True, True, False, False])
            )
        )
        self.updater.evidence_slope_trackers = {"object1": tracker1}

        mapper = ChannelMapper(channel_sizes={"patch": channel_size})
        channel_hyps, _ = self.updater.update_hypotheses(
            hypotheses=hypotheses,
            features={"patch": {"pose_fully_defined": True}},
            displacements={"patch": None},
            graph_id="object1",
            mapper=mapper,
            evidence_update_threshold=0,
        )

        assert_array_equal(channel_hyps[0].possible, np.array([True, False]))
        assert_array_equal(channel_hyps[0].evidence, np.array([1, 2]))

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
        tracker.add_hyp(3, "patch")
        tracker.update(np.array([0.0, 0.2, 0.1]), "patch")
        tracker.update(np.array([0.25, 0.5, -0.1]), "patch")
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
        tracker.add_hyp(3, "patch")
        # Initial evidence then high update produces high slope
        tracker.update(np.array([0.0, 0.0, 0.0]), "patch")
        tracker.update(np.array([2.0, 2.0, 2.0]), "patch")
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
        tracker.add_hyp(3, "patch")
        tracker.update(np.array([0.0, 0.0, 0.0]), "patch")
        tracker.update(np.array([0.5, 0.5, 0.5]), "patch")
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
        tracker.add_hyp(3, "patch")
        tracker.update(np.array([0.0, 0.0, 0.0]), "patch")
        tracker.update(np.array([2.0, 2.0, 2.0]), "patch")
        self.updater.evidence_slope_trackers = {"object1": tracker}

        with self.updater:
            self.assertEqual(self.updater.sampling_burst_steps, 0)

        self.assertEqual(self.updater.sampling_burst_steps, 0)

    @given(
        sampling_multiplier=st.floats(min_value=0.0, max_value=3.0),
        graph_num_nodes=st.integers(min_value=1, max_value=100),
        pose_fully_defined=st.booleans(),
    )
    def test_sample_count_returns_informed_count_during_burst(
        self, sampling_multiplier, graph_num_nodes, pose_fully_defined
    ) -> None:
        """Test informed_count with various burst sampling parameters.

        When sampling_burst_steps > 0, _sample_count should calculate and
        return a positive informed_count based on graph nodes and sampling_multiplier.

        The sampling_multiplier is capped at num_hyps_per_node:
            - 2 for pose_fully_defined=True,
            - umbilical_num_poses for pose_fully_defined=False

        Informed_count cannot exceed graph_num_nodes * num_hyps_per_node.
        """
        self.updater.sampling_burst_steps = 3
        self.updater.sampling_multiplier = sampling_multiplier
        channel_features = {"pose_fully_defined": pose_fully_defined}
        num_hyps_per_node = self.updater._num_hyps_per_node(
            channel_features=channel_features
        )

        self.mock_graph_memory.get_locations_in_graph = Mock(
            return_value=np.zeros((graph_num_nodes, 3))
        )

        tracker = EvidenceSlopeTracker(min_age=0)
        mapper = ChannelMapper()

        _, informed_count = self.updater._sample_count(
            input_channel="patch",
            channel_features=channel_features,
            graph_id="object1",
            mapper=mapper,
            tracker=tracker,
        )

        # The number of required hypotheses cannot be negative
        self.assertGreaterEqual(informed_count, 0)

        # Divisible by num_hyps_per_node
        self.assertEqual(informed_count % num_hyps_per_node, 0)

        # Cannot exceed the available max number of hypotheses
        self.assertLessEqual(informed_count, graph_num_nodes * num_hyps_per_node)

    @given(
        sampling_multiplier=st.floats(min_value=0.0, max_value=2.0),
        pose_fully_defined=st.booleans(),
    )
    def test_sample_count_returns_zero_informed_count_when_not_in_burst(
        self,
        sampling_multiplier: float,
        pose_fully_defined: bool,
    ) -> None:
        """Test that _sample_count returns informed_count == 0 when not in burst.

        When sampling_burst_steps == 0, _sample_count should return
        informed_count == 0 regardless of other parameters (e.g, sampling_multiplier).
        """
        self.updater.sampling_burst_steps = 0
        self.updater.sampling_multiplier = sampling_multiplier

        tracker = EvidenceSlopeTracker(min_age=0)
        mapper = ChannelMapper()

        _, informed_count = self.updater._sample_count(
            input_channel="patch",
            channel_features={"pose_fully_defined": pose_fully_defined},
            graph_id="object1",
            mapper=mapper,
            tracker=tracker,
        )

        self.assertEqual(informed_count, 0)

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
        tracker.add_hyp(3, "patch")
        tracker.update(np.array([0.0, 0.0, 0.0]), "patch")
        tracker.update(np.array([0.5, 0.5, 0.5]), "patch")
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

        mapper = ChannelMapper(channel_sizes={"patch": channel_size})
        self.updater.evidence_slope_trackers = {}

        # Create a pre-initialized tracker with the "patch" channel
        # This will be used in the mocked call to `EvidenceSlopeTracker`
        new_tracker = EvidenceSlopeTracker()
        new_tracker.add_hyp(channel_size, "patch")
        new_tracker.update(np.array([1.0, 2.0, 3.0]), "patch")

        # Mock the EvidenceSlopeTracker to return our pre-initialized tracker
        # with the correct channels and hypotheses
        with patch(
            "tbp.monty.frameworks.models.evidence_matching."
            "burst_sampling.EvidenceSlopeTracker",
            return_value=new_tracker,
        ):
            self.updater.update_hypotheses(
                hypotheses=hypotheses,
                features={"patch": {"pose_fully_defined": True}},
                displacements={"patch": None},
                graph_id="new_object",
                mapper=mapper,
                evidence_update_threshold=0,
            )

        # Verify the new tracker was added to evidence_slope_trackers for graph_id
        self.assertIn("new_object", self.updater.evidence_slope_trackers)
        self.assertIs(self.updater.evidence_slope_trackers["new_object"], new_tracker)

    def test_sample_existing_returns_empty_when_no_hypotheses_maintained(self) -> None:
        """Test that _sample_existing returns empty arrays when maintain_ids is empty.

        When HypothesesSelection has no hypotheses to maintain, _sample_existing
        should clear the tracker and return empty ChannelHypotheses.
        """
        tracker = EvidenceSlopeTracker(min_age=0)
        tracker.add_hyp(3, "patch")
        tracker.update(np.array([1.0, 2.0, 3.0]), "patch")

        hypotheses = Hypotheses(
            evidence=np.array([1.0, 2.0, 3.0]),
            locations=np.zeros((3, 3)),
            poses=np.zeros((3, 3, 3)),
            possible=np.array([True, True, True]),
        )

        # All hypotheses should be removed (empty maintain_mask)
        hypotheses_selection = HypothesesSelection(
            maintain_mask=np.array([False, False, False])
        )

        mapper = ChannelMapper(channel_sizes={"patch": 3})

        result = self.updater._sample_existing(
            hypotheses_selection=hypotheses_selection,
            hypotheses=hypotheses,
            input_channel="patch",
            mapper=mapper,
            tracker=tracker,
        )

        # Verify empty arrays are returned
        self.assertEqual(result.input_channel, "patch")
        self.assertEqual(result.locations.shape, (0, 3))
        self.assertEqual(result.poses.shape, (0, 3, 3))
        self.assertEqual(result.evidence.shape, (0,))
        self.assertEqual(result.possible.shape, (0,))

        # Verify tracker was cleared for this channel
        self.assertEqual(tracker.total_size("patch"), 0)

    def test_max_global_slope_skips_empty_channels(self) -> None:
        """Test that _max_global_slope skips channels with zero total_size.

        When a tracker has a channel with total_size == 0, that channel
        should be skipped and not affect the max slope calculation.
        """
        tracker = EvidenceSlopeTracker(min_age=0)

        # Add hypotheses to "patch" channel with some evidence
        # Max slope here is 1.0
        tracker.add_hyp(3, "patch")
        tracker.update(np.array([0.0, 0.0, 0.0]), "patch")
        tracker.update(np.array([1.0, 1.0, 1.0]), "patch")

        # Create an empty channel by adding and then clearing.
        # This simulates what happens during episode pruning of hypotheses.
        tracker.add_hyp(2, "empty_channel")
        tracker.update(np.array([0.0, 0.0]), "empty_channel")
        tracker.update(np.array([2.0, 2.0]), "empty_channel")
        tracker.clear_hyp("empty_channel")

        self.updater.evidence_slope_trackers = {"object1": tracker}

        max_slope = self.updater._max_global_slope()

        # Should return 1.0 from the "patch" channel, ignoring the empty channel.
        self.assertEqual(max_slope, 1.0)

    def test_max_global_slope_skips_channels_with_nan_slopes(self) -> None:
        """Test that _max_global_slope handles channels where slopes.size == 0.

        When a channel has hypotheses but calculate_slopes returns an nan
        array (e.g., due to min age requirements), it should be skipped.
        """
        tracker = EvidenceSlopeTracker(min_age=5)  # High min_age

        # Only one update, so slopes will be nan
        tracker.add_hyp(3, "patch")
        tracker.update(np.array([1.0, 2.0, 3.0]), "patch")

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
    def test_sample_informed_returns_empty_when_informed_count_zero(
        self, pose_fully_defined
    ) -> None:
        tracker = EvidenceSlopeTracker()

        result = self.updater._sample_informed(
            channel_features={"pose_fully_defined": pose_fully_defined},
            informed_count=0,
            graph_id="object1",
            input_channel="patch",
            tracker=tracker,
        )

        self.assertEqual(result.input_channel, "patch")
        self.assertEqual(result.locations.shape, (0, 3))
        self.assertEqual(result.poses.shape, (0, 3, 3))
        self.assertEqual(result.evidence.shape, (0,))
        self.assertEqual(result.possible.shape, (0,))

    @given(
        num_nodes=st.integers(min_value=1, max_value=10),
        num_hyps_per_node=st.integers(min_value=1, max_value=10),
    )
    def test_sample_informed_without_feature_matching(
        self, num_nodes, num_hyps_per_node
    ) -> None:
        """Test _sample_informed when use_features_for_matching is False.

        When feature matching is disabled, hypotheses should be sampled from
        all nodes with zero initial evidence.
        """
        informed_count = num_nodes * num_hyps_per_node

        # Set up graph memory mocks
        self.mock_graph_memory.get_num_nodes_in_graph = Mock(return_value=num_nodes)
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

        result = self.updater._sample_informed(
            channel_features={"pose_fully_defined": True},
            informed_count=informed_count,
            graph_id="object1",
            input_channel="patch",
            tracker=tracker,
        )

        self.assertEqual(result.evidence.shape[0], informed_count)
        self.assertEqual(result.locations.shape[0], informed_count)
        self.assertEqual(result.poses.shape[0], informed_count)

        # Evidence should be all zeros when not using feature matching
        assert_array_equal(result.evidence, np.zeros(informed_count))

        # All hypotheses should be marked as not possible (newly sampled)
        assert_array_equal(result.possible, np.zeros(informed_count, dtype=np.bool_))

        # Tracker should have the new hypotheses added
        self.assertEqual(tracker.total_size("patch"), informed_count)

    def test_sample_informed_with_feature_matching(self) -> None:
        """Test _sample_informed when use_features_for_matching is True.

        When feature matching is enabled, hypotheses should be sampled from
        top-k nodes based on feature evidence scores.
        """
        num_nodes = 5
        informed_count = 4  # Request 4 hypotheses (2 nodes * 2 hyps/node)

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

        result = updater._sample_informed(
            channel_features={"pose_fully_defined": True},
            informed_count=informed_count,
            graph_id="object1",
            input_channel="patch",
            tracker=EvidenceSlopeTracker(),
        )

        # Should have 4 hypotheses
        self.assertEqual(result.evidence.shape[0], informed_count)

        # Feature calculator should have been called
        mock_calculator.calculate.assert_called_once()

        # Evidence should be from top-k nodes
        # Indices 3 and 1 have highest scores (0.5 and 0.9)
        self.assertTrue(np.all(result.evidence >= 0.5))

    def test_sample_informed_with_initial_poses_none(self) -> None:
        """Test _sample_informed when initial_possible_poses is None.

        When initial_possible_poses is None, rotations should be computed using
        the graph rotation features.
        """
        num_nodes = 3
        informed_count = 4

        self.updater.initial_possible_poses = None
        self.updater.use_features_for_matching = {"patch": False}

        # Set up graph memory mocks
        self.mock_graph_memory.get_num_nodes_in_graph = Mock(return_value=num_nodes)
        self.mock_graph_memory.get_locations_in_graph = Mock(
            return_value=np.random.rand(num_nodes, 3)
        )

        # Each node has 3 orthonormal rotation vectors (3x3 matrix)
        # We use identity matrices here for simplicity
        self.mock_graph_memory.get_rotation_features_at_all_nodes = Mock(
            return_value=np.tile(np.eye(3), (3, 1, 1)).astype(np.float64)
        )

        result = self.updater._sample_informed(
            channel_features={
                "pose_fully_defined": True,
                "pose_vectors": np.eye(3, dtype=np.float64),
            },
            informed_count=informed_count,
            graph_id="object1",
            input_channel="patch",
            tracker=EvidenceSlopeTracker(),
        )

        # Verify rotation features were fetched
        self.mock_graph_memory.get_rotation_features_at_all_nodes.assert_called_once()

        # Should have 4 hypotheses, each pose is 3x3 rotation
        self.assertEqual(result.poses.shape[0], informed_count)
        self.assertEqual(result.poses.shape[1:], (3, 3))

    @given(
        num_nodes=st.integers(min_value=2, max_value=10),
        num_rotations=st.integers(min_value=1, max_value=10),
    )
    def test_sample_informed_with_initial_poses_set(
        self, num_nodes, num_rotations
    ) -> None:
        """Test _sample_informed when initial_possible_poses is set.

        When initial_possible_poses is a list of rotations, those rotations
        should be tiled across all selected nodes.
        """
        num_selected_nodes = 2
        informed_count = num_selected_nodes * num_rotations

        # Set up graph memory mocks
        self.mock_graph_memory.get_num_nodes_in_graph = Mock(return_value=num_nodes)
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

        result = self.updater._sample_informed(
            channel_features={"pose_fully_defined": True},
            informed_count=informed_count,
            graph_id="object1",
            input_channel="patch",
            tracker=EvidenceSlopeTracker(),
        )

        self.assertEqual(result.poses.shape[0], informed_count)

        # Verify poses are correctly tiled from initial_possible_poses
        expected_rot_mats = np.array(
            [r.as_matrix() for r in self.updater.initial_possible_poses]
        )
        expected_tiled = np.repeat(expected_rot_mats, num_selected_nodes, axis=0)
        np.testing.assert_array_almost_equal(result.poses, expected_tiled, decimal=5)
