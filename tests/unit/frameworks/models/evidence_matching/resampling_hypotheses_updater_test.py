# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from unittest.mock import Mock

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.ma.testutils import assert_array_equal

from tbp.monty.frameworks.models.evidence_matching.hypotheses import (
    Hypotheses,
)
from tbp.monty.frameworks.models.evidence_matching.resampling_hypotheses_updater import (  # noqa: E501
    ResamplingHypothesesUpdater,
)

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

import tempfile
from unittest import TestCase

import hydra

from tbp.monty.frameworks.utils.evidence_matching import (
    ChannelMapper,
    EvidenceSlopeTracker,
    HypothesesSelection,
    InvalidEvidenceThresholdConfig,
)


class ResamplingHypothesesUpdaterTest(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        output_dir = tempfile.mkdtemp()

        with hydra.initialize(version_base=None, config_path="../../../../../conf"):
            cls.cfg = hydra.compose(
                config_name="test",
                overrides=[
                    "test=frameworks/models/evidence_matching/resampling_hypothese_updater",
                    f"test.config.logging.output_dir={output_dir}",
                ],
            )

        exp = hydra.utils.instantiate(cls.cfg.test)
        with exp:
            exp.run()

        rlm = exp.model.learning_modules[0]
        rlm.hypotheses_updater.sampling_burst_steps = 5
        rlm.channel_hypothesis_mapping["capsule3DSolid"] = ChannelMapper()
        rlm.hypotheses_updater.evidence_slope_trackers["capsule3DSolid"] = (
            EvidenceSlopeTracker(min_age=0)
        )
        cls.rlm = rlm

    def _graph_node_count(self, rlm, graph_id):
        """Returns the number of graph points on a specific graph object."""
        return rlm.graph_memory.get_locations_in_graph(graph_id, "patch").shape[0]

    def _num_hyps_multiplier(self, rlm, pose_defined):
        """Returns the expected hyps multiplier based on Principal curvatures."""
        return 2 if pose_defined else rlm.hypotheses_updater.umbilical_num_poses

    def run_sample_count(
        self,
        rlm,
        resampling_multiplier,
        deletion_trigger_slope,
        pose_defined,
        graph_id,
    ):
        rlm.hypotheses_updater.resampling_multiplier = resampling_multiplier
        rlm.hypotheses_updater.deletion_trigger_slope = deletion_trigger_slope
        test_features = {"patch": {"pose_fully_defined": pose_defined}}
        return rlm.hypotheses_updater._sample_count(
            input_channel="patch",
            channel_features=test_features["patch"],
            graph_id=graph_id,
            mapper=rlm.channel_hypothesis_mapping[graph_id],
            tracker=rlm.hypotheses_updater.evidence_slope_trackers[graph_id],
        )

    def _resampling_multiplier_maximum(self, rlm, pose_defined):
        """Tests that the resampling multiplier respects the maximum scaling boundary.

        The resampling multiplier is used to scale the hypothesis space between
        steps. For example, a multiplier of 2, will request to add hypotheses of
        count that is twice the number of nodes in the object graph. However, there
        is a limit to how many hypotheses we can resample. For existing hypotheses,
        the limit is to resample all of them. For newly resampled informed hypotheses,
        the limit depends on whether the pose is defined or not. This test ensures
        that `_sample_count` respects the maximum sampling limit.

        Maximum multiplier cannot exceed the num_hyps_per_node (2 if
        `pose_defined=True` or umbilical_num_poses if `pose_defined=False`).
        """
        graph_id = "capsule3DSolid"
        graph_num_nodes = self._graph_node_count(rlm, graph_id)
        before_count = graph_num_nodes * self._num_hyps_multiplier(rlm, pose_defined)
        rlm.channel_hypothesis_mapping[graph_id].add_channel("patch", before_count)

        resampling_multiplier = 100
        expected_count = before_count + (
            graph_num_nodes * self._num_hyps_multiplier(rlm, pose_defined)
        )
        _, informed_count = self.run_sample_count(
            rlm=rlm,
            resampling_multiplier=resampling_multiplier,
            deletion_trigger_slope=0.0,
            pose_defined=pose_defined,
            graph_id=graph_id,
        )
        self.assertEqual(expected_count, before_count + informed_count)

        # Reset mapper
        rlm.channel_hypothesis_mapping[graph_id] = ChannelMapper()

    def test_resampling_multiplier(self):
        """Tests that the resampling multiplier correctly scales the hypothesis space.

        The resampling multiplier parameter is used to scale the hypothesis space
        between steps. For example, a multiplier of 2, will request to increase the
        number of hypotheses by 2x the number of graph nodes.
        """
        graph_id = "capsule3DSolid"
        pose_defined = True
        graph_num_nodes = self._graph_node_count(self.rlm, graph_id)
        before_count = graph_num_nodes * self._num_hyps_multiplier(
            self.rlm, pose_defined
        )
        self.rlm.channel_hypothesis_mapping[graph_id].add_channel("patch", before_count)
        self.rlm.hypotheses_updater.evidence_slope_trackers[graph_id].add_hyp(
            before_count, "patch"
        )
        resampling_multipliers = [0.5, 1, 2]

        for resampling_multiplier in resampling_multipliers:
            _, informed_count = self.run_sample_count(
                rlm=self.rlm,
                resampling_multiplier=resampling_multiplier,
                deletion_trigger_slope=0.0,
                pose_defined=pose_defined,
                graph_id=graph_id,
            )
            self.assertEqual(graph_num_nodes * resampling_multiplier, informed_count)

        # Reset mapper
        self.rlm.channel_hypothesis_mapping[graph_id] = ChannelMapper()

    def test_resampling_multiplier_maximum_pose_defined(self):
        """Test that resampling multiplier respects maximum scaling (pose defined)."""
        self._resampling_multiplier_maximum(self.rlm, pose_defined=True)

    def test_resampling_multiplier_maximum_pose_undefined(self):
        """Test that resampling multiplier respects maximum scaling (pose undefined)."""
        self._resampling_multiplier_maximum(self.rlm, pose_defined=False)


class ResamplingHypothesesUpdaterUnitTestCase(TestCase):
    def setUp(self) -> None:
        # We'll add specific mocked functions for the graph memory in
        # individual tests, since they'll change from test to test.
        self.mock_graph_memory = Mock()

        self.updater = ResamplingHypothesesUpdater(
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
            ResamplingHypothesesUpdater(
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
            return_value=["patch1"]
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

        mapper = ChannelMapper(channel_sizes={"patch1": channel_size})
        channel_hyps, _ = self.updater.update_hypotheses(
            hypotheses=hypotheses,
            features={"patch1": {"pose_fully_defined": True}},
            displacements={"patch1": None},
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
        resampling_multiplier=st.floats(min_value=0.0, max_value=3.0),
        graph_num_nodes=st.integers(min_value=1, max_value=100),
        pose_fully_defined=st.booleans(),
    )
    def test_sample_count_returns_informed_count_during_burst(
        self, resampling_multiplier, graph_num_nodes, pose_fully_defined
    ) -> None:
        """Test informed_count with various resampling parameters.

        When sampling_burst_steps > 0, _sample_count should calculate and
        return a positive informed_count based on graph nodes and resampling_multiplier.

        The resampling_multiplier is capped at num_hyps_per_node:
            - 2 for pose_fully_defined=True,
            - umbilical_num_poses for pose_fully_defined=False

        Informed_count cannot exceed graph_num_nodes * num_hyps_per_node.
        """
        self.updater.sampling_burst_steps = 3
        self.updater.resampling_multiplier = resampling_multiplier
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

    def test_sample_count_returns_zero_informed_count_when_not_in_burst(self) -> None:
        """Test that _sample_count returns informed_count == 0 when not in burst.

        When sampling_burst_steps == 0, _sample_count should return
        informed_count == 0 regardless of other parameters.
        """
        self.updater.sampling_burst_steps = 0
        self.updater.resampling_multiplier = 0.4

        tracker = EvidenceSlopeTracker(min_age=0)
        mapper = ChannelMapper()

        _, informed_count = self.updater._sample_count(
            input_channel="patch",
            channel_features={"pose_fully_defined": True},
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

    def test_burst_triggers_on_first_step_with_no_trackers(self) -> None:
        """Test that burst triggers on first step when no trackers exist.

        At the start of an episode (no trackers), max_slope is -inf which is
        below any threshold, so a burst should be triggered.
        """
        self.updater.burst_trigger_slope = 1.0
        self.updater.sampling_burst_duration = 5
        self.updater.sampling_burst_steps = 0
        self.updater.evidence_slope_trackers = {}

        with self.updater:
            self.assertEqual(self.updater.sampling_burst_steps, 5)
