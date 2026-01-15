# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import copy
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.ma.testutils import assert_array_equal

from tbp.monty.frameworks.experiments import (
    MontySupervisedObjectPretrainingExperiment,
)
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
    InvalidEvidenceThresholdConfig,
)


class ResamplingHypothesesUpdaterTest(TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.output_dir = tempfile.mkdtemp()

        with hydra.initialize(version_base=None, config_path="../../../../../conf"):
            self.cfg = hydra.compose(
                config_name="test",
                overrides=[
                    "test=frameworks/models/evidence_matching/resampling_hypothese_updater",
                    f"test.config.logging.output_dir={self.output_dir}",
                ],
            )

    def get_resampling_updater(self):
        train_config = copy.deepcopy(self.pretraining_configs)
        with MontySupervisedObjectPretrainingExperiment(train_config) as train_exp:
            train_exp.setup_experiment(train_exp.config)

        return train_exp.model.learning_modules[0].hypotheses_updater

    def get_pretrained_resampling_lm(self):
        exp = hydra.utils.instantiate(self.cfg.test)
        with exp:
            exp.run()

        rlm = exp.model.learning_modules[0]
        rlm.channel_hypothesis_mapping["capsule3DSolid"] = ChannelMapper()
        rlm.hypotheses_updater.evidence_slope_trackers["capsule3DSolid"] = (
            EvidenceSlopeTracker(min_age=0)
        )
        return rlm

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

    def _resampling_multiplier(self, rlm):
        """Tests that the resampling multiplier correctly scales the hypothesis space.

        The resampling multiplier parameter is used to scale the hypothesis space
        between steps. For example, a multiplier of 2, will request to increase the
        number of hypotheses by 2x the number of graph nodes.
        """
        graph_id = "capsule3DSolid"
        pose_defined = True
        graph_num_nodes = self._graph_node_count(rlm, graph_id)
        before_count = graph_num_nodes * self._num_hyps_multiplier(rlm, pose_defined)
        rlm.channel_hypothesis_mapping[graph_id].add_channel("patch", before_count)
        rlm.hypotheses_updater.evidence_slope_trackers[graph_id].add_hyp(
            before_count, "patch"
        )
        resampling_multipliers = [0.5, 1, 2]

        for resampling_multiplier in resampling_multipliers:
            _, informed_count = self.run_sample_count(
                rlm=rlm,
                resampling_multiplier=resampling_multiplier,
                deletion_trigger_slope=0.0,
                pose_defined=pose_defined,
                graph_id=graph_id,
            )
            self.assertEqual(graph_num_nodes * resampling_multiplier, informed_count)

        # Reset mapper
        rlm.channel_hypothesis_mapping[graph_id] = ChannelMapper()

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

    def test_sampling_count(self):
        """This function tests different aspects of _sample_count.

        We define three different tests of `_sample_count`:
            - Testing the requested count for initialization of hypotheses space
            - Testing the resampling multiplier parameter
            - Testing the resampling multiplier parameter maximum limit
        """
        rlm = self.get_pretrained_resampling_lm()

        # test count multiplier
        self._resampling_multiplier(rlm)
        self._resampling_multiplier_maximum(rlm, pose_defined=True)
        self._resampling_multiplier_maximum(rlm, pose_defined=False)

        # test existing to informed ratio
        self._count_ratio(rlm, pose_defined=True)
        self._count_ratio(rlm, pose_defined=False)


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
        tracker1.removable_indices_mask = Mock(
            return_value=np.ones((channel_size,), dtype=np.bool_)
        )
        tracker1.calculate_keep_and_remove_ids = Mock(
            return_value=(
                # keep_ids
                np.array([False, True, True, False, False]),
                # remove_ids
                np.array([True, False, False, True, True]),
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
