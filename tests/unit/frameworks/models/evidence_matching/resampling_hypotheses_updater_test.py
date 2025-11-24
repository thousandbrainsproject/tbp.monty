# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from collections import OrderedDict

import pytest

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

import copy
import tempfile
from unittest import TestCase

import numpy as np

from tbp.monty.frameworks.config_utils.config_args import (
    MontyFeatureGraphArgs,
    PatchAndViewMontyConfig,
    PretrainLoggingConfig,
)
from tbp.monty.frameworks.config_utils.make_env_interface_configs import (
    EnvironmentInterfacePerObjectTrainArgs,
    PredefinedObjectInitializer,
    SupervisedPretrainingExperimentArgs,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments.pretraining_experiments import (
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.evidence_matching.resampling_hypotheses_updater import (  # noqa: E501
    ResamplingHypothesesUpdater,
)
from tbp.monty.frameworks.utils.evidence_matching import (
    ChannelMapper,
    ConsistentHypothesesIds,
    EvidenceSlopeTracker,
)
from tbp.monty.simulators.habitat.configs import (
    EnvInitArgsPatchViewMount,
    PatchViewFinderMountHabitatEnvInterfaceConfig,
)


def make_consistent_ids(graph_id, sizes, ids):
    return ConsistentHypothesesIds(
        graph_id=graph_id,
        channel_sizes=OrderedDict(sizes),
        hypotheses_ids=np.asarray(ids, dtype=np.int64),
    )


class ResamplingHypothesesUpdaterTest(TestCase):
    def setUp(self) -> None:
        super().setUp()

        default_tolerances = {
            "hsv": np.array([0.1, 1, 1]),
            "principal_curvatures_log": np.ones(2),
        }

        resampling_lm_args = dict(
            max_match_distance=0.001,
            tolerances={"patch": default_tolerances},
            feature_weights={
                "patch": {
                    "hsv": np.array([1, 0, 0]),
                }
            },
            hypotheses_updater_class=ResamplingHypothesesUpdater,
        )

        default_evidence_lm_config = dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=resampling_lm_args,
            )
        )

        self.output_dir = tempfile.mkdtemp()

        self.pretraining_configs = dict(
            experiment_class=MontySupervisedObjectPretrainingExperiment,
            experiment_args=SupervisedPretrainingExperimentArgs(
                n_train_epochs=3,
            ),
            logging_config=PretrainLoggingConfig(
                output_dir=self.output_dir,
            ),
            monty_config=PatchAndViewMontyConfig(
                monty_class=MontyForEvidenceGraphMatching,
                monty_args=MontyFeatureGraphArgs(num_exploratory_steps=20),
                learning_module_configs=default_evidence_lm_config,
            ),
            env_interface_config=PatchViewFinderMountHabitatEnvInterfaceConfig(
                env_init_args=EnvInitArgsPatchViewMount(data_path=None).__dict__,
            ),
            train_env_interface_class=ED.InformedEnvironmentInterface,
            train_env_interface_args=EnvironmentInterfacePerObjectTrainArgs(
                object_names=["capsule3DSolid"],
                object_init_sampler=PredefinedObjectInitializer(),
            ),
        )

    def get_resampling_updater(self):
        train_config = copy.deepcopy(self.pretraining_configs)
        with MontySupervisedObjectPretrainingExperiment(train_config) as train_exp:
            train_exp.setup_experiment(train_exp.config)

        return train_exp.model.learning_modules[0].hypotheses_updater

    def get_pretrained_resampling_lm(self):
        train_config = copy.deepcopy(self.pretraining_configs)
        with MontySupervisedObjectPretrainingExperiment(train_config) as train_exp:
            train_exp.train()

        rlm = train_exp.model.learning_modules[0]
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

    def _single_channel_no_changes(self, updater):
        updater.resampling_telemetry = {
            "mug": {"patch": {"removed_ids": [], "added_ids": []}}
        }
        hyp_ids = make_consistent_ids(
            graph_id="mug", sizes=[("patch", 5)], ids=[0, 1, 3, 4]
        )

        hyp_ids = updater.remap_hypotheses_ids_to_present(hyp_ids)
        np.testing.assert_array_equal(hyp_ids.hypotheses_ids, np.array([0, 1, 3, 4]))

    def _single_channel_with_removals_shifts(self, updater):
        updater.resampling_telemetry = {
            "mug": {"patch": {"removed_ids": [1, 4, 6], "added_ids": []}}
        }
        hyp_ids = make_consistent_ids(
            graph_id="mug", sizes=[("patch", 8)], ids=[0, 2, 3, 5, 7]
        )
        hyp_ids = updater.remap_hypotheses_ids_to_present(hyp_ids)

        # Shift per searchsorted([1,4,6], x, 'left'): 0->0, 2->1, 3->1, 5->2, 7->3
        # new locals after shifting:  [0,1,2,3,4]
        np.testing.assert_array_equal(hyp_ids.hypotheses_ids, np.array([0, 1, 2, 3, 4]))

    def _single_channel_full_remap_misses_added(self, updater):
        """Tests that added ids do not show up in remapping.

        The remapping function finds the mapping between ids from the previous step
        to the current time step. The added_ids did not exist in previous steps,
        therefore should not appear in the mapping.
        """
        added_ids = [5, 6]

        updater.resampling_telemetry = {
            "mug": {"patch": {"removed_ids": [], "added_ids": added_ids}}
        }
        hyp_ids = make_consistent_ids(
            graph_id="mug", sizes=[("patch", 5)], ids=list(range(5))
        )
        hyp_ids = updater.remap_hypotheses_ids_to_present(hyp_ids)

        # In patch0: locals ids = [0,1,2,3,4]; removed = []; shift = [0,0,0,0,0].
        # So [0,1,2,3,4] becomes [0,1,2,3,4]
        np.testing.assert_array_equal(hyp_ids.hypotheses_ids, np.array([0, 1, 2, 3, 4]))

        # Added ids should NOT appear in the remapped ids
        self.assertFalse(np.isin(added_ids, hyp_ids.hypotheses_ids).any())

    def _multi_channel_rebase_due_to_resizing(self, updater):
        updater.resampling_telemetry = {
            "mug": {
                "patch0": {"removed_ids": [1, 3], "added_ids": [5, 6]},
                "patch1": {"removed_ids": [2], "added_ids": [4]},
            }
        }
        hyp_ids = make_consistent_ids(
            graph_id="mug", sizes=[("patch0", 5), ("patch1", 4)], ids=[0, 2, 4, 5, 7]
        )
        hyp_ids = updater.remap_hypotheses_ids_to_present(hyp_ids)

        # In patch0: locals ids = [0,2,4]; removed = [1,3]; shift = [0,1,2].
        # So [0, 2, 4] becomes [0, 1, 2]

        # new bases are the same since patch0 removed 2 and added 2.
        # So new_bases = [0,5]

        # In patch1: locals ids = [0,2]; removed = [2]; new base = 5
        # So [5, 7] becomes [5]
        np.testing.assert_array_equal(hyp_ids.hypotheses_ids, np.array([0, 1, 2, 5]))

    def _rebase_when_first_channel_shrinks(self, updater):
        updater.resampling_telemetry = {
            "mug": {
                "patch0": {"removed_ids": [1, 3], "added_ids": []},  # shrink by 2
                "patch1": {"removed_ids": [], "added_ids": []},
            }
        }
        hyp_ids = make_consistent_ids(
            graph_id="mug",
            sizes=[("patch0", 5), ("patch1", 4)],
            ids=[0, 2, 4, 5, 7],
        )
        hyp_ids = updater.remap_hypotheses_ids_to_present(hyp_ids)

        # In patch0: local = [0,2,4]; removed = [1,3]; shifts = [0,1,2]
        # So [0,2,4] becomes [0,1,2]

        # New bases: [0,3] so patch1 base is 3 (not 5)

        # In patch 1: locals = [0,2]
        # So [5,7] becomes [3,5]
        np.testing.assert_array_equal(hyp_ids.hypotheses_ids, np.array([0, 1, 2, 3, 5]))

    def _rebase_when_first_channel_grows(self, updater):
        updater.resampling_telemetry = {
            "mug": {
                "patch0": {"removed_ids": [], "added_ids": [5, 6]},  # grow by 2
                "patch1": {"removed_ids": [], "added_ids": []},
            }
        }
        hyp_ids = make_consistent_ids(
            graph_id="mug",
            sizes=[("patch0", 5), ("patch1", 4)],
            ids=[0, 4, 5, 6, 8],
        )
        out = updater.remap_hypotheses_ids_to_present(hyp_ids)

        # In patch0: local = [0,4]; added = [5,6]; No shifts
        # So [0,4] becomes [0,4]

        # New bases: [0,7]

        # In patch 1: locals = [0,2,4]
        # So [0,1,3] becomes [7,8,10]
        np.testing.assert_array_equal(out.hypotheses_ids, np.array([0, 4, 7, 8, 10]))

    def _all_ids_removed_in_a_channel(self, updater):
        updater.resampling_telemetry = {
            "mug": {
                "patch0": {"removed_ids": [0, 1, 2], "added_ids": []},
                "patch1": {"removed_ids": [], "added_ids": []},
            }
        }
        hyp_ids = make_consistent_ids(
            graph_id="mug", sizes=[("patch0", 3), ("patch1", 3)], ids=[0, 1, 2, 3, 4, 5]
        )
        hyp_ids = updater.remap_hypotheses_ids_to_present(hyp_ids)

        # Removed [0, 1, 2], so [3, 4, 5] was rebased to [0, 1, 2]
        np.testing.assert_array_equal(hyp_ids.hypotheses_ids, np.array([0, 1, 2]))

    def test_remap_hypotheses_ids(self):
        updater = self.get_resampling_updater()

        self._single_channel_no_changes(updater)
        self._single_channel_with_removals_shifts(updater)
        self._single_channel_full_remap_misses_added(updater)
        self._multi_channel_rebase_due_to_resizing(updater)
        self._rebase_when_first_channel_shrinks(updater)
        self._rebase_when_first_channel_grows(updater)
        self._all_ids_removed_in_a_channel(updater)
