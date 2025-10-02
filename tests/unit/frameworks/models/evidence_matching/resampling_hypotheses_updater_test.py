# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
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
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataLoaderPerObjectTrainArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
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
    EvidenceSlopeTracker,
)
from tbp.monty.simulators.habitat.configs import (
    EnvInitArgsPatchViewMount,
    PatchViewFinderMountHabitatDatasetArgs,
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
            experiment_args=ExperimentArgs(
                do_eval=False,
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
            dataset_args=PatchViewFinderMountHabitatDatasetArgs(
                env_init_args=EnvInitArgsPatchViewMount(data_path=None).__dict__,
            ),
            train_dataloader_class=ED.InformedEnvironmentDataLoader,
            train_dataloader_args=EnvironmentDataLoaderPerObjectTrainArgs(
                object_names=["capsule3DSolid"],
                object_init_sampler=PredefinedObjectInitializer(),
            ),
        )

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
        graph_num_points = rlm.graph_memory.get_locations_in_graph(
            graph_id, "patch"
        ).shape[0]
        return graph_num_points

    def _num_hyps_multiplier(self, rlm, pose_defined):
        """Returns the expected hyps multiplier based on Principal curvatures."""
        return 2 if pose_defined else rlm.hypotheses_updater.umbilical_num_poses

    def run_sample_count(
        self,
        rlm,
        resampling_multiplier,
        evidence_slope_threshold,
        pose_defined,
        graph_id,
    ):
        rlm.hypotheses_updater.resampling_multiplier = resampling_multiplier
        rlm.hypotheses_updater.evidence_slope_threshold = evidence_slope_threshold
        test_features = {"patch": {"pose_fully_defined": pose_defined}}
        return rlm.hypotheses_updater._sample_count(
            input_channel="patch",
            channel_features=test_features["patch"],
            graph_id=graph_id,
            mapper=rlm.channel_hypothesis_mapping[graph_id],
            tracker=rlm.hypotheses_updater.evidence_slope_trackers[graph_id],
        )

    def _initial_count(self, rlm, pose_defined):
        """This tests that the initial requested number of hypotheses is correct.

        In order to initialize a hypothesis space, the `_sample_count` should request
        that all resampled hypotheses be of the type informed. This tests the informed
        sampling with defined and undefined poses.
        """
        graph_id = "capsule3DSolid"
        hypotheses_selection, informed_count = self.run_sample_count(
            rlm=rlm,
            resampling_multiplier=0.1,
            evidence_slope_threshold=0.0,
            pose_defined=pose_defined,
            graph_id=graph_id,
        )
        self.assertEqual(len(hypotheses_selection.maintain_ids), 0)
        self.assertEqual(
            informed_count,
            self._graph_node_count(rlm, graph_id)
            * self._num_hyps_multiplier(rlm, pose_defined),
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
            hypotheses_selection, informed_count = self.run_sample_count(
                rlm=rlm,
                resampling_multiplier=resampling_multiplier,
                evidence_slope_threshold=0.0,
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
        hypotheses_selection, informed_count = self.run_sample_count(
            rlm=rlm,
            resampling_multiplier=resampling_multiplier,
            evidence_slope_threshold=0.0,
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

        # test initial count
        self._initial_count(rlm, pose_defined=True)
        self._initial_count(rlm, pose_defined=False)

        # test count multiplier
        self._resampling_multiplier(rlm)
        self._resampling_multiplier_maximum(rlm, pose_defined=True)
        self._resampling_multiplier_maximum(rlm, pose_defined=False)
