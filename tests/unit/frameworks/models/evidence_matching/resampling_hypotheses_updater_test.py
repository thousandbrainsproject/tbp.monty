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
    EvidenceSlopeTracker,
)
from tbp.monty.simulators.habitat.configs import (
    EnvInitArgsPatchViewMount,
    PatchViewFinderMountHabitatEnvInterfaceConfig,
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
        count_multiplier,
        existing_to_new_ratio,
        pose_defined,
        graph_id,
    ):
        rlm.hypotheses_updater.hypotheses_count_multiplier = count_multiplier
        rlm.hypotheses_updater.hypotheses_existing_to_new_ratio = existing_to_new_ratio
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
        existing_count, informed_count = self.run_sample_count(
            rlm=rlm,
            count_multiplier=1,
            existing_to_new_ratio=0.1,
            pose_defined=pose_defined,
            graph_id=graph_id,
        )
        self.assertEqual(existing_count, 0)
        self.assertEqual(
            informed_count,
            self._graph_node_count(rlm, graph_id)
            * self._num_hyps_multiplier(rlm, pose_defined),
        )

    def _count_multiplier(self, rlm):
        """This tests that the count multiplier correctly scales the hypothesis space.

        The count multiplier parameter is used to scale the hypothesis space between
        steps. For example, a multiplier of 2, will request to double the number of
        hypotheses.
        """
        graph_id = "capsule3DSolid"
        pose_defined = True
        graph_num_nodes = self._graph_node_count(rlm, graph_id)
        before_count = graph_num_nodes * self._num_hyps_multiplier(rlm, pose_defined)
        rlm.channel_hypothesis_mapping[graph_id].add_channel("patch", before_count)
        rlm.hypotheses_updater.evidence_slope_trackers[graph_id].add_hyp(
            before_count, "patch"
        )
        count_multipliers = [0.5, 1, 2]

        for count_multiplier in count_multipliers:
            existing_count, informed_count = self.run_sample_count(
                rlm=rlm,
                count_multiplier=count_multiplier,
                existing_to_new_ratio=0.5,
                pose_defined=pose_defined,
                graph_id=graph_id,
            )
            self.assertEqual(
                before_count * count_multiplier, (existing_count + informed_count)
            )

        # Reset mapper
        rlm.channel_hypothesis_mapping[graph_id] = ChannelMapper()

    def _count_multiplier_maximum(self, rlm, pose_defined):
        """This tests that the count multiplier respects the maximum scaling boundary.

        The count multiplier parameter is used to scale the hypothesis space between
        steps. For example, a multiplier of 2, will request to double the number of
        hypotheses. However, there is a limit to how many hypotheses we can resample.
        For existing hypotheses, the limit is to resample all of them. For newly
        resampled informed hypotheses, the limit depends on whether the pose is defined
        or not. This test ensures that `_sample_count` respects the maximum sampling
        limit.

        In the case of `pose_defined = True`
        Existing is 72 and informed is 2*36=72 (total is 144)
        Maximum multiplier can be 2 if the pose is defined

        In the case of `pose_defined = False`
        Existing is 72 and informed is 8*36=288 (total is 360)
        Maximum multiplier can be umbilical_num_poses if the pose is undefined
        """
        graph_id = "capsule3DSolid"
        graph_num_nodes = self._graph_node_count(rlm, graph_id)
        before_count = graph_num_nodes * self._num_hyps_multiplier(rlm, pose_defined)
        rlm.channel_hypothesis_mapping[graph_id].add_channel("patch", before_count)

        requested_count_multiplier = 100
        expected_count = before_count + (
            graph_num_nodes * self._num_hyps_multiplier(rlm, pose_defined)
        )
        existing_count, informed_count = self.run_sample_count(
            rlm=rlm,
            count_multiplier=requested_count_multiplier,
            existing_to_new_ratio=0.5,
            pose_defined=pose_defined,
            graph_id=graph_id,
        )
        self.assertEqual(expected_count, existing_count + informed_count)

        # Reset mapper
        rlm.channel_hypothesis_mapping[graph_id] = ChannelMapper()

    def _count_ratio(self, rlm, pose_defined):
        """This tests that the resampling ratio of new hypotheses is correct.

        The existing_to_new_ratio parameter is used to control the ratio of how many
        existing vs. informed hypotheses to resample. This test ensures that the
        `_sample_count` function follows the expected behavior of this ratio parameter.

        Note that the `_sample_count` function will prioritize the multiplier count
        parameter over this ratio parameter. In other words, if not enough existing
        hypotheses are available, the function will attempt to fill the missing
        existing hypotheses with informed hypotheses.

        """
        graph_id = "capsule3DSolid"
        graph_num_nodes = self._graph_node_count(rlm, graph_id)
        available_existing_count = graph_num_nodes * self._num_hyps_multiplier(
            rlm, pose_defined
        )
        rlm.channel_hypothesis_mapping[graph_id].add_channel(
            "patch", available_existing_count
        )
        rlm.hypotheses_updater.evidence_slope_trackers[graph_id].add_hyp(
            available_existing_count, "patch"
        )
        count_multiplier = 2

        for ratio in [0.0, 0.1, 0.5, 0.9, 1.0]:
            requested_existing_count = (
                available_existing_count * count_multiplier * (1.0 - ratio)
            )
            requested_informed_count = (
                available_existing_count * count_multiplier * ratio
            )
            maximum_available_existing_count = available_existing_count
            maximum_available_informed_count = (
                graph_num_nodes * self._num_hyps_multiplier(rlm, pose_defined)
            )

            existing_count, informed_count = self.run_sample_count(
                rlm=rlm,
                count_multiplier=count_multiplier,
                existing_to_new_ratio=ratio,
                pose_defined=pose_defined,
                graph_id=graph_id,
            )
            expected_existing_count = min(
                maximum_available_existing_count,
                requested_existing_count,
            )
            self.assertEqual(existing_count, int(expected_existing_count))

            # `missing_existing_hypotheses` will be zero, or otherwise the count that
            # informed hypotheses need to fill in
            missing_existing_hypotheses = (
                requested_existing_count - expected_existing_count
            )
            expected_informed_count = min(
                maximum_available_informed_count,
                (requested_informed_count + missing_existing_hypotheses),
            )
            self.assertEqual(informed_count, int(expected_informed_count))

        # Reset mapper
        rlm.channel_hypothesis_mapping[graph_id] = ChannelMapper()

    def test_sampling_count(self):
        """This function tests different aspects of _sample_count.

        We define three different tests of `_sample_count`:
            - Testing the requested count for initialization of hypotheses space
            - Testing the count multiplier parameter
            - Testing the count ratio of resampled hypotheses
        """
        rlm = self.get_pretrained_resampling_lm()

        # test initial count
        self._initial_count(rlm, pose_defined=True)
        self._initial_count(rlm, pose_defined=False)

        # test count multiplier
        self._count_multiplier(rlm)
        self._count_multiplier_maximum(rlm, pose_defined=True)
        self._count_multiplier_maximum(rlm, pose_defined=False)

        # test existing to informed ratio
        self._count_ratio(rlm, pose_defined=True)
        self._count_ratio(rlm, pose_defined=False)
