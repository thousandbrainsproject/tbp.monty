# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import copy

import numpy as np

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.goal_state_generation import EvidenceGoalStateGenerator
from tests.unit.resources.unit_test_utils import BaseGraphTest


class EvidenceLMTest(BaseGraphTest):
    def setUp(self):
        """Code that gets executed before every test."""
        super().setUp()

        self.default_gsg_config = dict(
            elapsed_steps_factor=10,
            min_post_goal_success_steps=5,
            x_percent_scale_factor=0.75,
            desired_object_distance=0.03,
        )

    def get_elm_with_fake_object(
        self, fake_obs, initial_possible_poses="informed", gsg=None
    ):
        graph_lm = EvidenceGraphLM(
            max_match_distance=0.005,
            tolerances={
                "patch": {
                    "hsv": [0.1, 1, 1],
                    "principal_curvatures_log": [1, 1],
                }
            },
            feature_weights={
                "patch": {
                    "hsv": np.array([1, 0, 0]),
                }
            },
            # set graph size larger since fake obs displacements are meters
            max_graph_size=10,
            gsg=gsg,
            hypotheses_updater_args=dict(
                initial_possible_poses=initial_possible_poses,
            ),
        )
        graph_lm.mode = ExperimentMode.TRAIN
        for observation in fake_obs:
            graph_lm.exploratory_step(self.ctx, [observation])
        graph_lm.detected_object = "new_object0"
        graph_lm.detected_rotation_r = None
        graph_lm.buffer.stats["detected_location_rel_body"] = (
            graph_lm.buffer.get_current_location(input_channel="first")
        )

        self.assertEqual(
            len(graph_lm.buffer.get_all_locations_on_object(input_channel="first")),
            len(fake_obs),
            f"Should have stored exactly {fake_obs} locations in the buffer.",
        )
        graph_lm.post_episode()
        self.assertEqual(
            len(graph_lm.get_all_known_object_ids()),
            1,
            "Should have stored exactly one object in memory.",
        )
        self.assertEqual(
            graph_lm.get_all_known_object_ids()[0],
            "new_object0",
            "Learned object ID should be new_object0.",
        )
        return graph_lm

    def get_elm_with_two_fake_objects(
        self,
        fake_obs,
        fake_obs_two,
        initial_possible_poses,
        gsg,
    ) -> EvidenceGraphLM:
        """Train on two fake observation objects.

        Returns:
            Evidence GraphLearning Module
        """
        # Train on first object
        graph_lm = self.get_elm_with_fake_object(
            fake_obs,
            initial_possible_poses=initial_possible_poses,
            gsg=gsg,
        )

        # Train on second object
        obj_two_target = copy.deepcopy(self.placeholder_target)
        obj_two_target["object"] = "new_object1"
        graph_lm.pre_episode(primary_target=obj_two_target)
        for observation in fake_obs_two:
            graph_lm.exploratory_step(self.ctx, [observation])
        graph_lm.detected_object = obj_two_target["object"]
        graph_lm.detected_rotation_r = None
        graph_lm.buffer.stats["detected_location_rel_body"] = (
            graph_lm.buffer.get_current_location(input_channel="first")
        )

        self.assertEqual(
            len(graph_lm.buffer.get_all_locations_on_object(input_channel="first")),
            len(fake_obs_two),
            f"Should have stored exactly {fake_obs_two} locations in the buffer.",
        )
        graph_lm.post_episode()
        self.assertEqual(
            len(graph_lm.get_all_known_object_ids()),
            2,
            "Should have stored two objects in memory now.",
        )
        self.assertEqual(
            graph_lm.get_all_known_object_ids()[0],
            "new_object0",
            "Should still know first object.",
        )
        self.assertEqual(
            graph_lm.get_all_known_object_ids()[1],
            "new_object1",
            "Learned object ID should be new_object1.",
        )

        return graph_lm

    def test_symmetry_recognition(self):
        """Test that symmetry is recognized."""
        fake_obs_test = copy.deepcopy(self.fake_obs_symmetric)
        # Get LM with object learned from fake_obs
        graph_lm = self.get_elm_with_fake_object(self.fake_obs_symmetric)

        graph_lm.mode = ExperimentMode.EVAL
        # Don't need to give target object since we are not logging performance
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        num_steps_checked_symmetry = 0
        for i in range(12):
            observation = fake_obs_test[i % 4]
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step(self.ctx, [observation])
            # since we don't have a monty class here we have to call this check
            # manually. Usually monty class coordinates terminal condition checks and
            # updates to symmetry count.
            graph_lm.get_unique_pose_if_available("new_object0")
            max_obj_evidence = np.max(graph_lm.evidence["new_object0"])
            if max_obj_evidence > graph_lm.object_evidence_threshold:
                num_steps_checked_symmetry += 1
                # On the first step we just store previous hypothesis ids.
                if num_steps_checked_symmetry > 1:
                    # On the second step we will add 1 symmetry evidence
                    # every step because we can't resolve between
                    # 0,0,0 and 180, 0, 180 rotation.
                    self.assertEqual(
                        graph_lm.symmetry_evidence,
                        num_steps_checked_symmetry - 1,
                        "Symmetry evidence doesn't seem to be as expected.",
                    )
            self.assertEqual(
                len(graph_lm.get_possible_matches()),
                1,
                "Should have exactly one possible match.",
            )
            self.assertEqual(
                graph_lm.get_possible_matches()[0],
                "new_object0",
                "Should match to new_object0.",
            )
        self.assertListEqual(
            list(graph_lm.get_possible_poses()["new_object0"][0]),
            [0, 0, 0],
            "Should have rotation 0, 0, 0 as a possible pose.",
        )
        symmetry_pose = np.mod(
            np.array(graph_lm.get_possible_poses()["new_object0"][-1], dtype=float),
            360.0,
        )
        self.assertTrue(
            np.allclose(symmetry_pose, np.array([180.0, 0.0, 180.0])),
            "Since we have symmetry here, 180, 0, 180 should also be a possible pose.",
        )

    def test_same_sequence_recognition_elm(self):
        """Test that the object is recognized with same action sequence."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = ExperimentMode.EVAL
        # Don't need to give target object since we are not logging performance
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        target_evidence = 1
        for observation in fake_obs_test:
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step(self.ctx, [observation])
            self.assertEqual(
                len(graph_lm.get_possible_matches()),
                1,
                "Should have exactly one possible match.",
            )
            self.assertEqual(
                graph_lm.get_possible_matches()[0],
                "new_object0",
                "Should match to new_object0.",
            )
            self.assertEqual(
                graph_lm.get_current_mlh()["graph_id"],
                "new_object0",
                "new_object0 should be the mlh at every step.",
            )
            self.assertEqual(
                graph_lm.get_current_mlh()["evidence"],
                target_evidence,
                "Since we have exact matches, evidence should be incremented by "
                "2 at each step (1 for pose match and 1 for feature match).",
            )
            target_evidence += 2

        self.assertListEqual(
            list(graph_lm.get_current_mlh()["rotation"].as_euler("xyz", degrees=True)),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_reverse_sequence_recognition_elm(self):
        """Test that object is recognized irrespective of sampling order."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test.reverse()

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        target_evidence = 1
        for observation in fake_obs_test:
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step(self.ctx, [observation])
            self.assertEqual(
                len(graph_lm.get_possible_matches()),
                1,
                "Should have exactly one possible match.",
            )
            self.assertEqual(
                graph_lm.get_possible_matches()[0],
                "new_object0",
                "Should match to new_object0.",
            )
            self.assertEqual(
                graph_lm.get_current_mlh()["graph_id"],
                "new_object0",
                "new_object0 should be the mlh at every step.",
            )
            self.assertEqual(
                graph_lm.get_current_mlh()["evidence"],
                target_evidence,
                "Since we have exact matches, evidence should be incremented by "
                "2 at each step (1 for pose match and 1 for feature match).",
            )
            target_evidence += 2

        self.assertListEqual(
            list(graph_lm.get_current_mlh()["rotation"].as_euler("xyz", degrees=True)),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_offset_sequence_recognition_elm(self):
        """Test that the object is recognized irrespective of its location rel body."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        target_evidence = 1
        for observation in fake_obs_test:
            observation.location = observation.location + np.ones(3)
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step(self.ctx, [observation])
            self.assertEqual(
                len(graph_lm.get_possible_matches()),
                1,
                "Should have exactly one possible match.",
            )
            self.assertEqual(
                graph_lm.get_possible_matches()[0],
                "new_object0",
                "Should match to new_object0.",
            )
            self.assertEqual(
                graph_lm.get_current_mlh()["graph_id"],
                "new_object0",
                "new_object0 should be the mlh at every step.",
            )
            self.assertEqual(
                graph_lm.get_current_mlh()["evidence"],
                target_evidence,
                "Since we have exact matches, evidence should be incremented by "
                "2 at each step (1 for pose match and 1 for feature match).",
            )
            target_evidence += 2

        self.assertListEqual(
            list(graph_lm.get_current_mlh()["rotation"].as_euler("xyz", degrees=True)),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_new_sampling_recognition_elm(self):
        """Test object recognition with slightly perturbed observations.

        Test that the object is recognized if observations don't exactly fit
        the model.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test[0].location = fake_obs_test[0].location + np.ones(3) * 0.001
        fake_obs_test[2].location = fake_obs_test[2].location + np.ones(3) * 0.002

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        for observation in fake_obs_test:
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step(self.ctx, [observation])
            self.assertEqual(
                len(graph_lm.get_possible_matches()),
                1,
                "Should have exactly one possible match.",
            )
            self.assertEqual(
                graph_lm.get_possible_matches()[0],
                "new_object0",
                "Should match to new_object0.",
            )
            self.assertEqual(
                graph_lm.get_current_mlh()["graph_id"],
                "new_object0",
                "new_object0 should be the mlh at every step.",
            )

        self.assertListEqual(
            list(graph_lm.get_current_mlh()["rotation"].as_euler("xyz", degrees=True)),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_different_locations_not_recognized_elm(self):
        """Test that the object is not recognized if locations don't match."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test[0].location = fake_obs_test[0].location + np.ones(3) * 5

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        for i, observation in enumerate(fake_obs_test):
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step(self.ctx, [observation])
            if i < 2:
                self.assertEqual(
                    len(graph_lm.get_possible_matches()),
                    1,
                    "Should have one match at the first and second step.",
                )
            else:
                self.assertEqual(
                    len(graph_lm.get_possible_matches()),
                    0,
                    "Should have no possible matches.",
                )

    def test_channel_mapper_shape_elm(self):
        """Test that the channel mapper matches evidence keys and shape."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
        graph_lm.matching_step(self.ctx, [fake_obs_test[0]])

        self.assertEqual(
            graph_lm.evidence.keys(),
            graph_lm.channel_hypothesis_mapping.keys(),
            "Graph ID should match.",
        )

        self.assertEqual(
            graph_lm.evidence["new_object0"].shape[0],
            graph_lm.channel_hypothesis_mapping["new_object0"].total_size,
            "Channel mapper should have the total number of hypotheses in evidence",
        )

    def _evaluate_target_location(
        self, graph_lm, fake_obs_test, target_object, focus_on_pose=False
    ):
        """Helper function for hypothesis testing that retreives a target location."""
        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.pre_episode(primary_target=self.placeholder_target)

        # Observe 4 / 5 of the available features
        for ii in range(4):
            observation = fake_obs_test[ii]
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step(self.ctx, [observation])

        if not focus_on_pose:
            # Since up to now we had identical evidence for both cube and house, we give
            # the house an edge now so we test for it and get the expected result.
            graph_lm.current_mlh["graph_id"] = "new_object1"
            graph_lm.evidence["new_object1"] += 1

        # Based on most recent observation, propose the most misaligned graph
        # sub-regions
        graph_lm.gsg.focus_on_pose = focus_on_pose
        target_loc_id = graph_lm.gsg._compute_graph_mismatch()

        target_graph = graph_lm.get_graph(target_object)
        first_input_channel = graph_lm.get_input_channels_in_graph(target_object)[0]
        target_loc = target_graph[first_input_channel].pos[target_loc_id]

        assert np.all(np.isclose(target_loc, self.fake_obs_house[4].location)), (
            "Should propose testing 5th (indexed from 0) location on object"
        )

    def test_hypothesis_testing_proposal_for_id(self):
        """Test that the LM correctly predicts a location on a graph to test.

        Test that the LM correctly predicts a location on a graph to test, given
        the mismatch between the top two most likely graphs.

        The "square" object is made up of four simple features, while the "house"
        object is a 2D square with an additional, 5th feature above (like the point
        on a house's roof); these two objects are a 2D analogue for distinguishing e.g.
        a mug with a handle from a cylindrical can.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_house)

        graph_lm = self.get_elm_with_two_fake_objects(
            self.fake_obs_square,
            self.fake_obs_house,
            initial_possible_poses=[[0, 0, 0]],  # Note we isolate the influence of
            # ambiguous pose on the hypothesis testing
            gsg=EvidenceGoalStateGenerator(**self.default_gsg_config),
        )

        self._evaluate_target_location(
            graph_lm, fake_obs_test, target_object="new_object1"
        )

    def test_hypothesis_testing_proposal_for_id_with_transformation(self):
        """Test that the LM correctly predicts a location on a graph to test.

        Test that the LM correctly predicts a location on a graph to test, given
        the mismatch between the top two most likely graphs.

        As for test_hypothesis_testing_proposal_for_id, but in this case, the
        "house" object has been rotated and translated in the environment, and we check
        that the policy still proposes the correct location in object-centric
        coordinates.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_house_trans)

        graph_lm = self.get_elm_with_two_fake_objects(
            self.fake_obs_square,
            self.fake_obs_house,
            initial_possible_poses=[[45, 75, 190]],  # Note we isolate the influence of
            # ambiguous pose on the hypothesis testing
            gsg=EvidenceGoalStateGenerator(**self.default_gsg_config),
        )

        self._evaluate_target_location(
            graph_lm, fake_obs_test, target_object="new_object1"
        )

    def test_hypothesis_testing_proposal_for_pose(self):
        """Test that the LM correctly predicts a location on a graph to test.

        Test that the LM correctly predicts a location on a graph to test, given
        the mismatch between the top two most likely poses of a given object.

        In this case, the house can be either right-side up or upside-down, so the
        policy should select the tip of the house's roof to disambiguate this.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_house)

        # Only trained on one object
        graph_lm = self.get_elm_with_fake_object(
            self.fake_obs_house,
            initial_possible_poses=[[0, 0, 0], [0, 0, 180]],
            # Note pose *is* ambiguous in this unti test, vs. in proposal_for_id; in
            # particular, house can either be right-side up, or upside-down (rotated
            # about z)
            gsg=EvidenceGoalStateGenerator(**self.default_gsg_config),
        )

        self._evaluate_target_location(
            graph_lm, fake_obs_test, target_object="new_object0", focus_on_pose=True
        )

    def test_hypothesis_testing_proposal_for_pose_with_transformation(self):
        """Test that the LM correctly predicts a location on a graph to test.

        Test that the LM correctly predicts a location on a graph to test, given
        the mismatch between the top two most likely poses of a given object.

        In this case, the house can be either right-side up or upside-down, so the
        policy should select the tip of the house's roof to disambiguate this.
        In addition, the house has been transformed in environmental coordinates
        (translated and rotated), and we confirm that the object-centric target
        point is still correct.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_house_trans)

        # Only trained on one object
        graph_lm = self.get_elm_with_fake_object(
            self.fake_obs_house,
            initial_possible_poses=[[45, 75, 190], [315, 285, 10]],
            # Note pose *is* ambiguous in this unti test, vs. in proposal_for_id; in
            # particular, house can either be right-side up, or upside-down (was rotated
            # about z before the additional complex transformation was applied)
            gsg=EvidenceGoalStateGenerator(**self.default_gsg_config),
        )

        self._evaluate_target_location(
            graph_lm, fake_obs_test, target_object="new_object0", focus_on_pose=True
        )

    def test_different_features_still_recognized(self):
        """Test that the object is still recognized if features don't match.

        The evidence LM just uses features as an aid to recognize objects faster
        but should still be able to recognize an object if features don't match
        as long as the morphology matches.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test[0].non_morphological_features["hsv"] = [0.5, 1, 1]
        fake_obs_test[1].non_morphological_features["hsv"] = [0.9, 1, 1]
        fake_obs_test[2].non_morphological_features["hsv"] = [0.5, 1, 1]
        fake_obs_test[3].non_morphological_features["hsv"] = [0.7, 1, 1]

        for i in range(4):
            fake_obs_test[i].non_morphological_features["principal_curvatures_log"] = [
                10,
                10,
            ]

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        # We start at evidence 0 since we don't get feature evidence at initialization
        for target_evidence, observation in enumerate(fake_obs_test):
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step(self.ctx, [observation])
            self.assertEqual(
                len(graph_lm.get_possible_matches()),
                1,
                "Should have exactly 1 possible match.",
            )
            self.assertEqual(
                graph_lm.get_current_mlh()["evidence"],
                target_evidence,
                "Since none of the features match we should only be getting evidence "
                "from morphology which is 1 at each step (since m matches perfectly).",
            )

        self.assertListEqual(
            list(graph_lm.get_current_mlh()["rotation"].as_euler("xyz", degrees=True)),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_different_pose_features_not_recognized_elm(self):
        """Test that the object is not recognized if pose features don't match."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test[0].morphological_features["pose_vectors"] = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        for step, observation in enumerate(fake_obs_test):
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step(self.ctx, [observation])
            if step == 0:
                self.assertEqual(
                    len(graph_lm.get_possible_matches()),
                    1,
                    "Should have one possible match at first step.",
                )
                self.assertEqual(
                    graph_lm.get_current_mlh()["evidence"],
                    1,
                    "First step should have evidence 1 since features match and poses"
                    " are just being initialized.",
                )
            elif step == 1:
                self.assertGreater(
                    graph_lm.get_current_mlh()["evidence"],
                    1,
                    "Second step should still have evidence >1 since initialized pose"
                    " is still possible.",
                )
            elif step == 2:
                self.assertLess(
                    graph_lm.get_current_mlh()["evidence"],
                    1,
                    "Third step should have evidence <1 since initialized pose"
                    " is not valid anymore -> subtracting 1.",
                )
            elif step == 3:
                self.assertLess(
                    graph_lm.get_current_mlh()["evidence"],
                    0,
                    "Fourth step should have evidence <0 since initialized pose"
                    " is not valid anymore -> subtracting 1.",
                )
                self.assertEqual(
                    len(graph_lm.get_possible_matches()),
                    0,
                    "Should have no possible matches.",
                )

    def test_moving_off_object_and_back_elm(self):
        """Test that the object is still recognized after moving off the object.

        Test that the object is still recognized after moving off the object
        and that evidence is not incremented in that step.

        TODO: since the monty class checks use_state in combine_inputs it doesn't make
        much sense to test this here anymore with an isolated LM.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test[1].location = [1, 2, 1]
        fake_obs_test[1].morphological_features["on_object"] = 0
        fake_obs_test[1].use_state = False

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.pre_episode(primary_target=self.placeholder_target)
        target_evidence = 1
        for step, observation in enumerate(fake_obs_test):
            if not observation.use_state:
                pass
            else:
                graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
                graph_lm.matching_step(self.ctx, [observation])
                # If we are off the object, no evidence should be added
                if observation.morphological_features["on_object"] != 0 and step > 0:
                    target_evidence += 2
                self.assertEqual(
                    len(graph_lm.get_possible_matches()),
                    1,
                    "Should have exactly one possible match.",
                )
                self.assertEqual(
                    graph_lm.get_possible_matches()[0],
                    "new_object0",
                    "Should match to new_object0.",
                )
                self.assertEqual(
                    graph_lm.get_current_mlh()["graph_id"],
                    "new_object0",
                    "new_object0 should be the mlh at every step.",
                )
                self.assertEqual(
                    graph_lm.get_current_mlh()["evidence"],
                    target_evidence,
                    "Since we have exact matches, evidence should be incremented by "
                    "2 at each step (1 for pose match and 1 for feature match).",
                )

        self.assertListEqual(
            list(graph_lm.get_current_mlh()["rotation"].as_euler("xyz", degrees=True)),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )
