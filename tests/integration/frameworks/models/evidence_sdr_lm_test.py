# Copyright 2025-2026 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import copy
import shutil
import tempfile

import numpy as np

from tbp.monty.cmp import Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.evidence_sdr_matching import EvidenceSDRGraphLM
from tbp.monty.frameworks.models.goal_generation import EvidenceGoalGenerator
from tests.unit.resources.unit_test_utils import BaseGraphTest


def set_seed(seed):
    """Set seed for reproducibility."""
    np.random.seed(seed)


class EvidenceSDRIntegrationTest(BaseGraphTest):
    def setUp(self):
        """Setup function at the beginning of each experiment."""
        self.output_dir = tempfile.mkdtemp()

        # set seed for reproducibility
        seed = 42
        set_seed(seed)
        self.ctx = RuntimeContext(rng=np.random.RandomState(seed))

        self.default_obs_args = dict(
            location=np.array([0.0, 0.0, 0.0]),
            morphological_features={
                "pose_vectors": np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
                "pose_fully_defined": True,
                "on_object": 1,
            },
            non_morphological_features={
                "principal_curvatures_log": [0, 0.5],
                "hsv": [0, 1, 1],
            },
            confidence=1.0,
            use_state=True,
            sender_id="patch",
            sender_type="SM",
        )

    def get_rectangle_percepts(self) -> list[Message]:
        """Helper function to create percepts for a rectangle object.

        Returns:
            List of percepts
        """
        fo_0 = copy.deepcopy(self.default_obs_args)

        fo_1 = copy.deepcopy(self.default_obs_args)
        fo_1["location"] = np.array([1.0, 0.0, 0.0])

        fo_2 = copy.deepcopy(self.default_obs_args)
        fo_2["location"] = np.array([2.0, 0.0, 0.0])

        fo_3 = copy.deepcopy(self.default_obs_args)
        fo_3["location"] = np.array([0.0, 1.0, 0.0])

        fo_4 = copy.deepcopy(self.default_obs_args)
        fo_4["location"] = np.array([1.0, 1.0, 0.0])

        fo_5 = copy.deepcopy(self.default_obs_args)
        fo_5["location"] = np.array([2.0, 1.0, 0.0])

        return [
            Message(**fo_0),
            Message(**fo_1),
            Message(**fo_2),
            Message(**fo_3),
            Message(**fo_4),
            Message(**fo_5),
        ]

    def get_rectangle_long_percepts(self) -> list[Message]:
        """Helper function to create percepts for a long rectangle object.

        Returns:
            List of percepts.
        """
        fo_0 = copy.deepcopy(self.default_obs_args)

        fo_1 = copy.deepcopy(self.default_obs_args)
        fo_1["location"] = np.array([1.0, 0.0, 0.0])

        fo_2 = copy.deepcopy(self.default_obs_args)
        fo_2["location"] = np.array([2.0, 0.0, 0.0])

        fo_3 = copy.deepcopy(self.default_obs_args)
        fo_3["location"] = np.array([3.0, 0.0, 0.0])

        fo_4 = copy.deepcopy(self.default_obs_args)
        fo_4["location"] = np.array([0.0, 1.0, 0.0])

        fo_5 = copy.deepcopy(self.default_obs_args)
        fo_5["location"] = np.array([1.0, 1.0, 0.0])

        fo_6 = copy.deepcopy(self.default_obs_args)
        fo_6["location"] = np.array([2.0, 1.0, 0.0])

        fo_7 = copy.deepcopy(self.default_obs_args)
        fo_7["location"] = np.array([3.0, 1.0, 0.0])

        return [
            Message(**fo_0),
            Message(**fo_1),
            Message(**fo_2),
            Message(**fo_3),
            Message(**fo_4),
            Message(**fo_5),
            Message(**fo_6),
            Message(**fo_7),
        ]

    def get_triangle_percepts(self) -> list[Message]:
        """Helper function to create percepts for a traingle object.

        Returns:
            List of percepts.
        """
        fo_0 = copy.deepcopy(self.default_obs_args)

        fo_1 = copy.deepcopy(self.default_obs_args)
        fo_1["location"] = np.array([1.0, 0.0, 0.0])
        fo_2 = copy.deepcopy(self.default_obs_args)
        fo_2["location"] = np.array([0.5, 1.0, 0.0])
        return [
            Message(**fo_0),
            Message(**fo_1),
            Message(**fo_2),
        ]

    def get_eslm(self) -> EvidenceSDRGraphLM:
        """Helper function to return an Evidence SDR Learning Module.

        Returns:
            Evidence SDR Graph Learning Module.
        """
        return EvidenceSDRGraphLM(
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
            gsg=EvidenceGoalGenerator(
                elapsed_steps_factor=10,
                min_post_goal_success_steps=5,
                x_percent_scale_factor=0.75,
                desired_object_distance=0.03,
            ),
            hypotheses_updater_args=dict(
                initial_possible_poses=[[0, 0, 0]],
            ),
            sdr_args=dict(
                log_path=None,  # Temporary log path
                sdr_length=2048,  # Size of SDRs
                sdr_on_bits=41,  # Number of active bits in the SDRs
                sdr_lr=1e-2,  # Learning rate of the encoding algorithm
                n_sdr_epochs=1000,  # Number of training epochs per episode
                sdr_log_flag=True,  # log the output of the module
            ),
        )

    def learn_obj(self, lm, percepts, obj_name):
        """Helper function to learn a new object.

        Learns a new object from observations and adds it to the existing graphs.
        """
        obj_target = {
            "object": obj_name,
            "quat_rotation": [1, 0, 0, 0],
        }

        lm.mode = ExperimentMode.TRAIN
        lm.pre_episode(primary_target=obj_target)
        for percept in percepts:
            lm.exploratory_step(self.ctx, [percept])
        lm.detected_object = obj_name
        lm.detected_rotation_r = None
        lm.buffer.stats["detected_location_rel_body"] = lm.buffer.get_current_location(
            input_channel="first"
        )

        self.assertEqual(
            len(lm.buffer.get_all_locations_on_object(input_channel="first")),
            len(percepts),
            f"Should have stored exactly {len(percepts)} locations in the buffer.",
        )
        lm.post_episode()

    def match(self, lm, percepts):
        """Matching function without action policy and gsg.

        Note: Observations are fed to the LM without the need to
        suggest new location in this toy example.
        """
        first_movement_detected = lm._agent_moved_since_reset()
        buffer_data = lm._add_displacements(percepts)
        lm.buffer.append(buffer_data)
        lm.buffer.append_input_percepts(percepts)

        lm._compute_possible_matches(
            self.ctx, percepts, first_movement_detected=first_movement_detected
        )

        if len(lm.get_possible_matches()) == 0:
            lm.set_individual_ts(terminal_state="no_match")

        stats = lm.collect_stats_to_save()
        lm.buffer.update_stats(stats, append=lm.has_detailed_logger)

    def eval_obj(self, lm, percepts):
        """Helper function to match new object after learning."""
        placeholder_target = {
            "object": "placeholder",
            "quat_rotation": [1, 0, 0, 0],
        }

        lm.mode = ExperimentMode.EVAL
        lm.pre_episode(primary_target=placeholder_target)
        for percept in percepts[:-1]:
            lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            self.match(lm, [percept])

        lm.post_episode()

    def test_can_generate_reasonable_sdrs(self):
        """Test ability to generate reasonable SDRs.

        This test focuses on evaluating the ability of the model to generate
        reasonable SDRs from toy example of 2D shapes.

        Expected behavior:
            - Model should output SDRs with high overlaps
                between rectangle and rectangle_long
            - Model should output SDRs with low overlaps
                between the triangle and both rectangles
        """
        # get the evidence sdr learning module
        eslm = self.get_eslm()

        # define objects and their percepts
        objects = [
            self.get_triangle_percepts(),
            self.get_rectangle_percepts(),
            self.get_rectangle_long_percepts(),
        ]

        # learn each object and add graphs to memory
        for object_id, percepts in enumerate(objects):
            self.learn_obj(eslm, percepts, f"new_object{object_id}")

        # test to check objects are learned
        self.assertEqual(
            len(eslm.get_all_known_object_ids()),
            len(objects),
            f"Should have stored exactly {len(objects)} objects in memory.",
        )

        # matching phase
        for percepts in objects:
            self.eval_obj(eslm, percepts)

        # test that the correct number of object representations
        # exist in Evidence SDR LM
        self.assertEqual(eslm.sdr_encoder.n_objects, len(objects))

        # test that all sdrs have the correct sdr_length
        self.assertTrue(eslm.sdr_encoder.sdrs.shape[-1] == 2048)

        # test that all sdrs have the correct sdr_on_bits
        self.assertTrue(np.all(eslm.sdr_encoder.sdrs.sum(-1) == 41))

        # test output SDRs. Rectangles should be clustered together
        sdrs = eslm.sdr_encoder.sdrs
        overlaps = sdrs @ sdrs.T
        self.assertTrue(
            overlaps[1, 2] > overlaps[0, 2] and overlaps[1, 2] > overlaps[0, 1]
        )

    def tearDown(self):
        """Tear down function at the end of each experiment."""
        super().tearDown()
        shutil.rmtree(self.output_dir)
