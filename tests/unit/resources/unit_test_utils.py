# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.models.states import State


class BaseGraphTestCases:
    class BaseGraphTest(unittest.TestCase):
        def setUp(self):
            print("setting up")
            fake_sender_id = "patch"

            default_obs_args = dict(
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
                sender_id=fake_sender_id,
                sender_type="SM",
            )
            fo_1 = copy.deepcopy(default_obs_args)
            fo_1["location"] = np.array([1.0, 0.0, 0.0])
            fo_2 = copy.deepcopy(default_obs_args)
            fo_2["location"] = np.array([1.0, 1.0, 0.0])
            fo_3 = copy.deepcopy(default_obs_args)
            fo_3["location"] = np.array([1.0, 1.0, 1.0])
            self.fake_obs_learn = [
                State(**default_obs_args),
                State(**fo_1),
                State(**fo_2),
                State(**fo_3),
            ]

            # Create a symmetric synthetic object, where the location of the last
            # feature differs from the base-synthetic object, resulting in
            # ambiguous rotations
            fo_sym = copy.deepcopy(default_obs_args)
            fo_sym_1 = copy.deepcopy(fo_1)
            fo_sym_2 = copy.deepcopy(fo_2)
            fo_sym_3 = copy.deepcopy(fo_3)
            fo_sym_3["location"] = np.array([0.0, 1.0, 0.0])
            self.fake_obs_symmetric = [
                State(**fo_sym),
                State(**fo_sym_1),
                State(**fo_sym_2),
                State(**fo_sym_3),
            ]

            # === Synthetic objects for hypothesis-testing policy ===

            # These can be thought of as a square, and a "house" (square with
            # triangle point above); this helps simulate distinguishing e.g. a mug from
            # a can
            # The square
            fo_square = copy.deepcopy(default_obs_args)
            fo_square_1 = copy.deepcopy(fo_1)
            fo_square_2 = copy.deepcopy(fo_2)
            fo_square_3 = copy.deepcopy(fo_3)
            fo_square_3["location"] = np.array([0.0, 1.0, 0.0])
            self.fake_obs_square = [
                State(**fo_square),
                State(**fo_square_1),
                State(**fo_square_2),
                State(**fo_square_3),
            ]

            # The house; note it has an additional, 5th feature
            fo_house = copy.deepcopy(fo_square)
            fo_house_1 = copy.deepcopy(fo_square_1)
            fo_house_2 = copy.deepcopy(fo_square_2)
            fo_house_3 = copy.deepcopy(fo_square_3)
            fo_house_4 = copy.deepcopy(default_obs_args)
            fo_house_4["location"] = np.array([0.5, 1.5, 0.0])
            self.fake_obs_house = [
                State(**fo_house),
                State(**fo_house_1),
                State(**fo_house_2),
                State(**fo_house_3),
                State(**fo_house_4),
            ]

            # The house, but translated and rotated in the world
            # Define corners of the house
            house_points = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.5, 1.5, 0.0],  # The rooftop
                ]
            )
            # Define the local coordinate frame for each point
            ref_frames = np.array(
                [[[0, 1, 0], [1, 0, 0], [0, 0, -1]] for _ in house_points]
            )

            center_point = [0.5, 0.5, 0.0]
            # 0 the "center" of the house to make rotations easier to compose
            house_points = house_points - center_point

            rotation = Rotation.from_euler("xyz", [45, 75, 190], degrees=True)
            translation_vector = [0.1, 0.2, 0.3]

            house_points = rotation.apply(house_points)
            house_points = house_points + translation_vector

            rotated_ref_frames = [rotation.apply(frame) for frame in ref_frames]

            config_list = []
            for ii in range(5):
                config_list.append(copy.deepcopy(fo_house))
                config_list[ii]["location"] = house_points[ii]
                config_list[ii]["morphological_features"]["pose_vectors"] = (
                    rotated_ref_frames[ii]
                )

            self.fake_obs_house_trans = [State(**obs_dic) for obs_dic in config_list]

            self.placeholder_target = {
                "object": "placeholder",
                "quat_rotation": [1, 0, 0, 0],
            }

        def string_to_array(self, array_string, get_positive_rotations=False):
            """Convert string representation of an array into a numpy array.

            Is needed since the arrays we read out of the stats csv cells are strings.

            Returns:
                np_array: numpy array
            """
            np_array = np.array([])
            for i in array_string.split("[")[1].split("]")[0].split(" "):
                if len(i) > 0:
                    r = int(float(i))
                    if get_positive_rotations and r < 0:
                        r = 360 + r
                    np_array = np.append(np_array, r)
            return np_array

        def check_train_results(self, train_stats, num_lms=1):
            for lm_id in range(num_lms):
                self.assertEqual(
                    train_stats["result"][0 * num_lms + lm_id],
                    train_stats["result"][4 * num_lms + lm_id],
                    "Object capsule3DSolid at same orientation was not recognized.",
                )
                self.assertNotEqual(
                    train_stats["result"][0 * num_lms + lm_id],
                    train_stats["result"][1 * num_lms + lm_id],
                    "First two episodes should recognize different object.",
                )
                self.assertEqual(
                    "correct",
                    train_stats["primary_performance"][4 * num_lms + lm_id],
                    "capsule3DSolid performance is not correct.",
                )
                self.assertEqual(
                    train_stats["result"][1 * num_lms + lm_id],
                    train_stats["result"][5 * num_lms + lm_id],
                    "Object cubeSolid at same orientation was not recognized.",
                )
                self.assertEqual(
                    "correct",
                    train_stats["primary_performance"][5 * num_lms + lm_id],
                    "cubeSolid performance is not correct.",
                )
                # Rotated objects should not be recognized in this test because they
                # were never seen from this side and graphs are learned from scratch so
                # this info is not stored in the graphs.
                # Once we add dealing with incomplete models this test may need to be
                # adapted.
                self.assertNotEqual(
                    train_stats["result"][0 * num_lms + lm_id],
                    train_stats["result"][2 * num_lms + lm_id],
                    "Object capsule3DSolid at new orientation was recognized.",
                )
                self.assertEqual(
                    "no_match",
                    train_stats["primary_performance"][2 * num_lms + lm_id],
                    "capsule3DSolid performance should be no_match in new rotation.",
                )
                self.assertNotEqual(
                    train_stats["result"][1 * num_lms + lm_id],
                    train_stats["result"][3 * num_lms + lm_id],
                    "Object cubeSolid at different orientation was recognized. (Why?)",
                )
                self.assertEqual(
                    "no_match",
                    train_stats["primary_performance"][3 * num_lms + lm_id],
                    "cubeSolid performance should be no_match in unseen orientation.",
                )
                # Giving more slack here now. This is needed because we now use the
                # actual rotation object when merging graphs and not the rounded euler
                # rotation. This makes new points from an already known rotation more
                # affected by noise in the rotation estimate and more likely to be
                # stored in the graph.
                self.assertLessEqual(
                    np.abs(
                        train_stats["detected_scale"][4 * num_lms + lm_id]
                        - train_stats["primary_target_scale"][4 * num_lms + lm_id]
                    ),
                    0.01,
                    "Scale of capsule3DSolid not detected correctly.",
                )
                self.assertLessEqual(
                    np.abs(
                        train_stats["detected_scale"][5 * num_lms + lm_id]
                        - train_stats["primary_target_scale"][5 * num_lms + lm_id]
                    ),
                    0.05,
                    "Scale of cubeSolid not detected correctly.",
                )

                capsule_r = self.string_to_array(
                    train_stats["detected_rotation"][4 * num_lms + lm_id]
                )
                cube_r = self.string_to_array(
                    train_stats["detected_rotation"][5 * num_lms + lm_id]
                )
                self.assertEqual(
                    len(capsule_r), 3, "Should store 3d detected rotation."
                )
                self.assertEqual(len(cube_r), 3, "Should store 3d detected rotation.")
                self.assertLessEqual(
                    train_stats["rotation_error"][4 * num_lms + lm_id],
                    0.001,
                    "Rotation of capsule3DSolid not detected correctly.",
                )
                self.assertLessEqual(
                    train_stats["rotation_error"][5 * num_lms + lm_id],
                    0.001,
                    "Rotation of cubeSolid not detected correctly.",
                )

        def check_graphs_equal(self, g1, g2):
            """Used for checking saving and loading.

            `g1` and `g2` are torch_geometric.data.data.Data objects. Check if all
            fields match.
            """
            print("...check if graphs are equal...")
            # Same xs
            num_x_same = (g1.x == g2.x).sum()
            self.assertEqual(num_x_same, g1.x.size(0) * g1.x.size(1))

            # Same edges means same graph structure
            start_edge_same, end_edge_same = g1.edge_index == g2.edge_index
            n_start_edge_same, n_end_edge_same = (
                start_edge_same.sum(),
                end_edge_same.sum(),
            )
            self.assertEqual(n_start_edge_same, g1.edge_index.size(1))
            self.assertEqual(n_end_edge_same, g1.edge_index.size(1))

            # Same edge data
            num_edge_attrs_same = (g1.edge_attr == g2.edge_attr).sum()
            self.assertEqual(
                num_edge_attrs_same, g1.edge_attr.size(0) * g1.edge_attr.size(1)
            )

            # Same positions in space
            num_pos_same = (g1.pos == g2.pos).sum()
            self.assertEqual(num_pos_same, g1.pos.size(0) * g1.pos.size(1))

            # Same normals
            num_norm_same = (g1.norm == g2.norm).sum()
            self.assertEqual(num_norm_same, g1.norm.size(0) * g1.norm.size(1))

        def check_eval_results(self, eval_stats, num_lms=1):
            for lm_id in range(num_lms):
                self.assertEqual(
                    eval_stats["result"][0 * num_lms + lm_id],
                    eval_stats["result"][2 * num_lms + lm_id],
                    "Object capsule3DSolid at same orientation was not recognized "
                    f"by lm {lm_id}.",
                )
                self.assertNotEqual(
                    eval_stats["result"][0 * num_lms + lm_id],
                    eval_stats["result"][1 * num_lms + lm_id],
                    "Object capsule3DSolid at new orientation should be recognized as "
                    f"separate object by lm {lm_id}.",
                )
                self.assertAlmostEqual(
                    eval_stats["detected_scale"][2 * num_lms + lm_id],
                    eval_stats["primary_target_scale"][2 * num_lms + lm_id],
                    4,
                    "Scale of capsule3DSolid not detected correctly " f"by lm {lm_id}.",
                )
                capsule_r = self.string_to_array(
                    eval_stats["detected_rotation"][2 * num_lms + lm_id]
                )
                self.assertEqual(
                    len(capsule_r), 3, "Should store 3d detected rotation."
                )
                self.assertLessEqual(
                    eval_stats["rotation_error"][2 * num_lms + lm_id],
                    0.001,
                    "Rotation of capsule3DSolid not detected correctly "
                    f"by lm {lm_id}.",
                )
                for i in range(3):
                    self.assertEqual(
                        "correct",
                        eval_stats["primary_performance"][i * num_lms + lm_id],
                        f"capsule3DSolid was not detected by lm {lm_id}.",
                    )

        def check_multilm_train_results(self, train_stats, num_lms, min_done):
            for episode in range(4):
                no_match_count = 0
                for lm_id in range(num_lms):
                    if (
                        train_stats["primary_performance"][episode * num_lms + lm_id]
                        == "no_match"
                    ):
                        no_match_count += 1
                self.assertEqual(
                    no_match_count,
                    num_lms,
                    f"All LMs should detect no_match in episode {episode}",
                )

            for episode in [4, 5]:
                correct_count = 0
                for lm_id in range(num_lms):
                    if (
                        train_stats["primary_performance"][episode * num_lms + lm_id]
                        == "correct"
                    ):
                        correct_count += 1
                self.assertGreaterEqual(
                    correct_count,
                    min_done,
                    f"Not enough correct LMs for train episode {episode}",
                )

        def check_multilm_eval_results(
            self, eval_stats, num_lms, min_done, num_episodes=3
        ):
            for episode in range(num_episodes):
                correct_count = 0
                for lm_id in range(num_lms):
                    if (
                        eval_stats["primary_performance"][episode * num_lms + lm_id]
                        == "correct"
                    ):
                        correct_count += 1
                self.assertGreaterEqual(
                    correct_count,
                    min_done,
                    f"Not enough correct LMs for eval episode {episode}",
                )

        def check_hierarchical_lm_train_results(self, train_stats):
            for episode in range(4):
                self.assertEqual(
                    train_stats["primary_performance"][episode * 2],
                    "no_match",
                    f"LM0 should not match in episode {episode}",
                )
                self.assertEqual(
                    train_stats["primary_performance"][episode * 2 + 1],
                    "no_match",
                    f"LM1 should not match in episode {episode}",
                )

            for episode in [4, 5]:
                self.assertEqual(
                    train_stats["primary_performance"][episode * 2],
                    "correct",
                    f"LM0 should detect the correct object in episode {episode}",
                )
                self.assertIn(
                    train_stats["primary_performance"][episode * 2 + 1],
                    ["correct", "correct_mlh"],
                    f"LM1 should detect the correct object in episode {episode}"
                    "or have it as its most likely hypothesis.",
                )

        def check_hierarchical_lm_eval_results(self, eval_stats):
            for episode in range(3):
                self.assertEqual(
                    eval_stats["primary_performance"][episode * 2],
                    "correct",
                    f"LM0 should detect the correct object in episode {episode}",
                )
                # NOTE: LM1 gets no match (due to incomplete models, especially of LM
                # input channel). Will not test this here since maybe in the future this
                # will be better and it is not a feature of the system.

        def check_hierarchical_models(self, models):
            for model in ["new_object0", "new_object1"]:
                # Check that graph was extended when recognizing object.
                self.assertLess(
                    models["0"]["LM_0"][model]["patch_0"].num_nodes,
                    models["2"]["LM_0"][model]["patch_0"].num_nodes,
                    f"LM0 should have more points in the graph for {model} "
                    "after recognizing it and extending the graph.",
                )
                # Check LM0 has higher detail model of object thank LM1.
                self.assertGreater(
                    models["0"]["LM_0"][model]["patch_0"].num_nodes,
                    models["0"]["LM_1"][model]["patch_1"].num_nodes,
                    f"LM0 should have more points in the graph for {model} than LM1 "
                    "since it is receiving higher frequency input and has a smaller "
                    "voxel size.",
                )
            # Check that max_nodes_per_graph is applied correctly.
            for model in ["new_object0", "new_object1", "new_object2", "new_object3"]:
                num_nodes = models["2"]["LM_0"][model]["patch_0"].num_nodes
                self.assertLessEqual(
                    num_nodes,
                    50,
                    "LM0 should have <= max_nodes_per_graph nodes in"
                    f" its graph for {model} but has {num_nodes}",
                )
            # Check LM1 does not store LM0 input in first epoch yet.
            self.assertNotIn(
                "learning_module_0",
                models["0"]["LM_1"]["new_object0"].keys(),
                "models in LM1 should not store input from LM0 in episode " "0 yet.",
            )
            # Check that LM1 extended its graph to add LM0 as a input channel.
            channel_keys = models["2"]["LM_1"]["new_object0"].keys()
            self.assertIn(
                "learning_module_0",
                channel_keys,
                "models in LM1 should store input from LM0 in episode 2 "
                f"after extending the graph but only store {channel_keys}",
            )

        def check_possible_paths_or_poses(self, stats_1, stats_2, key):
            for paths1, paths2 in zip(stats_1[key], stats_2[key]):
                possible_objects_1 = set(list(paths1.keys()))
                possible_objects_2 = set(list(paths2.keys()))
                self.assertEqual(possible_objects_1, possible_objects_2)
                for obj in possible_objects_1:
                    # I refuse to go deeper
                    self.assertEqual(paths1[obj], paths2[obj])

        def convert_to_numpy_and_check_equal(self, list1, list2):
            if len(list1) > 0 and isinstance(list1[0], (list, np.ndarray)):
                for v1, v2 in zip(list1, list2):
                    self.convert_to_numpy_and_check_equal(v1, v2)

            else:
                v1np = np.array(list1)
                v2np = np.array(list2)
                np_equal = np.isclose(v1np, v2np, atol=0.000001)
                self.assertTrue(np.sum(np_equal) == len(np_equal))

        def compare_lm_stats(self, stats_1, stats_2):
            ignore_keys = [
                "time",
                "relative_time",
                "possible_matches",
                "stepwise_targets_list",
                "possible_paths",
                "possible_poses",
                "ppf",
                "incoming_id_votes",
                "incoming_location_votes",
                "removed_locations",
                "matching_step_when_output_goal_set",
                "goal_state_achieved",
            ]

            key = "possible_matches"
            for matches_1, matches_2 in zip(stats_1[key], stats_2[key]):
                self.assertEqual(set(matches_1), set(matches_2))

            key = "stepwise_targets_list"
            for matches_1, matches_2 in zip(stats_1[key], stats_2[key]):
                self.assertEqual(set(matches_1), set(matches_2))

            key = "possible_paths"
            self.check_possible_paths_or_poses(stats_1, stats_2, key)

            key = "possible_poses"
            self.check_possible_paths_or_poses(stats_1, stats_2, key)

            remaining_keys = [key for key in stats_1.keys() if key not in ignore_keys]
            for key in remaining_keys:
                val_old = stats_1[key]
                val_new = stats_2[key]
                if isinstance(val_old, str):
                    self.assertEqual(val_old, val_new)

                elif isinstance(val_old, (list, np.ndarray)):
                    self.convert_to_numpy_and_check_equal(val_old, val_new)

        def compare_sensor_module_logs(self, log_1, log_2):
            # Check that sensor modules are the same with just a few simple for loops!
            for key in ["SM_0", "SM_1"]:
                sm_data_old = log_1["0"][key]
                sm_data_new = log_2["0"][key]
                for key2 in sm_data_old.keys():
                    data_old = sm_data_old[key2]
                    data_new = sm_data_new[key2]
                    for i in range(len(data_old)):
                        step_old = data_old[i]
                        step_new = data_new[i]
                        for key3 in step_old.keys():
                            if key3 in [
                                "morphological_features",
                                "non_morphological_features",
                                "displacement",
                            ]:
                                feat_old = step_old[key3]
                                feat_new = step_new[key3]
                                for feature in feat_old.keys():
                                    self.assertEqual(
                                        feat_old[feature], feat_new[feature]
                                    )
                            elif key3 == "allowable_sender_types":
                                for f_idx in range(len(step_old[key3])):
                                    self.assertEqual(
                                        step_old[key3][f_idx], step_new[key3][f_idx]
                                    )
                            else:
                                if type(step_old[key3]) == str:
                                    # sm_id can not be compared as array
                                    self.assertEqual(step_old[key3], step_new[key3])
                                else:
                                    np_old = np.array(step_old[key3])
                                    np_new = np.array(step_new[key3])
                                    np_equal = np.isclose(np_old, np_new, atol=0.00001)
                                    self.assertEqual(np.sum(np_equal), np_equal.size)
