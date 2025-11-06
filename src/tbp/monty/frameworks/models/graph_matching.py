# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import logging
import os
from typing import ClassVar

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.environments.embodied_environment import SemanticID
from tbp.monty.frameworks.loggers.exp_logger import BaseMontyLogger
from tbp.monty.frameworks.loggers.graph_matching_loggers import (
    BasicGraphMatchingLogger,
    DetailedGraphMatchingLogger,
    SelectiveEvidenceLogger,
)
from tbp.monty.frameworks.models.abstract_monty_classes import LearningModule, LMMemory
from tbp.monty.frameworks.models.buffer import FeatureAtLocationBuffer
from tbp.monty.frameworks.models.goal_state_generation import GraphGoalStateGenerator
from tbp.monty.frameworks.models.monty_base import MontyBase
from tbp.monty.frameworks.models.object_model import GraphObjectModel
from tbp.monty.frameworks.models.states import GoalState

logger = logging.getLogger(__name__)


class MontyForGraphMatching(MontyBase):
    """General Monty model for recognizing object using graphs."""

    LOGGING_REGISTRY: ClassVar[dict[str, type[BaseMontyLogger]]] = {
        # Don't do any formal logging, just save models. Used for pretraining.
        "SILENT": BaseMontyLogger,
        # Log things like basic stats.csv files, data to reproduce experiments
        "BASIC": BasicGraphMatchingLogger,
        # Utter deforestation
        "DETAILED": DetailedGraphMatchingLogger,
        # Save specific stats necessary for object similarity analysis.
        "SELECTIVE": SelectiveEvidenceLogger,
    }

    def __init__(self, *args, **kwargs):
        """Initialize and reset LM."""
        super().__init__(*args, **kwargs)

    # =============== Public Interface Functions ===============
    # ------------------- Main Algorithm -----------------------
    def pre_episode(self, primary_target, semantic_id_to_label=None):
        """Reset values and call sub-pre_episode functions."""
        self._is_done = False
        self.reset_episode_steps()
        self.switch_to_matching_step()
        self.reset()
        self.primary_target = primary_target
        self.semantic_id_to_label = semantic_id_to_label

        for lm in self.learning_modules:
            lm.pre_episode(primary_target)

        for sm in self.sensor_modules:
            sm.pre_episode()

        logger.debug(
            f"Models in memory: {self.learning_modules[0].get_all_known_object_ids()}"
        )

    def send_vote_to_lm(self, lm, lm_id, combined_votes):
        """Route correct votes to a given LM."""
        logger.debug(f"Matches before voting (LM {lm_id}): {lm.get_possible_matches()}")
        if len(combined_votes) < 1:
            # Deal with set vote from displacement LM
            lm.receive_votes(combined_votes)
        else:
            lm.receive_votes(combined_votes[lm_id])

        logger.debug(f"Matches after voting (LM {lm_id}): {lm.get_possible_matches()}")

    def update_stats_after_vote(self, lm):
        """Add voting stats to buffer and check individual terminal condition."""
        stats = lm.collect_stats_to_save()
        lm.buffer.update_last_stats_entry(stats)
        num_matches = len(lm.get_possible_matches())
        if num_matches == 0:
            lm.set_individual_ts(terminal_state="no_match")
        elif num_matches > 0 and lm.terminal_state == "no_match":
            # Allow LM to recover from no_match state if votes from other LMs have
            # made it have possible matches now.
            lm.set_individual_ts(terminal_state=None)

    def check_if_any_lms_updated(self):
        """True if any LM received sensory information on the current episode step.

        Returns:
            True if any LM received sensory information on the current episode step,
            False otherwise.
        """
        for lm_ii in self.learning_modules:
            if lm_ii.buffer.get_last_obs_processed():
                # True if the last step was an lm-processed-step
                return True

        # Otherwise return False
        return False

    def deal_with_time_out(self):
        """Set LM terminal states to time_out."""
        self._set_time_outs(global_time_out=True)

    def check_terminal_conditions(self):
        """Check if all LMs have reached a terminal state.

        This could be no_match, match, or time_out. If all LMs have reached one of these
        states, end the episode.

        Currently the episode just ends if
            - min_lms_match lms have reached "match"
            - all lms have reached "no_match"
            - We have exceeded max_total_steps

        Note:
            In the future we may want to allow ending an episode when all states are
            either match or no_match. Right now, the match lms will have to convince the
            no_match lms of their detected object and pose for the episode to end which
            may be more difficult if not all LMs know about all objects.

        Returns:
            True if all LMs have reached a terminal state, False otherwise.
        """
        # First check if all LMs have no match (for example in the first episode when
        # we have no objects in memory yet). If that is the case there is no need to
        # take max_steps or to spend time updating the lms terminal conditions. We could
        # also reset the hypotheses of the no_match LMs to give them another chance to
        # recover.
        all_lms_no_match = True
        for lm in self.learning_modules:
            if lm.terminal_state != "no_match":
                all_lms_no_match = False

        if all_lms_no_match:
            # Take more exploratory steps if we are building a new graph
            self.num_exploratory_steps = self.num_exploratory_steps * 10
            # No need to check any other conditions if all LMs have no_match
            return True

        # No need to check pose consensus if we haven't moved min steps yet.
        if not self.exceeded_min_steps:
            return False

        # Check if >= min_lms_match LMs have reached match
        # TODO: we may also want to count no_match as done.
        num_lms_done = 0
        for lm in self.learning_modules:
            lm.update_terminal_condition()
            logger.debug(
                f"{lm.learning_module_id} has terminal state: {lm.terminal_state}"
            )
            # If any LM is not done yet, we are not done yet
            if lm.terminal_state == "match":
                num_lms_done += 1

        if num_lms_done >= self.min_lms_match:
            logger.info("\n\nMONTY DETECTED MATCH\n\n")
            return True

    def reset(self):
        """Reset monty status."""
        pass

    # ------------------ Getters & Setters ---------------------

    def set_is_done(self):
        """Set the model is_done flag.

        Method that e.g. experiment class can use to set model is_done flag if
        e.g. total number of episode steps possible has been exceeded
        """
        self._is_done = True

    # ------------------ Logging & Saving ----------------------
    def load_state_dict_from_parallel(self, parallel_dirs, save=False):
        lm_dict = {}
        for pdir in parallel_dirs:
            state_dict = torch.load(os.path.join(pdir, "model.pt"))
            for lm in state_dict["lm_dict"].keys():
                if lm not in lm_dict:
                    lm_dict[lm] = dict(
                        graph_memory={},
                        target_to_graph_id={},
                        graph_id_to_target={},
                    )

                lm_dict[lm]["graph_memory"].update(
                    state_dict["lm_dict"][lm]["graph_memory"]
                )

                # TODO: this is presumably going to be wrong, but we're not really using
                # this attribute right now.
                lm_dict[lm]["target_to_graph_id"].update(
                    state_dict["lm_dict"][lm]["target_to_graph_id"]
                )
                lm_dict[lm]["graph_id_to_target"].update(
                    state_dict["lm_dict"][lm]["graph_id_to_target"]
                )

                # TODO: handle target to graph id stuff here, but ignoring for now

        # Everything but lm dict for saving new model
        new_state_dict = {k: v for k, v in state_dict.items() if k != "lm_dict"}
        new_state_dict["lm_dict"] = lm_dict
        load_dir = os.path.dirname(parallel_dirs[0])

        if save:
            torch.save(new_state_dict, os.path.join(load_dir, "model.pt"))

        self.load_state_dict(new_state_dict)

    # ======================= Private ==========================
    # ------------------- Main Algorithm -----------------------

    def _step_learning_modules(self):
        """Collect inputs and step each learning module."""
        for i in range(len(self.learning_modules)):
            sensory_inputs = self._collect_inputs_to_lm(i)
            # If LM has any inputs, take a step
            if sensory_inputs is not None:
                self._set_stepwise_targets(self.learning_modules[i], sensory_inputs)

                if self.step_type == "matching_step":
                    input_channels = [obs.sender_id for obs in sensory_inputs]
                    logger.info(
                        f"Sending input from {input_channels}"
                        f" to {self.learning_modules[i].learning_module_id}"
                    )
                lm_step_method = getattr(self.learning_modules[i], self.step_type)
                assert callable(lm_step_method), f"{lm_step_method} must be callable"
                lm_step_method(sensory_inputs)
                if self.step_type == "matching_step":
                    logger.debug(f"Stepping learning module {i}")
                self.learning_modules[i].add_lm_processing_to_buffer_stats(
                    lm_processed=True
                )
            else:
                if self.step_type == "matching_step":
                    logger.info(f"Skipping step on learning module {i}")
                self.learning_modules[i].add_lm_processing_to_buffer_stats(
                    lm_processed=False
                )

                """
                Target-object remains unchanged if we're not passing information
                to the LM
                NOTE we may want to change this if we eventually want it to e.g.
                classify when it's off an object (i.e. "off-object" is the target); this
                could be done at the same time as we add better handling of off-object
                observations
                TODO make use of a buffer method to handle the below logging
                """
                self.learning_modules[i].stepwise_targets_list.append(
                    self.learning_modules[i].stepwise_target_object
                )

    def _combine_votes(self, votes_per_lm):
        """Combine outgoing votes using lm_to_lm_vote_matrix matrix.

        TODO: make custom Monty classes for feature, disp, and evidence modeling
        and separate out the vote functions.

        Args:
            votes_per_lm: outgoing votes from each LM

        Returns:
            Input votes for each LM.
        """
        combined_votes = []
        for i in range(len(self.learning_modules)):
            if isinstance(votes_per_lm[0], set):
                # Negative set voting for compatibility with displacement LM
                # TODO: make this cleaner.
                vote = None
                for j in self.lm_to_lm_vote_matrix[i]:
                    if vote is None:
                        vote = set(votes_per_lm[j])
                    else:
                        vote = vote.union(set(votes_per_lm[j]))
            else:
                neg_object_id_votes = {}
                pos_object_id_votes = {}
                lm_object_location_votes = {}
                lm_object_rotation_votes = {}
                receiving_lm_pose = votes_per_lm[i]["sensed_pose_rel_body"]
                for j in self.lm_to_lm_vote_matrix[i]:
                    lm_object_id_vote = votes_per_lm[j]["object_id_vote"]
                    for obj in lm_object_id_vote.keys():
                        if obj in pos_object_id_votes.keys():
                            pos_object_id_votes[obj] += int(lm_object_id_vote[obj])
                            neg_object_id_votes[obj] += int(not lm_object_id_vote[obj])
                        else:
                            pos_object_id_votes[obj] = int(lm_object_id_vote[obj])
                            neg_object_id_votes[obj] = int(not lm_object_id_vote[obj])
                        # Assume models of object have been learned in same
                        # reference frame. Otherwise, during learning we need to
                        # store a fixed transform between the two reference
                        # frames and apply it here every time.

                        # Get the displacement between the sending and receiving
                        # sensor and take this into account when transmitting
                        # possible locations on the object.
                        # "If I am here, you should be there."
                        lm_loc_vote = votes_per_lm[j]["location_vote"][obj]
                        lm_rot_vote = votes_per_lm[j]["rotation_vote"][obj]
                        logger.debug(
                            f"loc vote from LM {j} - {obj}: {lm_loc_vote.shape}"
                        )
                        logger.debug(
                            f"rot vote from LM {j} - {obj}: {len(lm_rot_vote)}"
                        )
                        sending_lm_pose = votes_per_lm[j]["sensed_pose_rel_body"]
                        sensor_disp = np.array(receiving_lm_pose[0]) - np.array(
                            sending_lm_pose[0]
                        )
                        sensor_rotation_disp, _ = Rotation.align_vectors(
                            sending_lm_pose[1:], receiving_lm_pose[1:]
                        )
                        logger.debug(
                            f"LM {i} to {j} - displacement: {sensor_disp}, "
                            f"rotation: "
                            f"{sensor_rotation_disp.as_euler('xyz', degrees=True)}"
                        )

                        # NOTE: ideally we also want negative votes here. Otherwise
                        # models with lots of points have a higher weight in the vote.
                        # Also, incomplete models and low-resolution models will cause
                        # problems here.
                        # Could also somehow normalize or weight votes.

                        lm_loc_vote_transformed = []
                        lm_rot_vote_transformed = []
                        # Take the location votes and transform them so they would
                        # apply to the receiving LMs sensor. Basically saying, if my
                        # sensor is here and in this pose then your sensor should be
                        # there (search_pos) in that pose (search_rot).
                        # NOTE: rotation votes are not being used right now.
                        for loc_id, location in enumerate(lm_loc_vote):
                            for pose in lm_rot_vote[loc_id]:
                                search_pos = location + pose.apply(sensor_disp.copy())
                                search_rot = pose * sensor_rotation_disp
                                lm_loc_vote_transformed.append(search_pos)
                                lm_rot_vote_transformed.append(search_rot)

                        if len(lm_loc_vote_transformed) > 0:
                            if obj in lm_object_location_votes.keys():
                                lm_object_location_votes[obj] = np.vstack(
                                    [
                                        lm_object_location_votes[obj],
                                        np.array(lm_loc_vote_transformed),
                                    ]
                                )
                                lm_object_rotation_votes[obj].append(
                                    lm_rot_vote_transformed
                                )
                            else:
                                lm_object_location_votes[obj] = np.array(
                                    lm_loc_vote_transformed
                                )
                                lm_object_rotation_votes[obj] = lm_rot_vote_transformed
                logger.info(
                    f"VOTE from LMs {self.lm_to_lm_vote_matrix[i]} to LM {i}: + "
                    f"{pos_object_id_votes}, - {neg_object_id_votes}"
                )
                vote = {
                    "pos_object_id_votes": pos_object_id_votes,
                    "neg_object_id_votes": neg_object_id_votes,
                    "pos_location_votes": lm_object_location_votes,
                    "pos_rotation_votes": lm_object_rotation_votes,
                }
                combined_votes.append(vote)
        return combined_votes

    def _vote(self):
        """Use lm_to_lm_vote_matrix to transmit votes between lms."""
        if self.lm_to_lm_vote_matrix is not None:
            # Send out votes
            votes_per_lm = []
            for i in range(len(self.learning_modules)):
                votes_per_lm.append(self.learning_modules[i].send_out_vote())

            combined_votes = self._combine_votes(votes_per_lm)
            # Receive votes
            for i in range(len(self.learning_modules)):
                logger.debug(f"------ Sending votes to LM {i} -------")
                self.send_vote_to_lm(self.learning_modules[i], i, combined_votes)
                self.update_stats_after_vote(self.learning_modules[i])

        # Log possible matches
        for lm in self.learning_modules:
            pm = (
                lm.get_possible_matches()
                if lm.buffer.get_num_observations_on_object()
                else []
            )
            logger.info(f"Possible matches for {lm.learning_module_id}: {pm}")

    def _pass_infos_to_motor_system(self):
        """Pass input observations to the motor system.

        Omit goal states in this case.
        """
        # TODO M: generalize to multiple sensor modules

        if (
            self.step_type == "matching_step"
            or self.sensor_module_outputs[0] is not None
        ):
            self._pass_input_obs_to_motor_system(self.sensor_module_outputs[0])

    def _set_step_type_and_check_if_done(self):
        """Check terminal conditions and decide if we change the step type."""
        self.update_step_counters()

        if self.step_type == "matching_step":
            # Check that at least one LM has processed information, such that we should
            # run check_terminal_conditions(); note in particular that
            # check_terminal_conditions will e.g. increment symmetry evidence, so we
            # should only run it if there was new information received
            if self.check_if_any_lms_updated():
                # Decide if we switch to exploratory step
                enough_lms_done = self.check_terminal_conditions()

                if enough_lms_done:
                    # set terminal state of lms that are not done yet to time_out or
                    # pose_time out. Other terminal states remain the same.
                    self._set_time_outs(global_time_out=False)

                    if self.experiment_mode == "train":
                        self.switch_to_exploratory_step()
                        for sm in self.sensor_modules:
                            sm.is_exploring = True

                    elif self.experiment_mode == "eval":
                        if self.matching_steps > self.min_eval_steps:
                            self._is_done = True

            else:
                self.matching_steps -= 1

        elif self.step_type == "exploratory_step":
            if self.check_if_any_lms_updated():
                if self.exploratory_steps >= self.num_exploratory_steps:
                    self._is_done = True

            else:
                # If information was not passed to the LMs, then don't count as a true
                # exploratory step
                self.exploratory_steps -= 1

                # Note that as for matching steps in MontyObjectRecognitionExperiment,
                # Monty experiment classes handle the case where
                # exploratory_steps is never being incremented (e.g. because we're in
                # a void without any objects), ensuring that we eventually time-out
                # according to max_total_steps

    def _pass_input_obs_to_motor_system(self, infos):
        """Pass processed observations to motor system.

        Give the motor system all information it needs for its policy to decide the
        next action. Here it needs the processed observation from the sensor patch.

        For some motor systems (e.g. curvature-informed surface-agent policy), also
        provides locations associated with tangential movements; this can help ensure we
        e.g. avoid revisiting old locations.
        """
        self.motor_system._policy.processed_observations = infos

        # TODO M clean up the below when refactoring the surface-agent policy
        if hasattr(self.motor_system._policy, "tangent_locs"):
            last_action = self.motor_system._policy.last_action

            if last_action is not None:
                if last_action.name == "orient_vertical":
                    # Only append locations associated with performing a tangential
                    # action, rather than some form of corrective movement; these
                    # movements are performed immediately after "orient_vertical"
                    # TODO generalize to multiple sensor modules
                    self.motor_system._policy.tangent_locs.append(
                        self.sensor_modules[0].visited_locs[-1]
                    )
                    self.motor_system._policy.tangent_norms.append(
                        self.sensor_modules[0].visited_normals[-1]
                    )

    # ------------------------ Helper --------------------------
    def _set_stepwise_targets(self, lm, sensory_inputs):
        """Set the "stepwise" target for each learning module.

        Based on the current sensory input, set the 'stepwise' target for each
        learning module, i.e. the class label of the object it is actually receiving
        sensory input from

        TODO seperate this out with the new Observation class; also the LM should
        have its own method to update this attribute, rather than the Monty class
        changing this
        TODO: Add unit tests for this
        """
        try:
            lm.stepwise_target_object = self.semantic_id_to_label[
                SemanticID(sensory_inputs[0]._semantic_id)
            ]
            logger.debug(f"Stepwise target: {lm.stepwise_target_object}")
        except KeyError:
            # Semantic sensor may not be available, or the "patch" key
            # may be different
            logger.debug("Semantic ID not available for stepwise-targets")
            lm.stepwise_target_object = "no_label"
        except TypeError:
            # semantic_id_to_label is not specified, e.g. in unit tests
            logger.debug("semantic_id_to_label mapping not specified")
            lm.stepwise_target_object = "no_label"
        except AttributeError:
            logger.debug("semantic_id_to_label mapping not specified")
            lm.stepwise_target_object = "no_label"

        # Add logging information : TODO use the buffer to log this appropriately
        lm.stepwise_targets_list.append(lm.stepwise_target_object)

    def _set_time_outs(self, global_time_out=False):
        """Set terminal state of LMs that are not done yet to time_out.

        Args:
            global_time_out: If True, set Monty state to done so we don't go into
                exploration mode anymore (if we timed out we didn't recognize an object
                so exploration makes no sense since we won't add anything to memory).
                This is set to False, if Monty didn't reach a global time out (exceeded
                max_steps) but instead, min_lms_match LMs have recognized an object.
                Then the other LMs will be set to time_out, but we still want to
                explore.
        """
        # Don't set LM states to time out if we were in exploratory mode
        if self.step_type != "exploratory_step":
            for lm in self.learning_modules:
                if lm.terminal_state is None:
                    lm.terminal_state = "time_out"
        if global_time_out:
            # Don't go into exploratory mode if we timed out
            self._is_done = True


class GraphLM(LearningModule):
    """General Learning Module that contains a graph memory."""

    def __init__(self, initialize_base_modules=True):
        """Initialize general Learning Module based on graphs.

        Args:
            initialize_base_modules: Provides option to not intialize
                the base modules if more specialized versions will be initialized in
                child LMs. Defaults to True.
        """
        super().__init__()
        self.buffer = FeatureAtLocationBuffer()
        self.buffer.reset()
        self.learning_module_id = "LM_0"

        if initialize_base_modules:
            self.graph_memory = GraphMemory(k=None, graph_delta_thresholds=None)
            self.gsg = GraphGoalStateGenerator(self)
            self.gsg.reset()

        self.mode = None  # initialize to neither training nor testing
        # Dictionaries to tell which objects were involved in building a graph
        # and which graphs correspond to each target object
        self.target_to_graph_id = {}
        self.graph_id_to_target = {}
        self.primary_target = None
        self.detected_object = None
        self.detected_pose = [None for _ in range(7)]
        # Will always be set during experiment setup, just setting here for unit tests
        self.has_detailed_logger = False
        self.symmetry_evidence = 0

    # =============== Public Interface Functions ===============

    # ------------------- Main Algorithm -----------------------
    def reset(self):
        """NOTE: currently not used in public interface."""
        (
            self.possible_paths,
            self.possible_poses,
        ) = self.graph_memory.get_initial_hypotheses()

    def pre_episode(self, primary_target):
        """Set target object var and reset others from last episode.

        primary_target : the primary target for the learning module/
            Monty system to recognize (e.g. the object the agent begins on, or an
            important object in the environment; NB that a learning module can also
            correctly classify a "stepwise_target", corresponding to the object that
            it is currently on, while it is attempting to classify the primary_target)
        """
        self.reset()
        self.buffer.reset()
        if self.gsg is not None:
            self.gsg.reset()
        self.primary_target = primary_target["object"]
        self.primary_target_rotation_quat = primary_target["quat_rotation"]
        self.stepwise_target_object = None
        self.stepwise_targets_list = []
        self.terminal_state = None
        self.detected_object = None
        self.detected_pose = [None for _ in range(7)]
        self.detected_rotation_r = None

    def matching_step(self, observations):
        """Update the possible matches given an observation."""
        first_movement_detected = self._agent_moved_since_reset()
        buffer_data = self._add_displacements(observations)
        self.buffer.append(buffer_data)
        self.buffer.append_input_states(observations)

        if first_movement_detected:
            logger.debug("performing matching step.")
        else:
            logger.debug("we have not moved yet.")

        self._compute_possible_matches(
            observations, first_movement_detected=first_movement_detected
        )

        if len(self.get_possible_matches()) == 0:
            self.set_individual_ts(terminal_state="no_match")

        if self.gsg is not None:
            self.gsg.step(observations)

        stats = self.collect_stats_to_save()
        self.buffer.update_stats(stats, append=self.has_detailed_logger)

    def exploratory_step(self, observations):
        """Step without trying to recognize object (updating possible matches)."""
        buffer_data = self._add_displacements(observations)
        self.buffer.append(buffer_data)
        self.buffer.append_input_states(observations)

    def post_episode(self):
        """If training, update memory after each episode."""
        if (self.mode == "train") and len(self.buffer) > 0:
            logger.info(f"\n---Updating memory of {self.learning_module_id}---")
            self._update_memory()
            self._update_target_graph_mapping(self.detected_object, self.primary_target)

    def send_out_vote(self):
        """Send out list ob objects that are not possible matches.

        By sending out the negavtive matches we avoid the problem that
        every LM needs to know about the same objects. We could think of
        this as more of an inhibitory signal (I know it can't be this
        object so you all don't need to check that anymore).

        Returns:
            Set of objects that are not possible matches.
        """
        possible_matches = set(self.get_possible_matches())
        all_objects = set(self.get_all_known_object_ids())
        vote = all_objects.difference(possible_matches)
        logger.debug(
            f"PM: {possible_matches} out of all: {all_objects} -> vote: {vote}"
        )
        return vote

    def receive_votes(self, vote_data):
        """Remove object ids that come in from the votes.

        Args:
            vote_data: set of objects that other LMs excluded from possible matches
        """
        if (vote_data is not None) and (
            self.buffer.get_num_observations_on_object() > 0
        ):
            current_possible_matches = self.get_possible_matches()
            for vote in vote_data:
                if vote in current_possible_matches:
                    logger.debug(f"REMOVING {vote} FROM MATCHES")
                    self.possible_matches.pop(vote)
            self._add_votes_to_buffer_stats(vote_data)

    def get_output(self):
        """Return the output of the learning module.

        Is currently only implemented for the evidence LM since the other LM versions
        do not have a notion of MLH and therefore can't produce an output until the last
        step of the episode.
        """
        pass

    def propose_goal_states(self) -> list[GoalState]:
        """Return the goal-states proposed by this LM's GSG.

        Only returned if the LM/GSG was stepped, otherwise returns empty list.
        """
        if self.buffer.get_last_obs_processed() and self.gsg is not None:
            return self.gsg.output_goal_states()

        return []

    def update_terminal_condition(self):
        """Check if we have reached a terminal condition for this episode.

        Returns:
            Terminal state of the LM.
        """
        possible_matches = self.get_possible_matches()
        # no possible matches
        if len(possible_matches) == 0:
            self.last_possible_hypotheses = None
            self.set_individual_ts("no_match")
            if (
                self.buffer.get_num_observations_on_object() > 0
            ):  # lm has gotten input during episode
                self.buffer.stats["detected_location_rel_body"] = (
                    self.buffer.get_current_location(input_channel="first")
                )
        # 1 possible match
        elif (
            (
                self.buffer.get_num_observations_on_object() > 0
            )  # had observations on object
            and len(possible_matches) == 1  # We have it narrowed down to 1 object
        ):
            object_id = possible_matches[0]
            pose = self.get_unique_pose_if_available(object_id)
            if pose is None:  # No pose determined yet
                logger.info(f"Pose for {self.learning_module_id} not narrowed down yet")
            else:
                self.set_individual_ts("match")
                logger.info(f"{self.learning_module_id} recognized object {object_id}")
        # > 1 possible match
        else:
            self.last_possible_hypotheses = None
            logger.info(f"{self.learning_module_id} did not recognize an object yet.")
        return self.terminal_state

    # ------------------ Getters & Setters ---------------------

    def set_experiment_mode(self, mode):
        """Set LM and GM mode to train or eval."""
        assert mode in [
            "train",
            "eval",
        ], "mode must be either `train` or `eval`"
        self.mode = mode

    def set_detected_object(self, terminal_state):
        """Set the current graph ID.

        If we didn't recognize the object this will be new_object{n} where n is
        len(graph_memory) + 1. Otherwise it is the id of the graph that we recognized.
        If we timed out it is None and we will not update the graph memory.
        """
        self.terminal_state = terminal_state
        if terminal_state is None:  # at beginning of episode
            graph_id = None
        elif (terminal_state == "no_match") or len(self.get_possible_matches()) == 0:
            graph_id = "new_object" + str(len(self.graph_memory))
        elif terminal_state == "match":
            graph_id = self.get_possible_matches()[0]
        else:
            graph_id = None
        self.detected_object = graph_id

    def get_possible_matches(self):
        """Get list of current possible objects.

        TODO: Maybe make this private -> check terminal condition

        Returns:
            List of current possible objects.
        """
        return list(self.possible_matches.keys())

    def get_possible_paths(self):
        """Return possible paths for each object.

        This is used for logging/plotting
        and to check if we know where on the object we are.

        Returns:
            Possible paths for each object.
        """
        return self.possible_paths.copy()

    def get_possible_locations(self):
        possible_paths = self.get_possible_paths()
        possible_locations = {}
        for obj in possible_paths.keys():
            possible_paths_obj = np.array(possible_paths[obj])
            if len(possible_paths_obj.shape) > 1:
                possible_locations[obj] = possible_paths_obj[:, -1]
            elif possible_paths_obj.shape[0] > 0:
                # deals with case where first observation is not on object
                possible_locations[obj] = np.array(
                    self.graph_memory.get_locations_in_graph(obj, input_channel="first")
                )
            else:
                possible_locations[obj] = np.array([])
        return possible_locations

    def get_possible_poses(self, as_euler=True):
        """Return possible poses for each object (for logging).

        Possible poses are narrowed down
        in the feature matching version. When using displacements or PPF this is
        empty.

        Returns:
            Possible poses for each object.
        """
        poses = self.possible_poses.copy()
        if as_euler:
            all_poses = {}
            for obj in poses.keys():
                euler_poses = []
                for path in poses[obj]:
                    path_poses = []
                    for pose in path:
                        euler_pose = np.round(
                            pose.inv().as_euler("xyz", degrees=True), 5
                        )
                        path_poses.append(euler_pose)
                    euler_poses.append(path_poses)
                all_poses[obj] = euler_poses
        else:
            all_poses = poses
        return all_poses

    def get_object_scale(self, _object_id):
        """Get object scale. TODO: implement solution for detecting scale.

        Returns:
            1
        """
        return 1

    def get_all_known_object_ids(self):
        """Get the IDs of all object models stored in memory.

        Returns:
            IDs of all object models stored in memory.
        """
        return self.graph_memory.get_memory_ids()

    def get_graph(self, model_id, input_channel=None):
        """Get learned graph from graph memory.

        Note:
            May generalize this in the future to get_object_model which doesn't
            have to be a graph but currently a lot of code expects a graph to be
            returned so this name is more meaningful.

        Returns:
            Graph.
        """
        return self.graph_memory.get_graph(model_id, input_channel)

    def get_input_channels_in_graph(self, model_id):
        """Get input channels stored for a graph in graph memory.

        Returns:
            Input channels stored for a graph in graph memory.
        """
        return self.graph_memory.get_input_channels_in_graph(model_id)

    def get_unique_pose_if_available(self, object_id):
        """Return a 7d pose array if pose is uniquely identified.

        This method should return a 7d pose array containing the detected
        object location, rotation and scale if the pose is uniquely identified.
        If not, it should contain None. This is used in the Monty class to
        determine whether we have reached a terminal state.

        Returns:
            7d pose array or None.
        """
        raise NotImplementedError("This should be implemented in any subclass.")

    # ------------------ Logging & Saving ----------------------

    def set_individual_ts(self, terminal_state):
        logger.info(
            f"Setting terminal state of {self.learning_module_id} to {terminal_state}"
        )
        self.set_detected_object(terminal_state)
        if terminal_state == "match":
            logger.info(
                f"{self.learning_module_id}: "
                f"Detected {self.detected_object} "
                f"at location {np.round(self.detected_pose[:3], 3)},"
                f" rotation {np.round(self.detected_pose[3:6], 3)},"
                f" and scale {self.detected_pose[6]}"
            )
            self.buffer.set_individual_ts(self.detected_object, self.detected_pose)
        else:
            self.buffer.set_individual_ts(None, None)

    def collect_stats_to_save(self):
        """Get all stats that this LM should store in the buffer for logging.

        Returns:
            Stats to store in the buffer.
        """
        stats = {
            "possible_matches": self.get_possible_matches(),
        }
        if self.has_detailed_logger:
            stats = self._add_detailed_stats(stats)
        return stats

    def add_lm_processing_to_buffer_stats(self, lm_processed):
        """Update the buffer stats with whether the LM processed an observation.

        Add boolean of whether the LM processed an observation on this particular
        episode step.

        Args:
            lm_processed: Boolean of whether the LM processed an observation on
                this particular episode step
        """
        self.buffer.update_stats(
            dict(lm_processed_steps=lm_processed), update_time=False
        )

    def state_dict(self):
        """Get the full state dict for logging and saving.

        Returns:
            Full state dict for logging and saving.
        """
        return dict(
            graph_memory=self.graph_memory.state_dict(),
            target_to_graph_id=self.target_to_graph_id,
            graph_id_to_target=self.graph_id_to_target,
        )

    def load_state_dict(self, state_dict):
        """Load state dict.

        Args:
            state_dict: State dict to load.
        """
        self.graph_memory.load_state_dict(state_dict["graph_memory"])
        self.target_to_graph_id = state_dict["target_to_graph_id"]
        self.graph_id_to_target = state_dict["graph_id_to_target"]

    # ======================= Private ==========================

    # ------------------- Main Algorithm -----------------------
    def _compute_possible_matches(self, observations, first_movement_detected=True):
        """Use graph memory to get the current possible matches.

        Args:
            observations: Observations to use for computing possible matches.
            first_movement_detected: Whether the agent has moved since the buffer reset
                signal.
        """
        if first_movement_detected:
            query = [
                self._select_features_to_use(observations),
                self.buffer.get_current_displacement(input_channel="all"),
            ]
        else:
            query = [
                self._select_features_to_use(observations),
                None,
            ]

        logger.debug(f"query: {query}")

        self._update_possible_matches(query=query)

    def _update_possible_matches(self):
        # QUESTION: Should we give this a more general name? Like update_hypotheses
        # or update_state?
        # QUESTION: Should this actually be something handled in LMs?
        raise NotImplementedError("Need to implement way to update memory hypotheses")

    def _update_memory(self):
        """Give all infos to graph_memory.update_memory to determine how to update."""
        args = self.buffer.get_infos_for_graph_update()
        args["graph_id"] = self.detected_object
        args["object_rotation"] = self.detected_rotation_r
        if args["object_rotation"] is not None:
            # TODO: find a solution that makes it more obvious when rotation is rel
            # the model or rel environment.
            args["object_rotation"] = args["object_rotation"].inv()
        self.graph_memory.update_memory(**args)

    def _update_target_graph_mapping(self, detected_object, target_object):
        """Update dicts that keep track which graphs were built from which objects."""
        if detected_object is not None:
            if detected_object not in self.graph_id_to_target.keys():
                self.graph_id_to_target[detected_object] = {target_object}
            else:
                self.graph_id_to_target[detected_object].add(target_object)

            if target_object not in self.target_to_graph_id.keys():
                self.target_to_graph_id[target_object] = {detected_object}
            else:
                self.target_to_graph_id[target_object].add(detected_object)

    def _agent_moved_since_reset(self):
        return len(self.buffer) > 0

    # ------------------------ Helper --------------------------

    def _add_displacements(self, obs):
        """Add displacements to the current observation.

        The observation consists of features at a location. To get the displacement we
        have to look at the previous observation stored in the buffer.

        Args:
            obs: Observations to add displacements to.

        Returns:
            Observations with displacements.
        """
        for o in obs:
            if self.buffer.get_buffer_len_by_channel(o.sender_id) > 0:
                displacement = o.location - self.buffer.get_current_location(
                    input_channel=o.sender_id
                )
            else:
                displacement = np.zeros(3)
            o.set_displacement(displacement)
        return obs

    def _select_features_to_use(self, states):
        """Extract the features from observations that are specified in tolerances.

        TODO: requires self.tolerances
        TODO S: if keeping the dict format, move this function to State class

        Returns:
            Features to use.
        """
        features_to_use = {}
        for state in states:
            input_channel = state.sender_id
            features_to_use[input_channel] = {}
            for feature in state.morphological_features.keys():
                # in evidence matching pose_vectors are always added to tolerances
                # since they are requires for matching.
                if (
                    feature in self.tolerances[input_channel].keys()
                    or feature == "pose_fully_defined"
                ):
                    features_to_use[input_channel][feature] = (
                        state.morphological_features[feature]
                    )
            for feature in state.non_morphological_features.keys():
                if feature in self.tolerances[input_channel].keys():
                    features_to_use[input_channel][feature] = (
                        state.non_morphological_features[feature]
                    )

        return features_to_use

    # ----------------------- Logging --------------------------

    def _add_votes_to_buffer_stats(self, vote_data):
        """Add votes to buffer stats.

        Args:
            vote_data: Votes to add to buffer stats.
        """
        vote_stats = {"vote": vote_data}
        self.buffer.update_stats(vote_stats, update_time=False)

    def _add_detailed_stats(self, stats):
        """Not adding more stats in this one, but custom classes do.

        Returns:
            Unmodified stats.
        """
        return stats


class GraphMemory(LMMemory):
    """General GraphMemory that stores & manipulates GraphObjectModel instances.

    You can think of the GraphMemory as a library of object models with a librarian
    managing them. The books ate GraphObjectModel instances. The LearningModule classes
    access the information stored in the books and can request books to be added to the
    library.

    Subclasses are DisplacementGraphMemory, FeatureGraphMemory and EvidenceGraphMemory.
    """

    def __init__(self, graph_delta_thresholds=None, k=None):
        """Initialize a graph memory structure. This can then be filled with graphs.

        Args:
            k: integer k as in KNN, used for creating edges between observations
            graph_delta_thresholds: thresholds for determining if two observations
                are sufficiently different to both be added to the object model.

        Examples::

            graph_memory = GraphMemory()
            graph_memory._add_graph_to_memory(cup_model, "cup")
            graph_memory.reset() # Call at beginning of episode
        """
        self.graph_delta_thresholds = graph_delta_thresholds
        self.k = k
        self.mode = None
        self.models_in_memory = {}

        # Array representation of features for each graph -> faster matching
        self.feature_array = {}
        self.feature_order = {}  # Order in which features are stored in feature_array

    # =============== Public Interface Functions ===============
    # ------------------- Main Algorithm -----------------------

    def update_memory(
        self,
        locations,
        features,
        graph_id,
        object_location_rel_body,
        location_rel_model,
        object_rotation,
    ):
        """Determine how to update memory and call corresponding function."""
        if graph_id is None:
            logger.info("no match found in time, not updating memory")
        else:
            for input_channel in features.keys():
                (
                    input_channel_features,
                    input_channel_locations,
                ) = self._extract_entries_with_content(
                    features[input_channel], locations[input_channel]
                )
                # Update graph
                if (
                    graph_id in self.get_memory_ids()
                    and input_channel in self.get_input_channels_in_graph(graph_id)
                ):
                    logger.info(
                        f"{graph_id} already in memory ({self.get_memory_ids()})"
                    )
                    self._extend_graph(
                        input_channel_locations,
                        input_channel_features,
                        graph_id,
                        input_channel,
                        object_location_rel_body,
                        location_rel_model,
                        object_rotation,
                    )
                else:
                    logger.info(f"{graph_id} not in memory ({self.get_memory_ids()})")
                    print(f"building graph for {input_channel}")
                    self._build_graph(
                        input_channel_locations,
                        input_channel_features,
                        graph_id,
                        input_channel,
                    )

    def memory_consolidation(self):
        """Is here just as a placeholder.

        This could be a function that cleans up graphs in memory to make
        more efficient use of their nodes by spacing them out evenly along
        the approximated object surface. It could be something that happens
        during sleep. During clean up, similar graphs could also be merged.

        Q: Should we implement something like this?
        """
        raise NotImplementedError("memory_consolidation has not been implemented yet.")

    def initialize_feature_arrays(self):
        for graph_id in self.get_memory_ids():
            if graph_id not in self.feature_array.keys():
                self.feature_array[graph_id] = {}
                self.feature_order[graph_id] = {}
            for input_channel in self.get_input_channels_in_graph(graph_id):
                (
                    self.feature_array[graph_id][input_channel],
                    self.feature_order[graph_id][input_channel],
                ) = self._get_all_node_features(graph_id, input_channel)

    # ------------------ Getters & Setters ---------------------
    def get_graph(self, graph_id, input_channel=None):
        """Return graph from graph memory.

        Args:
            graph_id: id of graph to retrieve
            input_channel: ?

        Raises:
            ValueError: If input_channel is defined, not "first", and not in the graph
        """
        if input_channel is None:
            return self.models_in_memory[graph_id]

        if input_channel == "first":
            # Arbitrarily take first input channel. Mostly used as placeholder for now.
            # Usually this will be input from a sensor module but we do nothing to
            # guarantee this.
            first_channel = self.get_input_channels_in_graph(graph_id)[0]
            return self.models_in_memory[graph_id][first_channel]

        if input_channel in self.get_input_channels_in_graph(graph_id):
            return self.models_in_memory[graph_id][input_channel]

        raise ValueError(f"{graph_id} has no data stored for {input_channel}.")

    def get_feature_array(self, graph_id):
        return self.feature_array[graph_id]

    def get_feature_order(self, graph_id):
        return self.feature_order[graph_id]

    def get_locations_in_graph(self, graph_id, input_channel):
        return self.get_graph(graph_id, input_channel).pos

    def get_all_models_in_memory(self):
        """Return models stored in memory."""
        return self.models_in_memory.copy()

    def get_initial_hypotheses(self):
        # At the first steps all objects and locations are possible so it returns all.
        # The object and pose hypotheses are then narrowed down by the LM.
        possible_matches = self.get_all_models_in_memory()  # TODO: just List[bool]
        possible_paths = {}
        return possible_matches, possible_paths

    def get_memory_ids(self):
        """Get list of all objects in memory.

        Returns:
            List of all objects in memory.
        """
        return list(self.models_in_memory.keys())

    def get_input_channels_in_graph(self, graph_id):
        return list(self.models_in_memory[graph_id].keys())

    def get_graph_node_ids(self, graph_id, input_channel):
        num_nodes = self.models_in_memory[graph_id][input_channel].x.shape[0]
        return np.linspace(0, num_nodes - 1, num_nodes, dtype=int)

    def get_num_nodes_in_graph(self, graph_id, input_channel=None):
        """Get number of nodes in graph.

        If input_channel is None, return sum over all input channels for this object.

        Returns:
            Number of nodes in graph.
        """
        if input_channel is not None:
            return self.models_in_memory[graph_id][input_channel].x.shape[0]

        return sum(
            self.get_num_nodes_in_graph(graph_id, input_channel)
            for input_channel in self.get_input_channels_in_graph(graph_id)
        )

    def get_features_at_node(self, graph_id, input_channel, node_id, feature_keys=None):
        """Get features at a specific node in the graph.

        Args:
            graph_id: Name of graph.
            input_channel: Input channel.
            node_id: Node ID of the node to get features from. Can also be an
                array of node IDs to return an array of features.
            feature_keys: Feature keys.

        Returns:
            Dict of features at this node.

        TODO: look into getting node_id > graph.x.shape[0] (by 1)
        """
        if feature_keys is None:
            feature_keys = self.features_to_use[input_channel]
        node_features = {}
        graph = self.get_graph(graph_id, input_channel)
        if graph is None:
            logger.debug(
                f"{input_channel} not stored in graph {graph_id} yet. "
                "-> Input not used for matching."
            )
        else:
            for key in feature_keys:
                key_ids = graph.feature_mapping[key]
                feature = graph.x[node_id, key_ids[0] : key_ids[1]]
                node_features[key] = feature
        return node_features

    def state_dict(self):
        """Return state_dict."""
        return self.models_in_memory

    def __len__(self):
        """Return number of graphs in memory."""
        return len(self.get_memory_ids())

    # ------------------ Logging & Saving ----------------------
    def load_state_dict(self, state_dict):
        """Load graphs from state dict and add to memory."""
        logger.info("loading models")
        for obj_name, model in state_dict.items():
            logger.info(f"loading {obj_name} with features from {model.keys()}")
            # Add loaded graph to memory
            self._add_graph_to_memory(model, obj_name)

    # ======================= Private ==========================

    # ------------------- Main Algorithm -----------------------
    def _add_graph_to_memory(self, model, graph_id):
        """Add pretrained graph to memory.

        Initializes GridObjectModel and calls set_model.

        Args:
            model: GraphObjectModel of torch graph to be added to memory
            graph_id: id of graph that should be added

        """
        print(f"loading graph {model} of type {type(model)}")

        self.models_in_memory[graph_id] = model

    def remove_graph_from_memory(self, graph_id):
        self.models_in_memory.pop(graph_id)

    def _build_graph(self, locations, features, graph_id, input_channel):
        """Build a graph from a list of features at locations and add to memory.

        Args:
            locations: List of x,y,z locations.
            features: List of features.
            graph_id: name of new graph.
            input_channel: ?
        """
        logger.info("Adding a new graph to memory.")
        model = GraphObjectModel(
            object_id=graph_id,
        )
        graph_delta_thresholds = (
            None
            if self.graph_delta_thresholds is None
            else self.graph_delta_thresholds[input_channel]
        )
        model.build_model(
            locations,
            features,
            k_n=None,
            graph_delta_thresholds=graph_delta_thresholds,
        )
        if graph_id not in self.models_in_memory:
            self.models_in_memory[graph_id] = {}
        self.models_in_memory[graph_id][input_channel] = model

        logger.info(f"Added new graph with id {graph_id} to memory.")

    def _extend_graph(
        self,
        locations,
        features,
        graph_id,
        input_channel,
        object_location_rel_body,
        location_rel_model,
        object_rotation,
    ):
        """Add new observations into an existing graph.

        Args:
            locations: List of x,y,z locations.
            features: Features observed at the locations.
            graph_id: name of graph to be extended.
            input_channel: ?
            object_location_rel_body: location of object relative to body.
            location_rel_model: location of last observation relative to object model
            object_rotation: detected rotation of object model relative to world.
        """
        logger.info(f"Updating existing graph for {graph_id}")

        self.models_in_memory[graph_id][input_channel].update_model(
            locations=locations,
            features=features,
            location_rel_model=location_rel_model,
            object_location_rel_body=object_location_rel_body,
            object_rotation=object_rotation,
        )

        logger.info(
            f"Extended graph {graph_id} with new points. New model:\n"
            f"{self.models_in_memory[graph_id][input_channel]}"
        )

    # ------------------------ Helper --------------------------

    def _get_all_node_features(
        self, graph_id, input_channel
    ) -> tuple[np.ndarray, list]:
        """Create an array of all features for all nodes in a graph.

        This can be used for fast feature matching

        Args:
            graph_id: The graph descriptor e.g. 'mug'
            input_channel: ?

        Returns:
            An array, num_nodes x num_features
        """
        all_node_ids = self.get_graph_node_ids(graph_id, input_channel).astype(int)
        feature_arrays = self._get_empty_feature_arrays(
            graph_id, input_channel, len(all_node_ids)
        )
        feature_order = []
        # TODO: This should be possible without this for loop (currently 3rd slowest).
        for i, node_id in enumerate(all_node_ids):
            node_features = self.get_features_at_node(graph_id, input_channel, node_id)
            start_idx = 0
            for feature in node_features.keys():
                if feature in [
                    "pose_vectors",
                    "pose_fully_defined",
                ]:
                    continue
                if i == 0:
                    # Store order in which features are put in array to match
                    # correctly later
                    feature_order.append(feature)
                end_idx = start_idx + len(node_features[feature])
                feature_arrays[node_id, start_idx:end_idx] = node_features[feature]
                start_idx = end_idx
        return feature_arrays, feature_order

    def _get_empty_feature_arrays(
        self, graph_id, input_channel, num_nodes
    ) -> np.ndarray:
        """Get nan array with space for all features per input channel.

        The size of the array is calculated by taking the length of all non-pose
        features stored in the graph and adding them up. This way we can turn the
        features in the form of a nested dict into an array for more efficient matrix
        operations.

        Args:
            graph_id: Graph for which to generate this array (looks at features
                stored in this graph to determine array size)
            input_channel: ?
            num_nodes: Number of nodes that will need to be stored in this array
                (determines size of array)

        Returns:
            An array filled with nans of size (sum(feature_lens), num_nodes)
        """
        node_features = self.get_features_at_node(graph_id, input_channel, node_id=0)
        feature_array_len = 0
        for feature in node_features.keys():
            if feature in [
                "pose_vectors",
                "pose_fully_defined",
            ]:
                continue
            feature_array_len += len(node_features[feature])
        return np.zeros((num_nodes, feature_array_len)) * np.nan

    def _extract_entries_with_content(self, features, locations):
        """Only keep features & locations at steps where information was received.

        Get only the features & locations at steps where information for this input
        channel was received.

        Returns:
            Features and locations with missing features removed.
        """
        # NOTE: Could use any feature here but using pose_fully_defined since it
        # is one dimensional and a required feature in each State.
        missing_features = np.isnan(features["pose_fully_defined"]).flatten()
        # Remove missing features (contain nan values)
        locations = locations[~missing_features]
        for feature in features.keys():
            features[feature] = features[feature][~missing_features]
        return features, locations

    # ----------------------- Logging --------------------------
