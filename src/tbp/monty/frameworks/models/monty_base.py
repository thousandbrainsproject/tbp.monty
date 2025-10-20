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

from tbp.monty.frameworks.loggers.exp_logger import TestLogger
from tbp.monty.frameworks.models.abstract_monty_classes import Monty
from tbp.monty.frameworks.models.motor_system import MotorSystem
from tbp.monty.frameworks.utils.communication_utils import get_first_sensory_state

logger = logging.getLogger(__name__)


class MontyBase(Monty):
    LOGGING_REGISTRY = dict(TEST=TestLogger)

    def __init__(
        self,
        sensor_modules,
        learning_modules,
        motor_system: MotorSystem,
        sm_to_agent_dict,
        sm_to_lm_matrix,
        lm_to_lm_matrix,
        lm_to_lm_vote_matrix,
        min_eval_steps,
        min_train_steps,
        num_exploratory_steps,
        max_total_steps,
    ):
        """Initialize the base class.

        Args:
            sensor_modules: list of sensor modules
            learning_modules: list of learning modules
            motor_system: class instance that aggregates proposed motor outputs
                of learning modules and decides next action. Conceptually, this is
                the subcortical motor area. Note: EnvironmentDataLoader takes a
                motor_system as an argument. That motor system is the same as this
                one.
            sm_to_agent_dict: dictionary mapping each sensor module id to the
                list of habitat agents it receives input from. This is to simulate
                columns with wide receptive fields that receive input from multiple
                sensors. TODO: Do we still need this?
            sm_to_lm_matrix: nested array that governs which sensor modules a
                learning module will receive input from, such that
                learning_modules[sm_to_lm_matrix[i][j]] is the jth sensor module
                that learning module i receives input from. For now, assume 1:1 mapping.
                Technically, this is a coupling matrix, but since it is sparse, the
                argument format is an array of arrays.
            lm_to_lm_matrix: just like sm_to_lm_matrix, but describes coupling
                between learning modules where one lms output becomes another lms input.
                The output of an lm needs to be the same format as its input.
            lm_to_lm_vote_matrix: describes lateral coupling between learning
                modules. This matrix is used for voting. Assumes no lateral voting if
                `None` is passed.
            min_eval_steps: Minimum number of steps required for evaluations.
            min_train_steps: Minimum number of steps required for training.
            num_exploratory_steps: Number of steps required by the exploratory phase.
            max_total_steps: Maximum number of steps to run the experiment.

        Raises:
            ValueError: If `sm_to_lm_matrix` is not defined
            ValueError: If the lengths of `learning_modules` and `sm_to_lm_matrix`
                do not match
            ValueError: If the keys of `sm_to_agent_dict` do not match the
                `sensor_module_id`s of `sensor_modules`
        """
        # Basic instance attributes
        self.sensor_modules = sensor_modules
        self.learning_modules = learning_modules
        self.motor_system = motor_system
        self.sm_to_agent_dict = sm_to_agent_dict
        self.sm_to_lm_matrix = sm_to_lm_matrix
        self.lm_to_lm_matrix = lm_to_lm_matrix
        self.lm_to_lm_vote_matrix = lm_to_lm_vote_matrix
        self.min_eval_steps = min_eval_steps
        self.min_train_steps = min_train_steps
        self.num_exploratory_steps = num_exploratory_steps
        self.max_total_steps = max_total_steps

        # Counters, logging, default step_type
        self.step_type = "matching_step"
        self.is_seeking_match = True  # for consistency with custom monty experiments
        self.experiment_mode = None  # initialize to neither training nor testing
        self.total_steps = 0
        # number of overall steps. Counts also steps where no LM update was perfomed.
        self.episode_steps = 0
        # Number of steps in which at least 1 LM received information during exploration
        self.exploratory_steps = 0
        # Number of steps in which at least 1 LM was updated. Is not the same as each
        # individual LMs number of matching steps
        self.matching_steps = 0

        if self.sm_to_lm_matrix is None:
            raise ValueError("sm_to_lm_matrix must be defined")

        # Validate number of learning modules
        lm_len = len(self.learning_modules)
        if lm_len != len(self.sm_to_lm_matrix):
            raise ValueError(
                "The lengths of learning_modules and sm_to_lm_matrix must match"
            )

        if self.lm_to_lm_vote_matrix is not None and lm_len != len(
            self.lm_to_lm_vote_matrix
        ):
            raise ValueError(
                "The lengths of learning_modules and lm_to_lm_vote_matrix must match"
            )

        # Check that every sensor module is assigned to an agent.
        sm_ids = [sm.sensor_module_id for sm in self.sensor_modules]
        if set(sm_ids) != set(self.sm_to_agent_dict.keys()):
            raise ValueError(
                "sm_to_agent_dict must contain exactly one key for each "
                "sensor_module id; no more, no less!"
            )

    ###
    # Basic methods that specify the algorithm
    ###

    def step(self, observation):
        # For the base class, just use matching step. Note that matching_step and
        # exploratory_step are fully implemented by the abstract class.
        if self.step_type == "matching_step":
            self._matching_step(observation)
        elif self.step_type == "exploratory_step":
            self._exploratory_step(observation)
        else:
            raise ValueError(f"step type {self.step_type} not found in base monty")

    def aggregate_sensory_inputs(self, observation):
        sensor_module_outputs = []
        for sensor_module in self.sensor_modules:
            raw_obs = self.get_observations(observation, sensor_module.sensor_module_id)
            sensor_module.update_state(self.get_agent_state())
            sm_output = sensor_module.step(raw_obs)
            sensor_module_outputs.append(sm_output)
        # Aggregate LM outputs here to be input to higher level LM at next step
        learning_module_outputs = []
        for learning_module in self.learning_modules:
            lm_out = learning_module.get_output()
            learning_module_outputs.append(lm_out)
        self.sensor_module_outputs = sensor_module_outputs
        # TODO: Maybe combine the two?
        self.learning_module_outputs = learning_module_outputs

    def pass_features_directly_to_motor_system(self, observation):
        """Pass features directly to motor system without stepping LMs."""
        self.aggregate_sensory_inputs(observation)
        self._pass_input_obs_to_motor_system(
            get_first_sensory_state(self.sensor_module_outputs)
        )
        self.total_steps += 1
        self.episode_steps += 1
        # For each of the learning modules, update the list of processed
        # steps, and the stepwise target object
        for ii in range(len(self.learning_modules)):
            self.learning_modules[ii].add_lm_processing_to_buffer_stats(
                lm_processed=False
            )
            self.learning_modules[ii].stepwise_targets_list.append(
                self.learning_modules[ii].stepwise_target_object
            )

    def check_reached_max_matching_steps(self, max_steps):
        """Check if max_steps was reached and deal with time_out.

        Returns:
            True if max_steps was reached, False otherwise.
        """
        if (
            self.is_seeking_match and self.matching_steps >= max_steps
            # Since we increment matching steps from 0 (i.e. the first matching
            # step is the "0th" step, this is set to >=, not >)
        ):
            self.deal_with_time_out()
            return True
        else:
            return False

    def deal_with_time_out(self):
        """Call any functions and logging in case of a time out."""
        pass

    def _step_learning_modules(self):
        for i in range(len(self.learning_modules)):
            sensory_inputs = self._collect_inputs_to_lm(i)
            getattr(self.learning_modules[i], self.step_type)(sensory_inputs)

    def _collect_inputs_to_lm(self, lm_id):
        """Use sm_to_lm_matrix and lm_to_lm_matrix to collect inputs to LM i.

        Args:
            lm_id: Index of receiving LM to collect inputs to.

        Returns:
            Sensory inputs to the LM.
        """
        sensory_inputs_from_sms = [
            self.sensor_module_outputs[j] for j in self.sm_to_lm_matrix[lm_id]
        ]
        if self.lm_to_lm_matrix is not None:
            sensory_inputs_from_lms = [
                self.learning_module_outputs[j] for j in self.lm_to_lm_matrix[lm_id]
            ]
        else:
            sensory_inputs_from_lms = []
        # Combine sensory inputs from SMs and LMs to LM i
        sensory_inputs = self._combine_inputs(
            sensory_inputs_from_sms, sensory_inputs_from_lms
        )
        return sensory_inputs

    def _combine_inputs(self, inputs_from_sms, inputs_from_lms) -> dict | None:
        """Combine all inputs to an LM into one dict.

        An LM only receives input from another LM if it also receives input from
        an SM. This makes sure that we keep a coarser resolution in the higher
        level LM.
        TODO H: Is this how we want to solve this? May want to change this in the future
        allowing high-level LMs that are not connected to SMs.

        TODO H: Take into account distance from center of receiving LMs RF. To do that
        in a good way, combine_input or LM selection may have to become part of LM class

        Args:
            inputs_from_sms: List of dicts of SM outputs.
            inputs_from_lms: List of dicts of LM outputs.

        Returns:
            Combined features and location from all inputs with interesting features.
            If there are no inputs or none of them are deemed interesting (i.e. off
            object or low confidence LM) this returns None.
        """
        combined_inputs = [
            inputs_from_sms[i]
            for i in range(len(inputs_from_sms))
            if inputs_from_sms[i].use_state
        ]
        if len(combined_inputs) == 0:
            # If we have no sensory input, we also don't use LM input
            return None

        for lm_input in inputs_from_lms:
            if lm_input.use_state:
                combined_inputs.append(lm_input)
        return combined_inputs

    def _vote(self):
        if self.lm_to_lm_vote_matrix is not None:
            # Send out votes
            votes_per_lm = []
            for i in range(len(self.learning_modules)):
                votes_per_lm.append(self.learning_modules[i].send_out_vote())
            # Receive votes
            for i in range(len(self.learning_modules)):
                voting_data = [votes_per_lm[j] for j in self.lm_to_lm_vote_matrix[i]]
                self.learning_modules[i].receive_votes(voting_data)

    def _pass_goal_states(self) -> None:
        """Pass goal states between learning modules.

        Currently we just aggregate these for later passing to the (single) motor
        system.

        TODO M implement more complex, hierarchical passing of goal-states.
        """
        self.gsg_outputs = []  # NB we reset these at each step to ensure the goal
        # states do not persist unless this is expected by the GSGs. NOTE we may need
        # to revisit this with heterarchy if we have some LMs that are being stepped
        # at higher frequencies than others.

        # Currently only use GSG outputs at inference
        if self.step_type == "matching_step":
            for lm in self.learning_modules:
                goal_states = lm.propose_goal_states()
                self.gsg_outputs.extend(goal_states)

    def _pass_infos_to_motor_system(self):
        """Pass input observations and goal states to the motor system."""
        pass

    def _set_step_type_and_check_if_done(self):
        """Check terminal conditions and decide if we change the step type.

        In the base case, we just run until min_{exploratory, train, eval} steps is
        reached, and then either change step type or end the experiment. NOTE: it is up
        to you to override this method and define your experimental flow.
        """
        self.update_step_counters()

        if self.exceeded_min_steps:
            if self.step_type == "exploratory_step":
                self._is_done = True
                logger.info(f"finished exploring after {self.exploratory_steps} steps")

            elif self.step_type == "matching_step":
                if self.experiment_mode == "train":
                    self.switch_to_exploratory_step()
                else:
                    self._is_done = True
                    logger.info(
                        f"finished evaluating after {self.matching_steps} steps"
                    )

    def _post_step(self):
        pass

    ###
    # Methods (other than step) that interact with the experiment
    ###

    def set_experiment_mode(self, mode):
        assert mode in ["train", "eval"], "mode must be either `train` or `eval`"
        self.experiment_mode = mode
        self.motor_system.set_experiment_mode(mode)
        self.step_type = "matching_step"
        for lm in self.learning_modules:
            lm.set_experiment_mode(mode)
        # for sm in self.sensor_modules: sm.set_experiment_mode() unused & removed

    def pre_episode(self):
        self._is_done = False
        self.reset_episode_steps()
        self.switch_to_matching_step()
        for lm in self.learning_modules:
            lm.pre_episode()

        for sm in self.sensor_modules:
            sm.pre_episode()

    def post_episode(self):
        for lm in self.learning_modules:
            lm.post_episode()
        # for sm in self.sensor_modules: sm.post_episode() unused & removed

    ###
    # Methods for saving and loading
    ###

    def load_state_dict(self, state_dict):
        assert len(state_dict["lm_dict"]) == len(self.learning_modules)
        lm_counter = 0
        lm_dict = state_dict["lm_dict"]
        for lm_key in lm_dict.keys():
            self.learning_modules[lm_counter].load_state_dict(lm_dict[lm_key])
            lm_counter = lm_counter + 1

    def state_dict(self):
        lm_dict = {
            i: module.state_dict() for i, module in enumerate(self.learning_modules)
        }
        sm_dict = {
            i: module.state_dict() for i, module in enumerate(self.sensor_modules)
        }
        motor_system_dict = self.motor_system._policy.state_dict()

        return dict(
            lm_dict=lm_dict,
            sm_dict=sm_dict,
            motor_system_dict=motor_system_dict,
            lm_to_lm_matrix=self.lm_to_lm_matrix,
            lm_to_lm_vote_matrix=self.lm_to_lm_vote_matrix,
            sm_to_lm_matrix=self.sm_to_lm_matrix,
        )

    ###
    # Helper methods for methods that specify the algorithm
    ###

    def get_observations(self, observations, sensor_module_id):
        """Get observations from all agents pertaining to a single sensor module.

        Observations are returned in the format
            {"agent_1":
                {"sm_1":
                    {"rgba": data,
                     "depth": data
                     "semantic": data}
                }
                {"sm_2":
                    {"rgba": data,
                    ...
                    }
                }
                ...
            "agent_2":
                {"sm_3":
                    {"rgba": data,
                    ...
                    }
                }
                ...
            ...
            "agent_n":
                {"sm_k":
                    {"rgba": data,
                    ...
                    }
                }
            }

        Returns:
            Observations from all agents pertaining to a single sensor module.
        """
        # the agent (actuator) this sensor is attached to
        agent_id = self.sm_to_agent_dict[sensor_module_id]
        agent_obs = observations[agent_id]
        sensor_obs = agent_obs[sensor_module_id]
        return sensor_obs

    def get_agent_state(self):
        """Get state of agent (dict).

        Returns:
            State of the agent.
        """
        # TODO: This is left in place for now to keep PR scope limited, but should be
        #       refactored in the future to simplify this access pattern.
        return self.motor_system._policy.get_agent_state(self.motor_system._state)

    @property
    def is_motor_only_step(self):
        return self.motor_system._policy.is_motor_only_step(self.motor_system._state)

    @property
    def is_done(self):
        return self._is_done

    def set_done(self):
        self._is_done = True

    @property
    def min_steps(self):
        if self.step_type == "matching_step":
            if self.experiment_mode == "train":
                return self.min_train_steps
            elif self.experiment_mode == "eval":
                return self.min_eval_steps
        elif self.step_type == "exploratory_step":
            return self.num_exploratory_steps

    @property
    def step_type_count(self):
        if self.step_type == "matching_step":
            return self.matching_steps
        elif self.step_type == "exploratory_step":
            return self.exploratory_steps

    @property
    def exceeded_min_steps(self):
        return self.step_type_count > self.min_steps

    def reset_episode_steps(self):
        self.episode_steps = 0
        self.matching_steps = 0
        self.exploratory_steps = 0

    def update_step_counters(self):
        self.total_steps += 1
        self.episode_steps += 1

        if self.step_type == "matching_step":
            self.matching_steps += 1
            logger.info(f"--- Global Matching Step {self.matching_steps} ---")
        elif self.step_type == "exploratory_step":
            self.exploratory_steps += 1

    def switch_to_matching_step(self):
        self.step_type = "matching_step"
        self.is_seeking_match = True
        logger.debug(f"Going into matching mode after {self.episode_steps} steps")

    def switch_to_exploratory_step(self):
        self.step_type = "exploratory_step"
        self.is_seeking_match = False
        logger.info(f"Going into exploratory mode after {self.matching_steps} steps")
