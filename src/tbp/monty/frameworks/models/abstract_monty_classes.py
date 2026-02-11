# Copyright 2025-2026 Thousand Brains Project
# Copyright 2021-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import abc
from typing import Dict, TypedDict

import numpy as np
import numpy.typing as npt

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.motor_system_state import AgentState
from tbp.monty.frameworks.models.states import GoalState
from tbp.monty.frameworks.sensors import SensorID

__all__ = [
    "AgentObservations",
    "GoalStateGenerator",
    "LMMemory",
    "LearningModule",
    "Monty",
    "ObjectModel",
    "Observations",
    "RuntimeContext",
    "SensorModule",
    "SensorObservations",
]


class SensorObservations(TypedDict, total=False):
    """Observations from a sensor."""

    rgba: npt.NDArray[np.int_]  # TODO: Verify specific type
    depth: npt.NDArray[np.float64]  # TODO: Verify specific type
    semantic: npt.NDArray[np.int_]  # TODO: Verify specific type
    semantic_3d: npt.NDArray[np.int_]  # TODO: Verify specific type
    sensor_frame_data: npt.NDArray[np.int_]  # TODO: Verify specific type
    world_camera: npt.NDArray[np.float64]  # TODO: Verify specific type
    pixel_loc: npt.NDArray[np.float64]  # TODO: Verify specific type
    raw: npt.NDArray[np.uint8]


class AgentObservations(Dict[SensorID, SensorObservations]):
    """Observations from an agent."""

    pass


class Observations(Dict[AgentID, AgentObservations]):
    """Observations from the environment."""

    pass


class Monty(metaclass=abc.ABCMeta):
    ###
    # Methods that specify the algorithm
    ###
    def _matching_step(self, ctx: RuntimeContext, observation):
        """Step format for matching observations to graph.

        Used during training or evaluation.
        """
        self.aggregate_sensory_inputs(ctx, observation)
        self._step_learning_modules(ctx)
        self._vote()
        self._pass_goal_states()
        self._pass_infos_to_motor_system()
        self._set_step_type_and_check_if_done()
        self._post_step()

    def _exploratory_step(self, ctx: RuntimeContext, observation):
        """Step format for adding data to an existing model.

        Used only during training.
        """
        self.aggregate_sensory_inputs(ctx, observation)
        self._step_learning_modules(ctx)
        self._pass_goal_states()
        self._pass_infos_to_motor_system()
        self._set_step_type_and_check_if_done()
        self._post_step()

    @abc.abstractmethod
    def step(self, ctx: RuntimeContext, observation):
        """Take a matching, exploratory, or custom user-defined step.

        Step taken depends on the value of self.step_type.
        """
        pass

    @abc.abstractmethod
    def aggregate_sensory_inputs(self, ctx: RuntimeContext, observation):
        """Receive data from environment, organize on a per sensor module basis."""
        pass

    @abc.abstractmethod
    def _step_learning_modules(self, ctx: RuntimeContext):
        """Pass data from SMs to LMs, and have each LM take a step.

        LM step type depends on self.step_type.
        """
        pass

    @abc.abstractmethod
    def _vote(self):
        """Share information across learning modules.

        Use LM.send_out_vote and LM.receive_votes.
        """
        pass

    @abc.abstractmethod
    def _pass_goal_states(self):
        """Pass goal states in the network between learning-modules.

        Aggregate any goal states for sending to the motor-system.
        """
        pass

    @abc.abstractmethod
    def _pass_infos_to_motor_system(self):
        """Pass input observations and goal states to the motor system."""
        pass

    @abc.abstractmethod
    def _set_step_type_and_check_if_done(self):
        """Check terminal conditions and decide if to change the step type.

        Update what self.is_done returns to the experiment.
        """
        pass

    @abc.abstractmethod
    def _post_step(self):
        """Hook for doing things like updating counters."""
        pass

    ###
    # Saving, loading, and logging
    ###

    @abc.abstractmethod
    def state_dict(self):
        """Return a serializable dict with everything needed to save/load monty."""
        pass

    @abc.abstractmethod
    def load_state_dict(self, state_dict):
        """Take a state dict as an argument and set state for monty and children."""
        pass

    ###
    # Methods that interact with the experiment
    ###

    @abc.abstractmethod
    def pre_episode(self) -> None:
        """Recursively call pre_episode on child classes."""
        pass

    @abc.abstractmethod
    def post_episode(self):
        """Recursively call post_episode on child classes."""
        pass

    @abc.abstractmethod
    def set_experiment_mode(self, mode: ExperimentMode) -> None:
        """Set the experiment mode.

        Update state variables based on which method (train or evaluate) is being
        called at the experiment level.

        Args:
            mode: The experiment mode.
        """
        pass

    @abc.abstractmethod
    def is_done(self):
        """Return bool to tell the experiment if we are done with this episode."""
        pass


class LearningModule(metaclass=abc.ABCMeta):
    ###
    # Methods that interact with the experiment
    ###
    @abc.abstractmethod
    def reset(self):
        """Do things like reset buffers or possible_matches before training."""
        pass

    @abc.abstractmethod
    def pre_episode(self) -> None:
        """Do things like reset buffers or possible_matches before training."""
        pass

    @abc.abstractmethod
    def post_episode(self):
        """Do things like update object models with stored data after an episode."""
        pass

    @abc.abstractmethod
    def set_experiment_mode(self, mode: ExperimentMode) -> None:
        """Set the experiment mode.

        Update state variables based on which method (train or evaluate) is being called
        at the experiment level.

        Args:
            mode: The experiment mode.
        """
        pass

    ###
    # Methods that define the algorithm
    ###
    @abc.abstractmethod
    def matching_step(self, ctx: RuntimeContext):
        """Matching / inference step called inside of monty._step_learning_modules."""
        pass

    @abc.abstractmethod
    def exploratory_step(self, ctx: RuntimeContext):
        """Model building step called inside of monty._step_learning_modules."""
        pass

    @abc.abstractmethod
    def receive_votes(self, votes):
        """Process voting data sent out from other learning modules."""
        pass

    @abc.abstractmethod
    def send_out_vote(self):
        """This method defines what data are sent to other learning modules."""
        pass

    @abc.abstractmethod
    def propose_goal_states(self) -> list[GoalState]:
        """Return the goal-states proposed by this LM's GSG if they exist."""
        pass

    @abc.abstractmethod
    def get_output(self):
        """Return learning module output (same format as input)."""
        pass

    ###
    # Saving, loading
    ###

    @abc.abstractmethod
    def state_dict(self):
        """Return a serializable dict with everything needed to save/load this LM."""
        pass

    @abc.abstractmethod
    def load_state_dict(self, state_dict):
        """Take a state dict as an argument and set state for this LM."""
        pass


class LMMemory(metaclass=abc.ABCMeta):
    """Like a long-term memory storing all the knowledge an LM has."""

    ###
    # Methods that define the algorithm
    ###
    @abc.abstractmethod
    def update_memory(self, observations):
        """Update models stored in memory given new observation & classification."""
        pass

    @abc.abstractmethod
    def memory_consolidation(self):
        """Consolidate/clean up models stored in memory."""
        pass

    ###
    # Saving, loading
    ###

    @abc.abstractmethod
    def state_dict(self):
        """Return a serializable dict with everything needed to save/load the memory."""
        pass

    @abc.abstractmethod
    def load_state_dict(self):
        """Take a state dict as an argument and set state for the memory."""
        pass


class ObjectModel(metaclass=abc.ABCMeta):
    """Model of an object. Is stored in Memory and used by LM."""

    @abc.abstractmethod
    def build_model(self, observations):
        """Build a new model."""
        pass

    @abc.abstractmethod
    def update_model(self, observations):
        """Update an existing model with new observations."""
        pass


class GoalStateGenerator(metaclass=abc.ABCMeta):
    """Generate goal-states that other learning modules and motor-systems will attempt.

    Generate goal-states potentially (in the case of LMs) by outputting their own
    sub-goal-states. Provides a mechanism for implementing hierarchical action policies
    that are informed by world models/hypotheses.
    """

    @abc.abstractmethod
    def set_driving_goal_state(self):
        """Set the driving goal state.

        e.g., from a human operator or a high-level LM.
        """
        pass

    @abc.abstractmethod
    def output_goal_states(self) -> list[GoalState]:
        """Return output goal-states."""
        pass

    @abc.abstractmethod
    def step(self, ctx: RuntimeContext, observations: Observations):
        """Called on each step of the LM to which the GSG belongs."""
        pass


class SensorModule(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def state_dict(self):
        """Return a serializable dict with this sensor module's state.

        Includes everything needed to save/load this sensor module.
        """
        pass

    @abc.abstractmethod
    def update_state(self, agent: AgentState):
        pass

    @abc.abstractmethod
    def step(self, ctx: RuntimeContext, data):
        """Called on each step.

        Args:
            ctx: The runtime context.
            data: Sensor observations
        """
        pass

    @abc.abstractmethod
    def pre_episode(self) -> None:
        """This method is called before each episode."""
        pass

    def propose_goal_states(self) -> list[GoalState]:
        """Return the goal-states proposed by this Sensor Module."""
        return []
