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
from typing import Any, Collection, Dict, Protocol, Sequence

from tbp.monty.cmp import Goal, Message
from tbp.monty.context import RuntimeContext
from tbp.monty.experiment.learning_module import ExperimentLearningModule
from tbp.monty.experiment.monty import ExperimentMonty
from tbp.monty.experiment.sensor_module import ExperimentSensorModule
from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.environment import SemanticID
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    ProprioceptiveState,
)
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.memento import Memento, Snapshotable
from tbp.monty.observations import SensorObservation

__all__ = [
    "AgentObservations",
    "GoalGenerator",
    "LMMemory",
    "LearningModule",
    "Monty",
    "ObjectModel",
    "Observations",
    "RuntimeContext",
    "RuntimeLearningModule",
    "SensorModule",
]


class AgentObservations(Dict[SensorID, SensorObservation]):
    """Observations from an agent."""

    pass


class Observations(Dict[AgentID, AgentObservations]):
    """Observations from the environment."""

    pass


class RuntimeMonty(Protocol):
    """Runtime interface to Monty."""

    def step(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        proprioceptive_state: ProprioceptiveState,
    ) -> list[Action]:
        """Take a matching, exploratory, or custom user-defined step.

        Step taken depends on the value of self.step_type.

        Args:
            ctx: The runtime context.
            observations: The observations from the environment.
            proprioceptive_state: The proprioceptive state from the environment.

        Returns:
            The actions to take.
        """
        ...

    def motor_only_step(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        proprioceptive_state: ProprioceptiveState,
    ) -> list[Action]:
        """Take a step of the sensors and motor system only.

        This skips stepping the learning modules.

        Args:
            ctx: The runtime context.
            observations: The observations from the environment.
            proprioceptive_state: The proprioceptive state from the environment.

        Returns:
            The actions to take.
        """
        ...

    def aggregate_sensory_inputs(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        proprioceptive_state: ProprioceptiveState,
    ) -> None:
        """Receive data from environment, organize on a per sensor module basis.

        Args:
            ctx: The runtime context.
            observations: The observations from the environment.
            proprioceptive_state: The proprioceptive state from the environment.
        """
        ...


class Monty(ExperimentMonty, RuntimeMonty, Snapshotable, metaclass=abc.ABCMeta):
    def _matching_step(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        proprioceptive_state: ProprioceptiveState,
    ):
        """Step format for matching observations to graph.

        Used during training or evaluation.

        Args:
            ctx: The runtime context.
            observations: The observations from the environment.
            proprioceptive_state: The proprioceptive state from the environment.
        """
        self.aggregate_sensory_inputs(ctx, observations, proprioceptive_state)
        self._step_learning_modules(ctx)
        self._vote()
        self._pass_goals()
        self._step_motor_system(ctx, observations, proprioceptive_state)
        self._set_step_type_and_check_if_done()
        self._post_step()

    def _exploratory_step(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        proprioceptive_state: ProprioceptiveState,
    ):
        """Step format for adding data to an existing model.

        Used only during training.

        Args:
            ctx: The runtime context.
            observations: The observations from the environment.
            proprioceptive_state: The proprioceptive state from the environment.
        """
        self.aggregate_sensory_inputs(ctx, observations, proprioceptive_state)
        self._step_learning_modules(ctx)
        self._pass_goals()
        self._step_motor_system(ctx, observations, proprioceptive_state)
        self._set_step_type_and_check_if_done()
        self._post_step()

    @abc.abstractmethod
    def step(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        proprioceptive_state: ProprioceptiveState,
    ) -> list[Action]:
        pass

    @abc.abstractmethod
    def motor_only_step(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        proprioceptive_state: ProprioceptiveState,
    ) -> list[Action]:
        pass

    @abc.abstractmethod
    def aggregate_sensory_inputs(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        proprioceptive_state: ProprioceptiveState,
    ) -> None:
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
    def _pass_goals(self):
        """Pass goals in the network between learning-modules.

        Aggregate any goals for sending to the motor-system.
        """
        pass

    @abc.abstractmethod
    def _step_motor_system(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        proprioceptive_state: ProprioceptiveState,
    ):
        """Step the motor system.

        Args:
            ctx: The runtime context.
            observations: The observations from the environment.
            proprioceptive_state: The proprioceptive state from the environment.
        """
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
    def state_dict(self) -> Memento:
        pass

    @abc.abstractmethod
    def load_state_dict(self, memento: Memento) -> None:
        pass

    ###
    # Methods that interact with the experiment
    ###

    @abc.abstractmethod
    def reset(self) -> None:
        pass

    @abc.abstractmethod
    def fixme_set_ground_truth(
        self,
        primary_target: dict[str, Any] | None = None,
        semantic_id_to_label: dict[SemanticID, str] | None = None,
    ) -> None:
        pass

    @abc.abstractmethod
    def update_ltm(self) -> None:
        pass

    @abc.abstractmethod
    def set_experiment_mode(self, mode: ExperimentMode) -> None:
        pass

    @abc.abstractmethod
    def is_done(self) -> bool:
        pass


class RuntimeLearningModule(Protocol):
    """Monty runtime interface to a Learning Module."""

    def matching_step(self, ctx: RuntimeContext, percepts: Sequence[Message]) -> None:
        """Matching / inference step called inside of monty._step_learning_modules.

        Args:
            ctx: The runtime context.
            percepts: The percepts intended for this learning module.
        """
        ...

    def exploratory_step(
        self, ctx: RuntimeContext, percepts: Sequence[Message]
    ) -> None:
        """Model building step called inside of monty._step_learning_modules.

        Args:
            ctx: The runtime context.
            percepts: The percepts intended for this learning module.
        """
        ...

    def receive_votes(self, votes: Collection[Any]) -> None:
        """Process inbound voting data.

        TODO: Use `Message` type for votes rather than ad-hoc data?

        Args:
            votes: A collection of votes from other learning modules.
        """
        ...

    def send_out_vote(self) -> Any:
        """This method defines what data are sent to other learning modules.

        TODO: Use `Message` type for votes rather than ad-hoc data?

        Returns:
            This learning module's voting data.
        """
        ...

    def propose_goals(self) -> Sequence[Goal]:
        """Return the goals proposed by this LM's GSG if they exist.

        Returns:
            A collection of proposed Goals.
        """
        ...

    def get_output(self) -> Message | None:
        """Return learning module output (same format as input)."""
        ...


class LearningModule(
    RuntimeLearningModule, Snapshotable, ExperimentLearningModule, metaclass=abc.ABCMeta
):
    ###
    # Methods that interact with the experiment
    ###

    @abc.abstractmethod
    def reset_stm(self) -> None:
        pass

    @abc.abstractmethod
    def fixme_reset_ground_truth(self, primary_target=None) -> None:
        pass

    @abc.abstractmethod
    def update_ltm_from_stm(self) -> None:
        pass

    @abc.abstractmethod
    def fixme_update_ground_truth(self) -> None:
        pass

    @abc.abstractmethod
    def set_experiment_mode(self, mode: ExperimentMode) -> None:
        pass

    ###
    # Methods that define the algorithm
    ###
    @abc.abstractmethod
    def matching_step(self, ctx: RuntimeContext, percepts: Sequence[Message]) -> None:
        pass

    @abc.abstractmethod
    def exploratory_step(
        self, ctx: RuntimeContext, percepts: Sequence[Message]
    ) -> None:
        pass

    @abc.abstractmethod
    def receive_votes(self, votes: Collection[Any]) -> None:
        pass

    @abc.abstractmethod
    def send_out_vote(self) -> Any:
        pass

    @abc.abstractmethod
    def propose_goals(self) -> list[Goal]:
        pass

    @abc.abstractmethod
    def get_output(self) -> Message | None:
        pass

    ###
    # Saving, loading
    ###

    @abc.abstractmethod
    def state_dict(self) -> Memento:
        pass

    @abc.abstractmethod
    def load_state_dict(self, memento: Memento) -> None:
        pass


class LMMemory(Snapshotable, metaclass=abc.ABCMeta):
    """Like a long-term memory storing all the knowledge an LM has."""

    ###
    # Methods that define the algorithm
    ###
    @abc.abstractmethod
    def update_memory(self, observations):
        """Update models stored in memory given new observation & classification."""
        pass

    ###
    # Saving, loading
    ###

    @abc.abstractmethod
    def state_dict(self) -> Memento:
        pass

    @abc.abstractmethod
    def load_state_dict(self, memento: Memento) -> None:
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


class GoalGenerator(metaclass=abc.ABCMeta):
    """Generate goals that other learning modules and motor-systems will attempt.

    Generate goals potentially (in the case of LMs) by outputting their own
    sub-goals. Provides a mechanism for implementing hierarchical action policies
    that are informed by world models/hypotheses.
    """

    @abc.abstractmethod
    def set_driving_goal(self):
        """Set the driving goal.

        e.g., from a human operator or a high-level LM.
        """
        pass

    @abc.abstractmethod
    def output_goals(self) -> list[Goal]:
        """Return output goals."""
        pass

    @abc.abstractmethod
    def step(self, ctx: RuntimeContext, observations: Observations):
        """Called on each step of the LM to which the GSG belongs."""
        pass


class SensorModule(ExperimentSensorModule, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def state_dict(self) -> Memento:
        pass

    @abc.abstractmethod
    def update_state(self, agent: AgentState) -> None:
        pass

    @abc.abstractmethod
    def step(
        self,
        ctx: RuntimeContext,
        observation: SensorObservation,
        motor_only_step: bool = False,
    ) -> Message | None:
        pass

    def propose_goals(self) -> Collection[Goal]:
        return []

    @abc.abstractmethod
    def reset(self) -> None:
        pass
