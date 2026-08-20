# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""ARC-specific child-to-parent learning-module lifecycle."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from tbp.monty.cmp import Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.arc_sensor_module import ARC_FRAME_POSE
from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.motor_system_state import ProprioceptiveState
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.simulators.arc_agi.region_scan import SetArcRegionPose

__all__ = ["MontyForArcCompositionalLearning"]


class MontyForArcCompositionalLearning(MontyForEvidenceGraphMatching):
    """Recognize ARC regions and accumulate their identities in one parent map.

    Parent training uses oracle region labels. Evaluation only emits independently
    matched child predictions.
    """

    def __init__(
        self,
        *args,
        child_lm_index: int = 0,
        parent_lm_index: int = 1,
        region_sensor_id: SensorID | str = "patch_0",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        lm_count = len(self.learning_modules)
        if not 0 <= child_lm_index < lm_count:
            raise IndexError(f"child_lm_index {child_lm_index} is out of range")
        if not 0 <= parent_lm_index < lm_count:
            raise IndexError(f"parent_lm_index {parent_lm_index} is out of range")
        if child_lm_index == parent_lm_index:
            raise ValueError("child and parent learning modules must be different")
        self.child_lm_index = child_lm_index
        self.parent_lm_index = parent_lm_index
        self.region_sensor_id = SensorID(region_sensor_id)
        self.region_results: list[dict[str, Any]] = []
        self._pending_region_id: str | None = None
        self._region_emission_output: Message | None = None
        self._region_scan_complete = False

    @property
    def child_lm(self):
        return self.learning_modules[self.child_lm_index]

    @property
    def parent_lm(self):
        return self.learning_modules[self.parent_lm_index]

    def reset(self) -> None:
        super().reset()
        self.region_results = []
        self._pending_region_id = None
        self._region_emission_output = None
        self._region_scan_complete = False

    def set_experiment_mode(self, mode: ExperimentMode) -> None:
        super().set_experiment_mode(mode)
        # Child models are pretrained and remain read-only while the parent learns.
        self.child_lm.set_experiment_mode(ExperimentMode.EVAL)

    def update_ltm(self) -> None:
        """Commit only the parent map; the pretrained child remains frozen."""
        self.parent_lm.update_ltm_from_stm()
        self.parent_lm.fixme_update_ground_truth()

    def aggregate_sensory_inputs(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        proprioceptive_state: ProprioceptiveState,
    ) -> None:
        super().aggregate_sensory_inputs(ctx, observations, proprioceptive_state)
        region_observation = self.get_observations(observations, self.region_sensor_id)
        if region_observation.get("region_phase") != "emit":
            return

        if self._region_emission_output is None:
            raise RuntimeError("ARC emit phase requires one finalized child identity")
        child_output = copy.deepcopy(self._region_emission_output)
        child_output.location = np.asarray(
            region_observation["region_location"], dtype=float
        )
        self.learning_module_outputs[self.child_lm_index] = child_output

    def _set_step_type_and_check_if_done(self) -> None:
        """Update local terminal states without ending the enclosing map context.

        Raises:
            RuntimeError: If another step type bypasses region recognition.
        """
        if self.step_type != "matching_step":
            raise RuntimeError("ARC compositional lifecycle requires matching steps")
        self.update_step_counters()
        if not self.check_if_any_lms_updated():
            self.matching_steps -= 1
            return
        for learning_module in self.learning_modules:
            if learning_module.buffer.get_last_obs_processed():
                learning_module.update_terminal_condition()

    def _step_motor_system(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        proprioceptive_state: ProprioceptiveState,
    ) -> None:
        """Let the final boundary observation finish before ending the episode."""
        try:
            super()._step_motor_system(ctx, observations, proprioceptive_state)
        except StopIteration:
            self._actions = []
            self._region_scan_complete = True

    def step(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        proprioceptive_state: ProprioceptiveState,
    ) -> list[Action]:
        region_observation = self.get_observations(observations, self.region_sensor_id)
        current_phase = region_observation.get("region_phase")
        actions = super().step(ctx, observations, proprioceptive_state)
        region_action = next(
            (action for action in actions if isinstance(action, SetArcRegionPose)),
            None,
        )

        if (
            current_phase == "scan"
            and region_action is not None
            and region_action.phase == "skip"
        ):
            self._finalize_region(region_observation, region_action)

        if current_phase in {"emit", "skip"} and self._pending_region_id is not None:
            region_id = region_observation.get("region_id")
            if region_id != self._pending_region_id:
                raise RuntimeError(
                    f"ARC boundary for {region_id!r} does not match pending region "
                    f"{self._pending_region_id!r}"
                )
            emission_continues = bool(
                current_phase == "emit"
                and region_action is not None
                and region_action.phase == "emit"
                and region_action.region_id == region_id
            )
            if not emission_continues:
                self.child_lm.reset_stm()
                self._pending_region_id = None
                self._region_emission_output = None

        if self._region_scan_complete:
            self._region_scan_complete = False
            raise StopIteration

        return actions

    def _finalize_region(
        self,
        observation,
        boundary_action: SetArcRegionPose,
    ) -> None:
        child_output = self.child_lm.get_output()
        current_mlh = self.child_lm.get_current_mlh()
        matched = bool(
            child_output is not None
            and child_output.use_state
            and self.child_lm.terminal_state == "match"
        )
        teacher_forced = self.experiment_mode is ExperimentMode.TRAIN
        emitted = matched or teacher_forced
        boundary_action.phase = "emit" if emitted else "skip"
        region_id = observation["region_id"]
        self._pending_region_id = region_id
        if teacher_forced:
            self._region_emission_output = Message(
                location=np.asarray(observation["region_anchor"], dtype=float),
                morphological_features={
                    "pose_vectors": ARC_FRAME_POSE.copy(),
                    "pose_fully_defined": True,
                    "on_object": True,
                },
                non_morphological_features={
                    "object_id": self.child_lm._object_id_to_features(
                        observation["object_label"]
                    )
                },
                confidence=1.0,
                use_state=True,
                sender_id=self.child_lm.learning_module_id,
                sender_type="LM",
            )
        elif matched:
            self._region_emission_output = copy.deepcopy(child_output)
        else:
            self._region_emission_output = None
        self.region_results.append(
            {
                "region_id": region_id,
                "object_label": observation["object_label"],
                "predicted_object_id": self.child_lm.detected_object,
                "mlh_object_id": current_mlh["graph_id"],
                "mlh_evidence": float(current_mlh["evidence"]),
                "terminal_state": self.child_lm.terminal_state,
                "confidence": (
                    float(child_output.confidence) if child_output is not None else 0.0
                ),
                "observation_count": int(
                    self.child_lm.buffer.get_num_observations_on_object()
                ),
                "emitted": emitted,
                "emission_source": (
                    "oracle" if teacher_forced else "match" if matched else None
                ),
                "global_anchor": tuple(observation["region_anchor"]),
            }
        )
