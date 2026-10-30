# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Mapping, MutableMapping, Sequence, cast

import numpy as np
from pydantic import BaseModel
from typing_extensions import Self

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.abstract_monty_classes import LearningModule

if TYPE_CHECKING:
    from tbp.monty.frameworks.models.evidence_matching.learning_module import (
        EvidenceGraphLM,
    )
    from tbp.monty.frameworks.models.feature_location_matching import FeatureGraphLM
    from tbp.monty.frameworks.models.graph_matching import GraphLM


class TerminalState(Enum):
    UNDEFINED = "undefined"
    MATCH = "match"
    NO_MATCH = "no_match"
    TIME_OUT = "time_out"  # TODO: from MontyForGraphMatching._set_time_outs, yes/no?

    @classmethod
    def from_graph_lm(cls, lm: GraphLM) -> TerminalState:
        """Extracts the LM terminal state and turns it into an enum value.

        Returns:
            The enum value.
        """
        # TODO: maybe unneeded, do directly in set_individual_ts?
        try:
            return cls(lm.terminal_state)
        except ValueError:
            return cls.UNDEFINED


class LMTelemetryModel(BaseModel):
    """Learning module identity info shared by LM-related telemetry events."""

    lm_id: str
    """Learning module ID."""

    lm_type: type[LearningModule]
    """Learning module type."""


class LMRecognitionState(BaseModel):
    """Learning module state data emitted upon object recognition."""

    terminal_state: TerminalState
    """Terminal state of the learning module."""

    detected_object: str | None
    """ID of the object detected by the learning module."""


class LMRecognitionStateTelemetry(LMTelemetryModel):
    """Learning module state data emitted upon object recognition."""

    recognition_state: LMRecognitionState

    @classmethod
    def from_graph_lm(cls, lm: GraphLM) -> Self:
        """Populates an event from a `GraphLM`.

        Returns:
            The event schema.
        """
        return cls(
            lm_id=lm.learning_module_id,
            lm_type=type(lm),
            recognition_state=LMRecognitionState(
                terminal_state=TerminalState.from_graph_lm(lm),
                detected_object=lm.detected_object,
            ),
        )


class LMStepState(BaseModel):
    """Learning module state data emitted at every step."""

    is_on_object: bool
    """Indicates if ``GraphLM.buffer.on_object[0]`` is ``True``."""

    matching_steps: int
    """Result of `GraphLM.buffer.get_num_matching_steps()`."""

    location: Sequence[float]
    """Result of `GraphLM.buffer.get_current_location()`."""

    buffer_stats: MutableMapping
    """Equivalent of `GraphLM.buffer.stats`."""

    stepwise_target_object: str | None
    """Equivalent of `GraphLM.stepwise_target_object`."""

    graph_id_to_target: Mapping
    """Equivalent of `GraphLM.target_to_graph_id`."""

    target_to_graph_id: Mapping
    """Equivalent of `GraphLM.target_to_graph_id`."""

    detected_pose: Sequence[float | None]
    """Equivalent of `GraphLM.detected_pose`."""

    symmetry_evidence: int
    """Equivalent of `GraphLM.symmetry_evidence`."""

    current_mlh: Mapping | None
    """Result of `EvidenceGraphLM.buffer.get_current_mlh()`."""

    possible_matches: Sequence[str]
    """Result of `GraphLM.get_possible_matches()`."""

    possible_paths: Mapping
    """Equivalent of `GraphLM.possible_paths`."""

    possible_poses: Mapping | None
    """Equivalent of `GraphLM.possible_poses`."""

    path_similarity_threshold: float | None
    """Equivalent of `EvidenceGraphLM.path_similarity_threshold`."""

    pose_similarity_threshold: float | None
    """Equivalent of `EvidenceGraphLM.pose_similarity_threshold`."""


class LMStateTelemetry(LMTelemetryModel):
    """Event schema containing learning module state data emitted at every step."""

    recognition_state: LMRecognitionState
    step_state: LMStepState

    @classmethod
    def from_graph_lm(cls, lm: GraphLM) -> Self:
        """Populates a new event from a `GraphLM`.

        Returns:
            The event schema.
        """
        # First, try to get data that is specific to child classes of GraphLM

        eg_lm = cast("EvidenceGraphLM", lm)

        try:
            current_mlh = eg_lm._get_current_mlh()
        except AttributeError:
            current_mlh = None

        try:
            path_similarity_threshold = eg_lm.path_similarity_threshold
            pose_similarity_threshold = eg_lm.pose_similarity_threshold
        except AttributeError:
            path_similarity_threshold = None
            pose_similarity_threshold = None

        fg_lm = cast("FeatureGraphLM", lm)

        try:
            possible_poses = fg_lm.possible_poses
        except AttributeError:
            possible_poses = None

        if path_similarity_threshold is None and pose_similarity_threshold is None:
            try:
                path_similarity_threshold = fg_lm.path_similarity_threshold
                pose_similarity_threshold = fg_lm.pose_similarity_threshold
            except AttributeError:
                pass

        # Then, construct the event

        # TODO: maybe use numpydantic to keep ndarrays instead of .tolist()?

        location = lm.buffer.current_location()
        if isinstance(location, np.ndarray):
            location = location.tolist()

        detected_pose = lm.detected_pose
        if isinstance(lm.detected_pose, np.ndarray):
            detected_pose = lm.detected_pose.tolist()

        # CAVEAT: Dict / Mapping variables are live references owned by the LMs, subject
        # to having their content altered by LMs at every step. This could be
        # problematic for threaded telemetry subscribers. The ".copy()" TODOs indicate
        # those variables. Dict copying would isolate the contents for thread safety,
        # but at the cost of some performance overhead for non-threaded subscribers.
        # Perhaps `ThreadedTelemetrySubscriber` could handle copying dict members in a
        # generic manner as needed.
        return cls(
            lm_id=lm.learning_module_id,
            lm_type=type(lm),
            recognition_state=LMRecognitionState(
                terminal_state=TerminalState.from_graph_lm(lm),
                detected_object=lm.detected_object,
            ),
            step_state=LMStepState(
                is_on_object=(lm.buffer.on_object and lm.buffer.on_object[0]),
                matching_steps=lm.buffer.get_num_matching_steps(),
                location=location,
                buffer_stats=lm.buffer.stats,  # TODO .copy()?
                stepwise_target_object=lm.stepwise_target_object,
                graph_id_to_target=lm.graph_id_to_target,  # TODO .copy()?
                target_to_graph_id=lm.target_to_graph_id,  # TODO .copy()?
                detected_pose=detected_pose,
                symmetry_evidence=lm.symmetry_evidence,
                current_mlh=current_mlh,
                possible_matches=lm.get_possible_matches(),
                possible_paths=lm.possible_paths,  # TODO .copy()?
                possible_poses=possible_poses,  # TODO .copy()?
                path_similarity_threshold=path_similarity_threshold,
                pose_similarity_threshold=pose_similarity_threshold,
            ),
        )


class EpisodeTelemetryModel(BaseModel):
    """Base model mixin for telemetry schemas with fields for episode state.

    All instances carry universal context baggage (emitter, timestamp, episode, step,
    mode).
    """

    mode: ExperimentMode
    """Current experiment mode."""

    episode: int
    """Current episode number."""

    step: int
    """Current episode step number."""


class EpisodeStatsTelemetry(EpisodeTelemetryModel):
    """Event produced by `EpisodeTelemetryHandler`.

    Contains aggregated overall episode statistics of the experiment.
    """

    stats: Mapping
    """Stats data dictionary to pass along to subscribers, e.g. Wandb connector."""
