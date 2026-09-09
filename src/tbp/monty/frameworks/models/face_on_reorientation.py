# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import quaternion as qt

from tbp.monty.cmp import Goal
from tbp.monty.frameworks.models.sm_goal_generation import SMGoalGenerator

if TYPE_CHECKING:
    from tbp.monty.cmp import Message
    from tbp.monty.frameworks.models.motor_system_state import SensorState
    from tbp.monty.memento import Memento

logger = logging.getLogger(__name__)

# The camera looks along its own -Z axis.
VIEW_AXIS = np.array([0.0, 0.0, -1.0])


def view_direction(sensor_state: SensorState) -> np.ndarray:
    """The unit vector a sensor looks along, in body coordinates.

    Args:
        sensor_state: The sensor's proprioceptive state.

    Returns:
        The (3,) viewing direction.
    """
    return qt.rotate_vectors(sensor_state.rotation, VIEW_AXIS)


def view_angle(percept: Message, sensor_state: SensorState) -> float | None:
    """The angle, in degrees, between the viewing direction and the surface normal.

    Args:
        percept: The sensor module's percept; its first pose vector is the
            surface normal at the patch.
        sensor_state: The sensor's proprioceptive state.

    Returns:
        The angle, or None when the percept carries no usable normal (off
        the object, or the normal could not be estimated).
    """
    if not percept.process_features_in_lm or not percept.morphological_features:
        return None
    pose_vectors = percept.morphological_features.get("pose_vectors")
    if pose_vectors is None:
        return None
    normal = np.asarray(pose_vectors, dtype=float)[0]
    view = view_direction(sensor_state)
    norms = np.linalg.norm(view) * np.linalg.norm(normal)
    if norms == 0 or not np.isfinite(norms):
        return None
    cosine = -np.dot(view, normal) / norms
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


class FaceOnReorientation(SMGoalGenerator):
    """A sensor module goal generator that asks for a face-on view of its patch.

    Meant for the case where an object's pose leaves the surface under the
    patch at a steep angle to the camera. Each step, the generator measures
    the angle between the direction the sensor looks along and the surface
    normal at the patch (the percept's first pose vector), smoothed over the
    last ``window`` measured steps so a single saccade does not count. When
    the smoothed angle and this step's own angle both exceed
    ``max_view_angle``, it proposes a goal placing the agent on the surface
    normal, at the patch's current distance, looking back at the patch --
    a pose, so the motor system's jump policy re-orients the view; exploration
    then carries on as usual from the new vantage. The goal sits off the
    surface, so pair this with a region proposer that excites it (see
    ``sm_region_proposal.ExciteGoalLocations``), or the attention system
    filters it out.

    Nothing is proposed until ``window`` views have been measured. Goals are
    one-off (proposed on a single step), count as achieved when the first
    view measured after them is within ``max_view_angle``, and a new one is
    not proposed until ``min_steps_between_goals`` sensor steps have passed
    since the last, so re-orienting stays occasional.
    """

    def __init__(
        self,
        max_view_angle: float = 45.0,
        window: int = 5,
        min_steps_between_goals: int = 20,
        standoff: float | None = None,
        confidence: float = 1.0,
    ) -> None:
        """Initialize the generator.

        Args:
            max_view_angle: The largest smoothed angle, in degrees, between the
                viewing direction and the surface normal that still counts as
                face-on. Saccades alone swing the instantaneous angle by tens
                of degrees, so keep this above their range.
            window: How many recent measured steps the view angle is averaged
                over before it is compared with ``max_view_angle``.
            min_steps_between_goals: Sensor steps that must pass after a goal
                before another is proposed.
            standoff: Distance from the surface, in meters, to place the agent
                at; the sensor's current distance from the patch when None,
                keeping the viewing distance.
            confidence: The confidence of the goals; the motor system executes
                the highest-confidence goal when several modules propose one.
        """
        self.max_view_angle = max_view_angle
        self.window = window
        self.min_steps_between_goals = min_steps_between_goals
        self.standoff = standoff
        self.confidence = confidence
        self._init_episode()

    @property
    def view_angle(self) -> float | None:
        """This step's view angle, in degrees, or None when it was not measurable."""
        return self._view_angle

    @property
    def smoothed_view_angle(self) -> float | None:
        """The view angle averaged over the recent measured steps, or None."""
        if not self._recent_angles:
            return None
        return float(np.mean(self._recent_angles))

    def __call__(
        self,
        percept: Message,
        sensor_state: SensorState,
        motor_only_step: bool = False,
    ) -> list[Goal]:
        """Measure this step's view angle and decide whether to re-orient.

        Args:
            percept: The sensor module's percept this step.
            sensor_state: The sensor's proprioceptive state this step.
            motor_only_step: Whether the current step is a motor-only step;
                nothing is measured or proposed on those.

        Returns:
            The face-on goal, when the view is persistently steep; else nothing.
        """
        self._step += 1
        if self._steps_since_goal is not None:
            self._steps_since_goal += 1
        self._view_angle = (
            None if motor_only_step else view_angle(percept, sensor_state)
        )
        if self._view_angle is not None:
            self._recent_angles = (self._recent_angles + [self._view_angle])[
                -self.window :
            ]
        smoothed = self.smoothed_view_angle
        self._angles.append(np.nan if self._view_angle is None else self._view_angle)
        self._smoothed.append(np.nan if smoothed is None else smoothed)

        # The first view measured after a goal decides whether it worked.
        if self._pending is not None and self._view_angle is not None:
            self._pending["achieved"] = bool(self._view_angle <= self.max_view_angle)
            self._pending = None

        if not self._should_reorient(smoothed):
            return []
        goal, record = self._face_on_goal(percept, sensor_state)
        self._pending = record
        self._goal_records.append(record)
        self._steps_since_goal = 0
        return [goal]

    def reset(self) -> None:
        self._init_episode()

    def state_dict(self) -> Memento:
        """The per-step view angles and the goals proposed this episode.

        Returns:
            ``view_angle`` and ``smoothed_view_angle`` with one entry per
            sensor step (NaN where nothing was measured), and ``goals``, one
            record per proposed goal: the ``step`` it was proposed on, the
            goal's ``location`` and ``look_direction``, the
            ``surface_location`` it was computed from, the ``view_angle``
            that triggered it, and whether it was ``achieved`` (None until
            the next measured view).
        """
        return dict(
            view_angle=list(self._angles),
            smoothed_view_angle=list(self._smoothed),
            goals=list(self._goal_records),
        )

    def _init_episode(self) -> None:
        self._view_angle: float | None = None
        self._recent_angles: list[float] = []
        # Sensor steps so far, and since the last goal (None before any).
        self._step = -1
        self._steps_since_goal: int | None = None
        # The record of the last goal, until a measured view settles it.
        self._pending: dict[str, Any] | None = None
        self._angles: list[float] = []
        self._smoothed: list[float] = []
        self._goal_records: list[dict[str, Any]] = []

    def _should_reorient(self, smoothed: float | None) -> bool:
        """Whether to propose a goal: the view is steep and enough steps passed.

        Both the smoothed angle and this step's own angle must exceed
        ``max_view_angle``, so the jump is computed from a steep view, not a
        passing face-on one.

        Returns:
            True when the smoothing window is full, both angles exceed
            ``max_view_angle``, and more than ``min_steps_between_goals``
            steps passed since the last goal (or there was none).
        """
        if smoothed is None or self._view_angle is None:
            return False
        if len(self._recent_angles) < self.window:
            return False
        if smoothed <= self.max_view_angle or self._view_angle <= self.max_view_angle:
            return False
        return (
            self._steps_since_goal is None
            or self._steps_since_goal > self.min_steps_between_goals
        )

    def _face_on_goal(
        self, percept: Message, sensor_state: SensorState
    ) -> tuple[Goal, dict[str, Any]]:
        """An agent pose on the patch's surface normal, looking back at the patch.

        Returns:
            The goal and its telemetry record.
        """
        normal = np.asarray(percept.morphological_features["pose_vectors"], float)[0]
        normal = normal / np.linalg.norm(normal)
        surface_loc = np.asarray(percept.location, dtype=float)
        standoff = self.standoff
        if standoff is None:
            standoff = float(
                np.linalg.norm(
                    surface_loc - np.asarray(sensor_state.position, dtype=float)
                )
            )
        location = surface_loc + normal * standoff
        logger.debug(
            f"Face-on goal from {percept.sender_id}: view angle "
            f"{self._view_angle:.1f} deg, standoff {standoff:.3f} m"
        )
        goal = Goal(
            location=location,
            morphological_features={
                # Look along the negated normal; roll is left unspecified.
                "pose_vectors": np.array(
                    [-normal, [np.nan] * 3, [np.nan] * 3], dtype=float
                ),
                "pose_fully_defined": None,
                "on_object": 1,
            },
            non_morphological_features=None,
            confidence=self.confidence,
            # Meant for the motor system, not for a learning module.
            pass_message=False,
            sender_id=percept.sender_id,
            sender_type="SM",
            process_features_in_lm=False,
            goal_tolerances=None,
        )
        record = dict(
            step=self._step,
            location=location,
            look_direction=-normal,
            surface_location=surface_loc,
            view_angle=self._view_angle,
            achieved=None,
        )
        return goal, record
