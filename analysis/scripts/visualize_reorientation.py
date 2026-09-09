# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Animate a patch sensor module's face-on re-orientation over an episode.

Two panels per step, from the telemetry a ``CameraSM`` with a
``FaceOnReorientation`` goal generator records (its ``gsg`` block; the
module needs ``save_raw_obs`` on) and the view finder's frames:

1. The view finder's frame. On the step a face-on goal fires, the goal is
   drawn into it -- the sensed surface point and the normal axis it wants
   to look along (an arrow toward where the camera should go, clipped to
   the frame) -- and the frames right after the jump are tagged.
2. The view angle between the camera's viewing direction and the patch's
   surface normal, revealed as steps pass, with the generator's smoothed
   angle, its threshold, the steps on which the agent was repositioned
   (face-on jumps in pink, other jumps in black), and a cursor.

Run from the repo root, e.g. ``python -m analysis.scripts.visualize_reorientation
randrot_sm0_reorient_telemetry --episode 6 --format mp4``; the run is a
directory or a name under ``RESULTS_DIR``. ``--all-episodes`` renders every
episode the run holds. The animations go to ``<run_dir>/visualizations/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from analysis import views
from analysis.panels import Image, Panel
from analysis.scripts.animation import save_animation
from analysis.scripts.visualize_view_angle import (
    GOAL_COLOR,
    SURFACE_COLOR,
    clip_to_frame,
    jump_steps,
    project,
)
from analysis.telemetry import EpisodeTelemetry, available_episodes
from tbp.monty.frameworks.loggers.npz_handler import materialize

if TYPE_CHECKING:
    import os
    from collections.abc import Callable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# How many steps after a face-on jump the frame stays tagged.
TAG_STEPS = 4


class GoalInFrame(Panel):
    """The face-on goal drawn into the view finder's frame when it fires.

    On a goal's step: the sensed surface point and the normal axis toward
    the goal camera position, projected through that frame's camera. For
    ``TAG_STEPS`` steps after: a "face-on jump" tag.
    """

    def __init__(
        self,
        ax: Axes,
        goals: list[dict],
        cam_to_world: views.Stream[np.ndarray],
        shape: tuple[int, int],
    ) -> None:
        """Build the panel's artists on the frame's axes.

        Args:
            ax: The axes the frame is drawn on.
            goals: The generator's goal records (``step``,
                ``surface_location``, ``location``, ``view_angle``).
            cam_to_world: The view finder's camera-to-world transforms, one
                per frame.
            shape: The frame's ``(H, W)``.
        """
        self._goals = {int(goal["step"]): goal for goal in goals}
        self._cam_to_world = cam_to_world
        self._shape = shape
        (self._surface,) = ax.plot(
            [], [], "o", color=SURFACE_COLOR, markersize=9, markeredgecolor="black"
        )
        (self._axis,) = ax.plot([], [], "-", color=GOAL_COLOR, linewidth=2.5)
        (self._target,) = ax.plot(
            [], [], "*", color=GOAL_COLOR, markersize=14, markeredgecolor="black"
        )
        self._tag = ax.text(
            4,
            shape[0] - 6,
            "",
            color="white",
            fontsize=9,
            va="bottom",
            bbox=dict(facecolor=GOAL_COLOR, alpha=0.8, pad=3),
            visible=False,
        )

    def update(self, step: int) -> None:
        fired = self._goals.get(step)
        for artist in (self._surface, self._axis, self._target):
            artist.set_visible(fired is not None)
        if fired is not None:
            tick = self._cam_to_world.latest(step)
            cam_to_world = np.asarray(tick.value, dtype=float)
            surface = np.asarray(fired["surface_location"], dtype=float)
            target = np.asarray(fired["location"], dtype=float)
            (su, sv), s_ok = project(surface, cam_to_world, self._shape)
            (gu, gv), g_ok = project(target, cam_to_world, self._shape)
            self._surface.set_data([su], [sv])
            self._surface.set_visible(s_ok)
            if s_ok and g_ok:
                (eu, ev), _ = clip_to_frame((su, sv), (gu, gv), self._shape)
                self._axis.set_data([su, eu], [sv, ev])
                self._target.set_data([eu], [ev])
            else:
                self._axis.set_visible(False)
                self._target.set_data([su], [sv])
                self._target.set_visible(s_ok)
        recent = [s for s in self._goals if s <= step <= s + TAG_STEPS]
        self._tag.set_visible(bool(recent))
        if recent:
            goal = self._goals[max(recent)]
            self._tag.set_text(
                f"face-on jump @ {max(recent)}: {goal['view_angle']:.0f}° -> face-on"
            )


class ViewAngleTrace(Panel):
    """The view angle over the episode, revealed as steps pass."""

    def __init__(
        self,
        ax: Axes,
        angles: np.ndarray,
        smoothed: np.ndarray,
        jumps: list[int],
        face_on_steps: set[int],
        max_view_angle: float,
        title: str,
    ) -> None:
        """Build the panel's artists.

        Args:
            ax: The 2D axes to draw on.
            angles: The measured view angle per step (NaN where none).
            smoothed: The generator's smoothed angle per step (NaN where none).
            jumps: Steps on which the agent was repositioned.
            face_on_steps: The steps on which face-on goals fired.
            max_view_angle: The generator's threshold, drawn as a line.
            title: Base title; ``update`` appends the step counter.
        """
        self._ax = ax
        self._title = title
        self._steps = np.arange(len(angles))
        self._angles = angles
        self._smoothed = smoothed
        (self._angle_line,) = ax.plot(
            [], [], ".-", color="#4C72B0", markersize=4, label="view angle"
        )
        (self._smoothed_line,) = ax.plot(
            [], [], "-", color="#DD8452", linewidth=2, label="smoothed"
        )
        ax.axhline(max_view_angle, color="gray", linewidth=0.8, alpha=0.6)
        self._jumps: list[tuple[int, Line2D]] = [
            (
                jump,
                ax.axvline(
                    jump,
                    color=GOAL_COLOR if jump in face_on_steps else "black",
                    linestyle="--",
                    linewidth=1.2,
                    visible=False,
                ),
            )
            for jump in sorted(jumps)
        ]
        self._cursor = ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_xlim(0, max(len(angles) - 1, 1))
        ax.set_ylim(0, 95)
        ax.set_xlabel("episode step")
        ax.set_ylabel("angle to surface normal (deg)")
        # Proxies for the jump lines, which start hidden.
        handles = [self._angle_line, self._smoothed_line] + [
            Line2D([], [], color=color, linestyle="--", linewidth=1.2, label=label)
            for color, label in ((GOAL_COLOR, "face-on jump"), ("black", "other jump"))
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=8)

    def update(self, step: int) -> None:
        shown = self._steps <= step
        self._angle_line.set_data(self._steps[shown], self._angles[shown])
        self._smoothed_line.set_data(self._steps[shown], self._smoothed[shown])
        for jump, line in self._jumps:
            line.set_visible(step >= jump)
        self._cursor.set_xdata([step, step])
        self._ax.set_title(f"{self._title} (Step {step}/{len(self._steps) - 1})")


def build_reorientation_figure(
    ep: EpisodeTelemetry,
    patch_module: str,
    sensor_module: str,
    max_view_angle: float,
    title: str,
) -> tuple[Figure, Callable[[int], list], int]:
    """Assemble the figure and its per-step update for one episode.

    Args:
        ep: The episode's telemetry.
        patch_module: The sensor module whose goal generator telemetry to read.
        sensor_module: The view finder module whose frames to show.
        max_view_angle: The generator's threshold, drawn on the trace.
        title: The figure's title.

    Returns:
        The figure, the function that redraws it for a step, and the step
        count.
    """
    reorientation = materialize(ep.blocks[patch_module]["gsg"])
    goals = list(reorientation["goals"])
    angles = np.asarray(reorientation["view_angle"], dtype=float)
    smoothed = np.asarray(reorientation["smoothed_view_angle"], dtype=float)
    frames = views.rgba_stream(ep, sensor_module)
    cam_to_world = views.Stream(
        ep(f"{sensor_module}/raw_observations/*/cam_to_world"), frames.steps
    )
    jumps = jump_steps(ep)
    face_on_steps = {int(goal["step"]) for goal in goals}

    fig = plt.figure(figsize=(14, 5.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.4], wspace=0.25)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    frame_ax = fig.add_subplot(grid[0, 0])
    shape = next(iter(frames)).shape[:2]
    panels: list[Panel] = [
        Image(frames, frame_ax, title="View finder", fixation_marker=True),
        GoalInFrame(frame_ax, goals, cam_to_world, shape),
        ViewAngleTrace(
            fig.add_subplot(grid[0, 1]),
            angles,
            smoothed,
            jumps,
            face_on_steps,
            max_view_angle,
            title=f"{patch_module} view angle",
        ),
    ]

    def update_frame(step: int) -> list:
        for panel in panels:
            panel.update(step)
        return []

    return fig, update_frame, max(frames.n_steps, len(angles))


def create_reorientation_animation(
    run_dir: os.PathLike,
    patch_module: str = "SM_0",
    sensor_module: str = "SM_3",
    episode: int = 0,
    max_view_angle: float = 45.0,
    fps: int = 5,
    fmt: str = "gif",
) -> Path:
    """Animate the view finder and the patch module's view angle.

    Args:
        run_dir: Experiment directory.
        patch_module: The sensor module whose goal generator telemetry to read.
        sensor_module: The view finder module whose frames to show.
        episode: Episode number to visualize.
        max_view_angle: The generator's threshold, drawn on the trace.
        fps: Frames per second of the saved animation.
        fmt: Output format, "gif" (PillowWriter) or "mp4" (FFMpegWriter,
            needs ffmpeg on the PATH).

    Returns:
        Path to the saved animation,
        ``<run_dir>/visualizations/reorientation_<episode>.<fmt>``.
    """
    run_dir = Path(run_dir)
    ep = EpisodeTelemetry.load(run_dir, episode)
    fig, update_frame, n_steps = build_reorientation_figure(
        ep, patch_module, sensor_module, max_view_angle, f"{run_dir.name} ep {episode}"
    )
    return save_animation(
        fig,
        update_frame,
        n_steps,
        run_dir / "visualizations" / f"reorientation_{episode}.{fmt}",
        fps=fps,
        fmt=fmt,
    )


if __name__ == "__main__":
    import argparse

    from analysis.cli import run_directory

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "run",
        type=run_directory,
        help="experiment output directory, or a run name under RESULTS_DIR",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--all-episodes", action="store_true", help="render every episode in the run"
    )
    parser.add_argument("--patch-module", default="SM_0", help="module name")
    parser.add_argument("--sensor-module", default="SM_3", help="module name")
    parser.add_argument("--max-view-angle", type=float, default=45.0)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument(
        "--format",
        choices=("gif", "mp4"),
        default="gif",
        help="output format; mp4 needs ffmpeg on the PATH",
    )
    args = parser.parse_args()
    episodes = available_episodes(args.run) if args.all_episodes else [args.episode]
    for episode in episodes:
        create_reorientation_animation(
            args.run,
            patch_module=args.patch_module,
            sensor_module=args.sensor_module,
            episode=episode,
            max_view_angle=args.max_view_angle,
            fps=args.fps,
            fmt=args.format,
        )
