# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Animate a SalienceSM's view, its salience map, and the goals sent to motor.

Three panels per step:

1. The sensor module's camera view (raw rgba frame).
2. The salience map computed from that frame.
3. The goals the motor system received (those that passed the attention
   filter), as a 3D scatterplot.

Views (see ``views``): the sensor module's frames and salience maps, its
frame centers for the fixation marker, and the goals the motor system
received. The animation clock is the episode step; each panel shows what
its stream holds for that step.

Run from the repo root, e.g.
``python -m analysis.scripts.visualize_salience debug_3lm --sensor-module SM_3``; the
run is a directory or a name under ``RESULTS_DIR``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from analysis import views
from analysis.panels import Image, Panel, Scatter3D
from analysis.scripts.animation import save_animation
from analysis.telemetry import EpisodeTelemetry

if TYPE_CHECKING:
    import os


def create_salience_animation(
    run_dir: os.PathLike,
    sensor_module: str,
    episode: int = 0,
    fps: int = 2,
    marker_size: int = 5,
    fmt: str = "gif",
    title: str | None = None,
) -> Path:
    """Animate the camera view, salience map, and motor-bound goals.

    Args:
        run_dir: Experiment directory.
        sensor_module: Name of the sensor module whose view and salience map
            to read.
        episode: Episode number to visualize.
        fps: Frames per second of the saved animation.
        marker_size: Scatter marker size for goal points.
        fmt: Output format, "gif" (PillowWriter) or "mp4" (FFMpegWriter,
            needs ffmpeg on the PATH).
        title: Figure title; the run directory's name when None.

    Returns:
        Path to the saved animation,
        ``<run_dir>/visualizations/salience_<episode>.<fmt>``.
    """
    run_dir = Path(run_dir)
    ep = EpisodeTelemetry.load(run_dir, episode)

    # What to show: the views.
    frames = views.rgba_stream(ep, sensor_module)
    salience = views.salience_map_stream(ep, sensor_module)
    goals = views.goal_point_stream(ep)
    fixation = views.fixation_point_stream(ep, sensor_module)

    # How to show it: the layout.
    fig = plt.figure(figsize=(16, 5.5))
    grid = fig.add_gridspec(1, 3, wspace=0.3)
    fig.suptitle(
        run_dir.name if title is None else title, fontsize=14, fontweight="bold"
    )
    panels: list[Panel] = [
        Image(
            frames,
            fig.add_subplot(grid[0, 0]),
            fixation_marker=True,
            reserve_colorbar=True,
        ),
        Image(
            salience,
            fig.add_subplot(grid[0, 1]),
            cmap="inferno",
            colorbar="salience",
            fixation_marker=True,
        ),
        Scatter3D(
            goals,
            fig.add_subplot(grid[0, 2], projection="3d"),
            color="confidence",
            title="Goals",
            cmap="inferno",
            vlim=(0.0, 1.0),
            marker_size=marker_size,
            alpha=0.6,
            empty_text="No goals",
            fixation=fixation,
        ),
    ]

    def update_frame(step: int) -> list:
        for panel in panels:
            panel.update(step)
        return []

    return save_animation(
        fig,
        update_frame,
        frames.n_steps,
        run_dir / "visualizations" / f"salience_{episode}.{fmt}",
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
    parser.add_argument("--sensor-module", default="SM_3", help="module name")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument(
        "--format",
        choices=("gif", "mp4"),
        default="gif",
        help="output format; mp4 needs ffmpeg on the PATH",
    )
    args = parser.parse_args()
    create_salience_animation(
        args.run,
        sensor_module=args.sensor_module,
        episode=args.episode,
        fps=args.fps,
        fmt=args.format,
    )
