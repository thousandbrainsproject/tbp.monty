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
3. The goals the attention system passed on to the motor system
   (``post_filter_goals``), as a 3D scatterplot.

The detailed stats hold:

    stats["SM_3"] = {
        "raw_observations": [...],        # via SalienceSMTelemetry
        "salience_maps":    [(H, W), ...],
        ...
    }
    stats["attention_system"] = {
        "post_filter_goals": [[goal, ...], ...],  # goal: {"location", ...}
    }

The SM streams record one entry per sensor step while the attention system
records one entry per monty step; when the counts differ, the goal panel
clamps to its last recorded step.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from detailed_stats import DetailedStats
from plots import (
    GoalsScatter3D,
    RGBFrame,
    SalienceFrame,
    animation_writer,
    save_animation,
)

if TYPE_CHECKING:
    import os


def create_salience_animation(
    exp_dir: os.PathLike,
    episode: int = 0,
    sensor_module_id: str | int = "SM_3",
    fps: int = 2,
    interval: int = 500,
    marker_size: int = 5,
    fmt: str = "gif",
    title: str | None = None,
) -> Path:
    """Animate the camera view, salience map, and motor-bound goals.

    Args:
        exp_dir: Experiment directory.
        episode: Episode to visualize.
        sensor_module_id: Sensor module whose view and salience map to read.
        fps: Frames per second of the saved animation.
        interval: Milliseconds between animation frames.
        marker_size: Scatter marker size for goal points.
        title: Figure title; the run directory's name when None.
        fmt: Output format, "gif" (PillowWriter) or "mp4" (FFMpegWriter,
            needs ffmpeg on the PATH).

    Returns:
        Path to the saved animation.
    """
    writer = animation_writer(fmt, fps)

    exp_dir = Path(exp_dir)
    stats = DetailedStats.load(exp_dir, episode=episode)
    if isinstance(sensor_module_id, int):
        sensor_module_id = f"SM_{sensor_module_id}"

    fig = plt.figure(figsize=(16, 5.5))
    grid = fig.add_gridspec(1, 3, wspace=0.3)

    fig.suptitle(
        Path(exp_dir).name if title is None else title,
        fontsize=14,
        fontweight="bold",
    )

    camera = RGBFrame(
        stats,
        sensor_module_id,
        fig.add_subplot(grid[0, 0]),
        fixation_marker=True,
        invisible_colorbar=True,
    )
    panels = [
        camera,
        SalienceFrame(
            stats,
            sensor_module_id,
            fig.add_subplot(grid[0, 1]),
            fixation_marker=True,
        ),
        GoalsScatter3D(
            stats,
            fig.add_subplot(grid[0, 2], projection="3d"),
            fov_sensor_module_id=sensor_module_id,
            marker_size=marker_size,
        ),
    ]
    n_frames = camera.n_frames

    def update_frame(step: int):
        for panel in panels:
            panel.update(step)
        return []

    return save_animation(
        fig,
        update_frame,
        n_frames,
        exp_dir / "visualizations" / f"salience_{episode}.{fmt}",
        writer,
        interval=interval,
    )


if __name__ == "__main__":
    exp_dir = Path.home() / "tbp/results/comp_benefits_figures/no_attention"
    create_salience_animation(exp_dir, episode=0, sensor_module_id="SM_3")
