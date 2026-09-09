# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Animate a frame with its segmentation and each module's attention regions.

Panels, per monty step: the sensor's frame with its segmentation mask
blended over it, the attention region the sensor module proposed, and the
region each learning module proposed (every one that recorded regions, or
one named module). The 3D region panels share bounds and one weight scale
so they compare.

Run from the repo root, e.g. ``python -m analysis.scripts.visualize_regions
debug_3lm --sensor-module SM_3``; the run is a directory or a name under
``RESULTS_DIR``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from analysis import views
from analysis.panels import (
    Image,
    Panel,
    Scatter3D,
    shared_scale,
)
from analysis.scripts.animation import save_animation
from analysis.telemetry import EpisodeTelemetry

if TYPE_CHECKING:
    import os


def create_regions_animation(
    run_dir: os.PathLike,
    sensor_module: str,
    episode: int = 0,
    fps: int = 2,
    marker_size: int = 5,
    fmt: str = "gif",
    learning_module: str | None = None,
) -> Path:
    """Animate the segmentation and the attention regions each module emits.

    Args:
        run_dir: Experiment directory.
        sensor_module: Name of the sensor module whose frame, segmentation,
            and regions to show.
        episode: Episode number to visualize.
        fps: Frames per second of the saved animation.
        marker_size: Scatter marker size for region points.
        fmt: Output format, "gif" (PillowWriter) or "mp4" (FFMpegWriter,
            needs ffmpeg on the PATH).
        learning_module: Learning module whose regions to show; every one
            that recorded regions when None.

    Returns:
        Path to the saved animation,
        ``<run_dir>/visualizations/regions_<episode>.<fmt>``.
    """
    run_dir = Path(run_dir)
    ep = EpisodeTelemetry.load(run_dir, episode)

    # What to show: the frame with its segmentation, and the regions of
    # every module that recorded any.
    frames = views.rgba_stream(ep, sensor_module)
    segmentation = views.segmentation_overlay_stream(ep, sensor_module)
    lms = (
        [learning_module]
        if learning_module
        else [
            lm
            for lm in ep.learning_modules
            if len(ep.blocks[lm].get("attention_regions", []))
        ]
    )
    regions = {
        module: views.attention_region_stream(ep, module)
        for module in [sensor_module, *lms]
    }
    bounds, vlim = shared_scale(*regions.values(), color="weight")

    # How to show it: the layout.
    n_panels = 1 + len(regions)
    fig = plt.figure(figsize=(5.5 * n_panels, 5.5))
    grid = fig.add_gridspec(1, n_panels, wspace=0.35)
    fig.suptitle(run_dir.name, fontsize=14, fontweight="bold")
    panels: list[Panel] = [
        Image(
            frames,
            fig.add_subplot(grid[0, 0]),
            title="Frame + Segmentation",
            overlay=segmentation,
            fixation_marker=True,
            reserve_colorbar=True,
        )
    ]
    panels += [
        Scatter3D(
            stream,
            fig.add_subplot(grid[0, i], projection="3d"),
            color="weight",
            title=f"{module} Regions",
            cmap="plasma",
            vlim=vlim,
            bounds=bounds,
            marker_size=marker_size,
            empty_text="No region",
        )
        for i, (module, stream) in enumerate(regions.items(), start=1)
    ]

    def update_frame(step: int) -> list:
        for panel in panels:
            panel.update(step)
        return []

    return save_animation(
        fig,
        update_frame,
        frames.n_steps,
        run_dir / "visualizations" / f"regions_{episode}.{fmt}",
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
    parser.add_argument(
        "--learning-module",
        default=None,
        help="LM whose regions to show (default: every one recording them)",
    )
    args = parser.parse_args()
    create_regions_animation(
        args.run,
        sensor_module=args.sensor_module,
        episode=args.episode,
        fps=args.fps,
        fmt=args.format,
        learning_module=args.learning_module,
    )
