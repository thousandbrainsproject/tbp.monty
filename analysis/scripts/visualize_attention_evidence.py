# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Animate the attention voxel grid beside the lower-level LMs' evidence.

Panels, per monty step: the sensor's frame with its segmentation mask
blended over it and the attention system's voxel grid (3D, colored by
weight) on the first row; one max-evidence plot per chosen learning module
on the second. LM steps from which a module had recognized its object are
shaded green on its plot once the animation reaches them.

Run from the repo root, e.g. ``python -m
analysis.scripts.visualize_attention_evidence debug_inhibition_flip
--learning-modules LM_0 LM_1``; the run is a directory or a name under
``RESULTS_DIR``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from analysis import views
from analysis.panels import (
    EvidenceTraces,
    Image,
    Panel,
    Scatter3D,
    shared_scale,
)
from analysis.scripts.animation import save_animation
from analysis.telemetry import EpisodeTelemetry

if TYPE_CHECKING:
    import os
    from collections.abc import Sequence


def create_attention_evidence_animation(
    run_dir: os.PathLike,
    sensor_module: str,
    learning_modules: Sequence[str],
    episode: int = 0,
    fps: int = 2,
    voxel_marker_size: int | None = None,
    fmt: str = "gif",
) -> Path:
    """Animate the segmentation, the attention voxel grid, and LM evidence.

    Args:
        run_dir: Experiment directory.
        sensor_module: Name of the sensor module whose frame and
            segmentation to show, e.g. ``"SM_3"``.
        learning_modules: Names of the learning modules whose evidence to
            plot, one panel each.
        episode: Episode number to visualize.
        fps: Frames per second of the saved animation.
        voxel_marker_size: Scatter marker size for attention voxels. Defaults
            to a size inversely proportional to the number of voxels, so
            coarse grids read as blocks and fine grids as point clouds.
        fmt: Output format, "gif" (PillowWriter) or "mp4" (FFMpegWriter,
            needs ffmpeg on the PATH).

    Returns:
        Path to the saved animation,
        ``<run_dir>/visualizations/attention_evidence_<episode>.<fmt>``.
    """
    run_dir = Path(run_dir)
    ep = EpisodeTelemetry.load(run_dir, episode)

    # What to show: the views.
    frames = views.rgba_stream(ep, sensor_module)
    segmentation = views.segmentation_overlay_stream(ep, sensor_module)
    voxels = views.attention_grid_stream(ep)

    bounds, vlim = shared_scale(voxels, color="weight")
    if voxel_marker_size is None:
        max_voxels = max((len(grid) for grid in voxels), default=0)
        voxel_marker_size = max(5, min(150, 6000 // max(max_voxels, 1)))

    # How to show it: row 0 the frame and the voxel grid; row 1 one
    # evidence plot per learning module.
    ncols = max(2, len(learning_modules))
    fig = plt.figure(figsize=(7 * ncols, 5.5 * 2))
    grid = fig.add_gridspec(2, ncols, wspace=0.3, hspace=0.3)
    fig.suptitle(run_dir.name, fontsize=14, fontweight="bold")
    panels: list[Panel] = [
        Image(
            frames,
            fig.add_subplot(grid[0, 0]),
            title="Frame + Segmentation",
            overlay=segmentation,
            fixation_marker=True,
        ),
        Scatter3D(
            voxels,
            fig.add_subplot(grid[0, 1], projection="3d"),
            color="weight",
            title="Attention Voxel Grid",
            cmap="viridis",
            vlim=vlim,
            bounds=bounds,
            marker_size=voxel_marker_size,
            empty_text="No voxels",
        ),
    ]
    for column, lm in enumerate(learning_modules):
        evidences = views.evidence_stream(ep, lm)
        panels.append(
            EvidenceTraces(
                evidences.map(views.max_evidence),
                evidences.map(views.hypothesis_count),
                fig.add_subplot(grid[1, column]),
                title=f"{lm} Max Evidence",
                recognized=views.recognized_steps(ep, lm),
            )
        )

    def update_frame(step: int) -> list:
        for panel in panels:
            panel.update(step)
        return []

    return save_animation(
        fig,
        update_frame,
        frames.n_steps,
        run_dir / "visualizations" / f"attention_evidence_{episode}.{fmt}",
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
    parser.add_argument(
        "--learning-modules",
        nargs="+",
        default=["LM_0", "LM_1"],
        help="module names, one evidence panel each",
    )
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument(
        "--format",
        choices=("gif", "mp4"),
        default="gif",
        help="output format; mp4 needs ffmpeg on the PATH",
    )
    args = parser.parse_args()
    create_attention_evidence_animation(
        args.run,
        sensor_module=args.sensor_module,
        learning_modules=args.learning_modules,
        episode=args.episode,
        fps=args.fps,
        fmt=args.format,
    )
