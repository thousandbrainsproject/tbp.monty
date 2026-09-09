# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Animate what a SalienceSM saw and what the AttentionSystem retained, per step.

Panels, per monty step: the sensor's frame with its segmentation mask
blended over it, the region the sensor module proposed (3D, colored by
weight), the attention system's voxel grid (3D, colored by weight), and --
when a learning module recorded evidence -- its per-object max-evidence
traces and ranking.

Views (see ``views``): the sensor module's frames and segmentation
overlays, its proposed attention regions, the attention system's voxel
grids, and the learning module's evidence. The animation clock is the episode
step; the LM's evidence, recorded only on steps it processed, is shown as
of the latest such step.

Run from the repo root, e.g. ``python -m analysis.scripts.visualize_attention
debug_3lm --sensor-module SM_3``; the run is a
directory or a name under ``RESULTS_DIR``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from analysis import views
from analysis.panels import (
    EvidenceRanking,
    EvidenceTraces,
    Image,
    Panel,
    Scatter3D,
    shared_scale,
)
from analysis.scripts.animation import save_animation
from analysis.telemetry import EpisodeTelemetry


def create_attention_animation(
    run_dir: Path,
    sensor_module: str,
    episode: int = 0,
    fps: int = 2,
    marker_size: int = 5,
    voxel_marker_size: int | None = None,
    fmt: str = "gif",
    learning_module: str | None = "LM_0",
) -> Path:
    """Animate the segmentation, region proposal, and attention voxel grid.

    Args:
        run_dir: Experiment directory.
        sensor_module: Name of the sensor module to read, e.g. ``"SM_3"``.
        episode: Episode number to visualize.
        fps: Frames per second of the saved animation.
        fmt: Output format, "gif" (PillowWriter) or "mp4" (FFMpegWriter,
            needs ffmpeg on the PATH).
        marker_size: Scatter marker size for region points.
        voxel_marker_size: Scatter marker size for attention voxels. Defaults to
            a size inversely proportional to the number of voxels, so coarse
            grids read as blocks and fine grids as point clouds.
        learning_module: Name of the learning module whose evidence to plot;
            None for no evidence panels.

    Returns:
        Path to the saved animation,
        ``<run_dir>/visualizations/attention_<episode>.<fmt>``.
    """
    run_dir = Path(run_dir)
    ep = EpisodeTelemetry.load(run_dir, episode)

    # What to show: the views.
    frames = views.rgba_stream(ep, sensor_module)
    segmentation = views.segmentation_overlay_stream(ep, sensor_module)
    regions = views.attention_region_stream(ep, sensor_module)
    voxels = views.attention_grid_stream(ep)
    evidences = (
        views.evidence_stream(ep, learning_module) if learning_module else None
    )
    if evidences is not None and not len(evidences):
        evidences = None

    # Regions and voxels share bounds and a weight scale so they compare.
    bounds, vlim = shared_scale(regions, voxels, color="weight")
    if voxel_marker_size is None:
        max_voxels = max((len(grid) for grid in voxels), default=0)
        voxel_marker_size = max(5, min(150, 6000 // max(max_voxels, 1)))

    # How to show it: row 0 frame, region proposal, attention voxel grid;
    # row 1 (with evidence) the evidence traces and the ranked table.
    nrows = 1 + int(evidences is not None)
    fig = plt.figure(figsize=(6.5 * 3, 5.5 * nrows))
    grid = fig.add_gridspec(nrows, 3, wspace=0.35, hspace=0.3)
    fig.suptitle(
        f"Attention Visualization (voxel_size={ep('attention_system/voxel_size')})",
        fontsize=14,
        fontweight="bold",
    )
    panels: list[Panel] = [
        Image(
            frames,
            fig.add_subplot(grid[0, 0]),
            title="Frame + Segmentation",
            overlay=segmentation,
            fixation_marker=True,
        ),
        Scatter3D(
            regions,
            fig.add_subplot(grid[0, 1], projection="3d"),
            color="weight",
            title="Region Proposal",
            cmap="plasma",
            vlim=vlim,
            bounds=bounds,
            marker_size=marker_size,
            empty_text="No region",
        ),
        Scatter3D(
            voxels,
            fig.add_subplot(grid[0, 2], projection="3d"),
            color="weight",
            title="Attention Voxel Grid",
            cmap="viridis",
            vlim=vlim,
            bounds=bounds,
            marker_size=voxel_marker_size,
            empty_text="No voxels",
        ),
    ]
    if evidences is not None:
        maxes = evidences.map(views.max_evidence)
        counts = evidences.map(views.hypothesis_count)
        panels += [
            EvidenceTraces(
                maxes,
                counts,
                fig.add_subplot(grid[1, 0]),
                title=f"{learning_module} Max Evidence",
            ),
            EvidenceRanking(
                maxes,
                fig.add_subplot(grid[1, 1]),
                title=f"{learning_module} Evidence Ranking",
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
        run_dir / "visualizations" / f"attention_{episode}.{fmt}",
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
    parser.add_argument("--learning-module", default="LM_1", help="module name")
    args = parser.parse_args()
    create_attention_animation(
        args.run,
        sensor_module=args.sensor_module,
        episode=args.episode,
        fps=args.fps,
        fmt=args.format,
        learning_module=args.learning_module,
    )
