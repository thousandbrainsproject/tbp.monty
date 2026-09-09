# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Animate a frame with its segmentation beside every LM's max evidence.

Panels: the sensor's frame with its segmentation mask blended over it
(top-left), then one max-evidence plot per learning module. LM steps from
which a module had recognized its object (its individual terminal state
was a match) are shaded green on its plot once the animation reaches
them.

Run from the repo root, e.g. ``python -m analysis.scripts.visualize_evidence
debug_3lm --sensor-module SM_3``; the run is a directory or a name under
``RESULTS_DIR``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from analysis import views
from analysis.panels import (
    EvidenceTraces,
    Image,
    Panel,
)
from analysis.scripts.animation import save_animation
from analysis.telemetry import EpisodeTelemetry

if TYPE_CHECKING:
    import os

# Evidence plots per figure row; the frame takes the first cell.
N_COLUMNS = 2


def create_evidence_animation(
    run_dir: os.PathLike,
    sensor_module: str,
    episode: int = 0,
    fps: int = 2,
    fmt: str = "gif",
) -> Path:
    """Animate the segmentation and every learning module's max evidence.

    Args:
        run_dir: Experiment directory.
        sensor_module: Name of the sensor module whose frame and
            segmentation to show.
        episode: Episode number to visualize.
        fps: Frames per second of the saved animation.
        fmt: Output format, "gif" (PillowWriter) or "mp4" (FFMpegWriter,
            needs ffmpeg on the PATH).

    Returns:
        Path to the saved animation,
        ``<run_dir>/visualizations/evidence_<episode>.<fmt>``.
    """
    run_dir = Path(run_dir)
    ep = EpisodeTelemetry.load(run_dir, episode)

    # What to show: the frame with its segmentation, and every learning
    # module's evidence with the steps it held a recognized object.
    frames = views.rgba_stream(ep, sensor_module)
    segmentation = views.segmentation_overlay_stream(ep, sensor_module)
    lms = ep.learning_modules

    # How to show it: the frame top-left, evidence plots filling the grid.
    n_panels = 1 + len(lms)
    nrows = max(1, math.ceil(n_panels / N_COLUMNS))
    fig = plt.figure(figsize=(7.5 * N_COLUMNS, 5 * nrows))
    grid = fig.add_gridspec(nrows, N_COLUMNS, wspace=0.25, hspace=0.35)
    fig.suptitle(run_dir.name, fontsize=14, fontweight="bold")
    panels: list[Panel] = [
        Image(
            frames,
            fig.add_subplot(grid[0, 0]),
            title="Frame + Segmentation",
            overlay=segmentation,
            fixation_marker=True,
        )
    ]
    for i, lm in enumerate(lms, start=1):
        evidences = views.evidence_stream(ep, lm)
        panels.append(
            EvidenceTraces(
                evidences.map(views.max_evidence),
                evidences.map(views.hypothesis_count),
                fig.add_subplot(grid[divmod(i, N_COLUMNS)]),
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
        run_dir / "visualizations" / f"evidence_{episode}.{fmt}",
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
    create_evidence_animation(
        args.run,
        sensor_module=args.sensor_module,
        episode=args.episode,
        fps=args.fps,
        fmt=args.format,
    )
