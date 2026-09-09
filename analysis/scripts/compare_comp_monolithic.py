# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Bar charts comparing a compositional and a monolithic eval run.

Three stacked panels of horizontal bars, one bar per run, in the style of
the wandb summary charts: the higher-level LM's percent correct (episodes
it ended with the right object, at convergence or as its most likely
hypothesis after timing out), the steps to convergence (mean Monty matching
steps per episode), and the average rotation error over the episodes it got
right. All three are read from each run's ``eval_stats.csv`` the way
``run_parallel``'s overall stats compute them.

Run from the repo root, e.g. ``python -m analysis.scripts.compare_comp_monolithic
one_rot_cylinder_comp_models_mujoco one_rot_cylinder_monolithic_models_mujoco``;
runs are directories or names under ``RESULTS_DIR``. The figure goes to
``~/tbp/projects/comp_benefits_figures/figures/`` unless ``--output`` says
otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import os

    from matplotlib.axes import Axes

# wandb's run colors, as in the PR the chart mimics.
COLORS = {"Compositional": "#7EC3DC", "Monolithic": "#E08BDC"}
DEFAULT_FIGURE_DIR = Path("~/tbp/projects/comp_benefits_figures/figures").expanduser()


@dataclass(frozen=True)
class RunSummary:
    """The three summary numbers of one eval run.

    Attributes:
        percent_correct: Episodes the LM ended on the right object, in percent.
        steps_to_convergence: Mean Monty matching steps per episode.
        rotation_error: Mean rotation error, in degrees, over correct episodes
            (NaN when none was correct).
        n_episodes: How many episodes the run holds.
    """

    percent_correct: float
    steps_to_convergence: float
    rotation_error: float
    n_episodes: int


def summarize_run(run_dir: os.PathLike, learning_module: str = "LM_2") -> RunSummary:
    """Summarize a run's ``eval_stats.csv`` for one learning module.

    Args:
        run_dir: Experiment directory holding ``eval_stats.csv``.
        learning_module: The module whose rows to read, e.g. ``"LM_2"``.

    Returns:
        The run's summary.

    Raises:
        ValueError: If the CSV holds no rows for the module.
    """
    stats = pd.read_csv(Path(run_dir) / "eval_stats.csv")
    rows = stats[stats["lm_id"] == learning_module]
    if rows.empty:
        raise ValueError(f"{run_dir} has no eval rows for {learning_module}")
    correct = rows["primary_performance"].str.startswith("correct")
    rotation_errors = pd.to_numeric(
        rows.loc[correct, "rotation_error"], errors="coerce"
    )
    return RunSummary(
        percent_correct=float(correct.mean() * 100),
        steps_to_convergence=float(rows["monty_matching_steps"].mean()),
        rotation_error=float(rotation_errors.mean()) if correct.any() else np.nan,
        n_episodes=len(rows),
    )


def draw_bars(ax: Axes, title: str, values: dict[str, float], unit: str = "") -> None:
    """One panel: a horizontal bar per run, labelled above the bar.

    Args:
        ax: The axes to draw on.
        title: The panel title.
        values: Bar length per run name, in plotting order top to bottom.
        unit: Suffix for the value printed at the end of each bar.
    """
    names = list(values)
    y = np.arange(len(names))[::-1]
    heights = [0.0 if np.isnan(v) else v for v in values.values()]
    ax.barh(y, heights, height=0.7, color=[COLORS.get(n, "gray") for n in names])
    for yi, name, value in zip(y, names, values.values()):
        ax.text(0, yi + 0.42, name, fontsize=8, color="dimgray", va="bottom")
        label = "n/a" if np.isnan(value) else f"{value:.1f}{unit}"
        ax.text(
            max(heights) * 0.01, yi, f" {label}", fontsize=9, va="center", color="black"
        )
    ax.set_yticks([])
    ax.set_ylim(-0.6, len(names) - 0.4 + 0.5)
    ax.set_xlim(0, max(*heights, 1e-9) * 1.05)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(axis="x", color="#eeeeee")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, colors="dimgray")


def create_comparison_figure(
    runs: dict[str, os.PathLike],
    learning_module: str = "LM_2",
    output: os.PathLike | None = None,
) -> Path:
    """Plot the three summary charts for the runs side by side.

    Args:
        runs: Run directory per display name (``"Compositional"``,
            ``"Monolithic"``).
        learning_module: The higher-level module the charts describe.
        output: Where to save the figure; defaults to
            ``DEFAULT_FIGURE_DIR / comp_vs_monolithic_<first run name>.png``.

    Returns:
        Path to the saved figure.
    """
    summaries = {
        name: summarize_run(path, learning_module) for name, path in runs.items()
    }
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 9))
    draw_bars(
        axes[0],
        f"HL-LM ({learning_module}) Percent Correct",
        {n: s.percent_correct for n, s in summaries.items()},
        unit="%",
    )
    draw_bars(
        axes[1],
        "Steps to Convergence",
        {n: s.steps_to_convergence for n, s in summaries.items()},
    )
    draw_bars(
        axes[2],
        "overall/avg_rotation_error",
        {n: s.rotation_error for n, s in summaries.items()},
        unit="°",
    )
    episodes = ", ".join(f"{n}: {s.n_episodes} episodes" for n, s in summaries.items())
    fig.suptitle(
        " vs ".join(Path(p).name for p in runs.values()) + f"\n({episodes})",
        fontsize=9,
    )
    fig.tight_layout()
    if output is None:
        first = Path(next(iter(runs.values()))).name
        output = DEFAULT_FIGURE_DIR / f"comp_vs_monolithic_{first}.png"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")
    return output


if __name__ == "__main__":
    import argparse

    from analysis.cli import run_directory

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("compositional", type=run_directory, help="run dir or name")
    parser.add_argument("monolithic", type=run_directory, help="run dir or name")
    parser.add_argument("--learning-module", default="LM_2", help="the HL LM")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    create_comparison_figure(
        {"Compositional": args.compositional, "Monolithic": args.monolithic},
        learning_module=args.learning_module,
        output=args.output,
    )
