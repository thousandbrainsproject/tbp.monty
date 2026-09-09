# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Summary figure comparing a compositional and a monolithic eval run.

Three panels side by side, always describing the higher-level LM (LM_2 by
default, never the low-level modules):

* Accuracy: one stacked bar per run. The solid segment counts episodes the
  LM converged to the right object (``primary_performance == "correct"``);
  the lighter segment on top counts episodes where the right object was
  merely the most likely hypothesis when time ran out (``"correct_mlh"``).
* Matching Steps: a violin with the per-episode Monty matching steps.
* Rotation Error: a violin with the per-episode rotation error, in degrees,
  over the episodes whose object was right (converged or MLH).

Run from the repo root, e.g.::

    python -m analysis.scripts.summarize_comp_vs_monolithic

which compares ``base_infer_objects_with_stickers_comp_models_mujoco``
against ``base_infer_objects_with_stickers_monolithic_models_mujoco``; pass
other run names or directories to compare other runs. The figure goes to
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

# wandb's run colors, as in compare_comp_monolithic.py.
COLORS = {"Monolithic": "#f737bd", "Compositional": "#00a0df"}
DEFAULT_FIGURE_DIR = Path("~/tbp/projects/comp_benefits_figures/figures").expanduser()
DEFAULT_RUNS = {
    "Monolithic": "base_infer_objects_with_stickers_monolithic_models_mujoco",
    "Compositional": "base_infer_objects_with_stickers_comp_models_mujoco",
}


@dataclass(frozen=True)
class RunSummary:
    """Per-episode outcomes of one eval run, for one learning module.

    Attributes:
        percent_converged: Episodes ended converged on the right object
            (``"correct"``), in percent.
        percent_correct_mlh: Episodes where the right object was only the
            most likely hypothesis at time-out (``"correct_mlh"``), in percent.
        matching_steps: Monty matching steps, one entry per episode.
        rotation_errors: Rotation error in degrees, one entry per episode
            whose object was right (converged or MLH).
        n_episodes: How many episodes the run holds.
    """

    percent_converged: float
    percent_correct_mlh: float
    matching_steps: np.ndarray
    rotation_errors: np.ndarray
    n_episodes: int

    @property
    def percent_correct(self) -> float:
        """Percent of episodes that ended on the right object, however."""
        return self.percent_converged + self.percent_correct_mlh


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
    performance = rows["primary_performance"]
    correct = performance.str.startswith("correct")
    rotation_errors = pd.to_numeric(rows.loc[correct, "rotation_error"], errors="coerce")
    return RunSummary(
        percent_converged=float((performance == "correct").mean() * 100),
        percent_correct_mlh=float((performance == "correct_mlh").mean() * 100),
        matching_steps=rows["monty_matching_steps"].to_numpy(dtype=float),
        rotation_errors=rotation_errors.dropna().to_numpy(),
        n_episodes=len(rows),
    )


def _lighten(color: str, amount: float = 0.6) -> np.ndarray:
    """Blend a color toward white.

    Args:
        color: Any matplotlib color.
        amount: 0 keeps the color, 1 is white.

    Returns:
        The blended RGB triple.
    """
    rgb = np.array(plt.matplotlib.colors.to_rgb(color))
    return rgb + (1.0 - rgb) * amount


def draw_accuracy(ax: Axes, summaries: dict[str, RunSummary]) -> None:
    """The stacked accuracy bars: converged below, correct-MLH on top.

    Args:
        ax: The axes to draw on.
        summaries: Summary per run name, in plotting order left to right.
    """
    x = np.arange(len(summaries))
    for xi, (name, summary) in zip(x, summaries.items()):
        color = COLORS.get(name, "gray")
        ax.bar(xi, summary.percent_converged, width=0.7, color=color)
        ax.bar(
            xi,
            summary.percent_correct_mlh,
            bottom=summary.percent_converged,
            width=0.7,
            color=_lighten(color),
        )
    # Legend colors are neutral grays; the run colors mark the runs, not
    # the converged/MLH split.
    ax.legend(
        handles=[
            plt.matplotlib.patches.Patch(color="dimgray", label="Converged (correct)"),
            plt.matplotlib.patches.Patch(
                color="lightgray", label="MLH only (correct_mlh)"
            ),
        ],
        fontsize=8,
        loc="upper right",
    )
    ax.set_xticks(x, summaries.keys())
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Accuracy", fontweight="bold")


def draw_violin(
    ax: Axes,
    title: str,
    values: dict[str, np.ndarray],
    ylabel: str,
    seed: int = 42,
) -> None:
    """One panel: a violin per run with the episode values scattered on top.

    Args:
        ax: The axes to draw on.
        title: The panel title.
        values: The episodes' values per run name, left to right.
        ylabel: The y axis label.
        seed: Seed for the horizontal jitter of the scatter.
    """
    rng = np.random.default_rng(seed)
    x = np.arange(len(values))
    for xi, (name, episode_values) in zip(x, values.items()):
        color = COLORS.get(name, "gray")
        if len(episode_values) == 0:
            ax.text(xi, 0, "no episodes", ha="center", fontsize=8, color="dimgray")
            continue
        if len(np.unique(episode_values)) > 1:
            violin = ax.violinplot(
                episode_values, positions=[xi], widths=0.8, showextrema=False
            )
            for body in violin["bodies"]:
                body.set_facecolor(_lighten(color, 0.35))
                body.set_alpha(0.5)
        jitter = rng.uniform(-0.08, 0.08, size=len(episode_values))
        ax.scatter(xi + jitter, episode_values, s=12, color=color, zorder=3, alpha=0.5)
    ax.set_xticks(x, values.keys())
    ax.set_xlim(-0.6, len(values) - 0.4)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")


def create_summary_figure(
    runs: dict[str, os.PathLike],
    learning_module: str = "LM_2",
    output: os.PathLike | None = None,
) -> Path:
    """Plot the three summary panels for the runs side by side.

    Args:
        runs: Run directory per display name (``"Compositional"``,
            ``"Monolithic"``).
        learning_module: The higher-level module the panels describe.
        output: Where to save the figure; defaults to
            ``DEFAULT_FIGURE_DIR / summary_<first run name>.png``.

    Returns:
        Path to the saved figure.
    """
    summaries = {
        name: summarize_run(path, learning_module) for name, path in runs.items()
    }
    for name, summary in summaries.items():
        print(
            f"{name} ({learning_module}, {summary.n_episodes} episodes): "
            f"{summary.percent_correct:.1f}% correct "
            f"({summary.percent_converged:.1f}% converged, "
            f"{summary.percent_correct_mlh:.1f}% MLH only), "
            f"matching steps {summary.matching_steps.mean():.1f} mean, "
            f"rotation error {np.mean(summary.rotation_errors):.1f}° mean "
            f"(over {len(summary.rotation_errors)} correct episodes)"
        )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    draw_accuracy(axes[0], summaries)
    draw_violin(
        axes[1],
        "Matching Steps",
        {n: s.matching_steps for n, s in summaries.items()},
        ylabel="Monty Matching Steps",
    )
    draw_violin(
        axes[2],
        "Rotation Error",
        {n: s.rotation_errors for n, s in summaries.items()},
        ylabel="Rotation Error (degrees)",
    )
    episodes = ", ".join(f"{n}: {s.n_episodes} episodes" for n, s in summaries.items())
    fig.suptitle(
        " vs ".join(Path(p).name for p in runs.values())
        + f"\n{learning_module} ({episodes})",
        fontsize=9,
    )
    fig.tight_layout()
    if output is None:
        first = Path(next(iter(runs.values()))).name
        output = DEFAULT_FIGURE_DIR / f"summary_{first}.png"
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
    parser.add_argument(
        "compositional",
        type=run_directory,
        nargs="?",
        default=run_directory(DEFAULT_RUNS["Compositional"]),
        help="run dir or name",
    )
    parser.add_argument(
        "monolithic",
        type=run_directory,
        nargs="?",
        default=run_directory(DEFAULT_RUNS["Monolithic"]),
        help="run dir or name",
    )
    parser.add_argument("--learning-module", default="LM_2", help="the HL LM")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    create_summary_figure(
        {"Compositional": args.compositional, "Monolithic": args.monolithic},
        learning_module=args.learning_module,
        output=args.output,
    )
