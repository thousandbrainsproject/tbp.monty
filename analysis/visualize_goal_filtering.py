"""Animate the goals entering and leaving the AttentionSystem's filter.

Each step, the attention system receives every goal the sensor and learning
modules proposed and returns only those inside its voxel grid (see
``AttentionSystem.step``). Both sides ride in the detailed stats under::

    stats["attention_system"] = {
        ...,
        "pre_filter_goals":  [[{"location": (3,), ...}, ...], ...],  # per step
        "post_filter_goals": [[{"location": (3,), ...}, ...], ...],
    }

The figure pairs the sensor view (with its segmentation tinted green) with a
3D scatter of the pre-filter goals in red and the surviving post-filter goals
in black on top.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from detailed_stats import available_episodes, load_episode_stats
from visualize_3d import DEFAULT_EXP_DIR, SMTelemetry, _bounds

PRE_COLOR = "red"
POST_COLOR = "black"


class GoalFilteringTelemetry:
    """Plot-ready views over the attention system's recorded goal filtering.

    Attributes:
        pre: Per step, the ``(N, 3)`` locations of the goals handed to the
            attention system.
        post: Per step, the ``(M, 3)`` locations of the goals that survived
            the voxel grid filter.
    """

    def __init__(self, stats: dict) -> None:
        """Read the goal filtering telemetry out of loaded episode stats.

        Goals without a location (which bypass the filter) have nothing to
        scatter, so they are dropped here.

        Args:
            stats: Loaded episode stats.
        """
        attention = stats.get("attention_system", {})
        self.pre = self._locations(attention.get("pre_filter_goals", []))
        self.post = self._locations(attention.get("post_filter_goals", []))

    @staticmethod
    def _locations(per_step_goals: list[list[dict]]) -> list[np.ndarray]:
        return [
            np.array(
                [g["location"] for g in goals if g.get("location") is not None],
                dtype=float,
            ).reshape(-1, 3)
            for goals in per_step_goals
        ]

    @property
    def has_goals(self) -> bool:
        """Whether any goal filtering was recorded."""
        return any(len(points) for points in self.pre)

    def at(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the pre- and post-filter goal locations of one step.

        Args:
            step: Which step to read.

        Returns:
            A ``(pre, post)`` pair of ``(N, 3)`` arrays, empty past the record.
        """
        pre = self.pre[step] if step < len(self.pre) else np.empty((0, 3))
        post = self.post[step] if step < len(self.post) else np.empty((0, 3))
        return pre, post

    def bounds_points(self) -> list[np.ndarray]:
        """Return every goal location the episode touches, for axis limits.

        Returns:
            Point arrays spanning the pre-filter goals (a superset of post).
        """
        return [points for points in self.pre if len(points)]


def create_goal_filtering_animation(
    exp_dir: Path,
    episode: int = 0,
    sensor_module_id: str | int = 1,
    fps: int = 2,
    interval: int = 500,
    marker_size: int = 5,
) -> Path:
    """Animate the sensor view beside the filtered and unfiltered goals.

    Args:
        exp_dir: Experiment directory.
        episode: Episode to visualize.
        sensor_module_id: Sensor module to read; SM_1 is the SalienceSM.
        fps: Frames per second of the saved gif.
        interval: Milliseconds between animation frames.
        marker_size: Scatter marker size in the 3D panel.

    Returns:
        Path to the saved gif.
    """
    stats = load_episode_stats(exp_dir, episode=episode)
    sm = SMTelemetry(stats, sensor_module_id)
    filtering = GoalFilteringTelemetry(stats)

    if not filtering.has_goals:
        print(
            "No goal filtering telemetry in this episode - re-run the "
            "experiment with the updated AttentionSystemTelemetry."
        )

    n_frames = sm.n_frames
    xlim, ylim, zlim = _bounds(filtering.bounds_points())

    fig = plt.figure(figsize=(13, 5.5))
    grid = fig.add_gridspec(1, 2, wspace=0.25)
    ax_image = fig.add_subplot(grid[0, 0])
    ax_goals = fig.add_subplot(grid[0, 1], projection="3d")

    fig.suptitle("Attention Goal Filtering", fontsize=14, fontweight="bold")

    def style_3d(ax) -> None:
        """Apply shared 3D styling. Re-applied after every ax.clear()."""
        # Post-filter goals sit at exactly the same locations as their
        # pre-filter counterparts, and Axes3D redraws collections sorted by
        # projected depth -- which buries the black points under the larger red
        # set. Draw in call order instead so post lands on top.
        ax.computed_zorder = False
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_box_aspect([1, 1, 1])
        # Top-down: elev=90 looks along Z, azim=-90 orients X/Y conventionally.
        ax.view_init(elev=90, azim=-90)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ax_image.axis("off")
    style_3d(ax_goals)

    image = ax_image.imshow(sm.overlay(0))

    # Mark the fixation: the sensor patch is centred on what it fixates.
    height, width = sm.rgbas[0].shape[:2]
    ax_image.add_patch(
        Rectangle(
            (width // 2 - 1.5, height // 2 - 1.5),
            3,
            3,
            linewidth=1,
            edgecolor="black",
            facecolor="none",
        )
    )

    def update_frame(step: int):
        image.set_data(sm.overlay(step))
        label = "Frame" if not sm.has_segmentation else "Frame + Segmentation"
        ax_image.set_title(f"{label} (Step {step}/{n_frames - 1})")

        ax_goals.clear()
        style_3d(ax_goals)
        pre, post = filtering.at(step)
        # Pre first so the surviving goals draw on top of the red field.
        if len(pre):
            ax_goals.scatter(
                pre[:, 0], pre[:, 1], pre[:, 2],
                c=PRE_COLOR, s=marker_size, alpha=0.5, label="pre-filter",
            )
        if len(post):
            ax_goals.scatter(
                post[:, 0], post[:, 1], post[:, 2],
                c=POST_COLOR, s=marker_size, alpha=0.8, label="post-filter",
            )
        ax_goals.set_title(
            f"Goals ({len(pre)} pre, {len(post)} post, "
            f"Step {step}/{n_frames - 1})"
        )
        if len(pre) or len(post):
            ax_goals.legend(loc="upper right", fontsize=8)
        else:
            ax_goals.text2D(
                0.5, 0.5, "No goals", transform=ax_goals.transAxes, ha="center"
            )

        return [image]

    anim = FuncAnimation(fig, update_frame, frames=n_frames, interval=interval,
                         blit=True)

    visualizations_dir = exp_dir / "visualizations"
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    gif_path = visualizations_dir / f"goal_filtering_{episode}.gif"
    anim.save(gif_path, writer=PillowWriter(fps=fps))

    print(f"Animation saved to: {gif_path}")
    plt.close()

    return gif_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animate the goals entering and leaving the attention filter."
    )
    parser.add_argument(
        "exp_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_EXP_DIR,
        help=f"Experiment output directory (default: {DEFAULT_EXP_DIR})",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Episode to animate (default: every recorded episode)",
    )
    parser.add_argument("--fps", type=int, default=2, help="GIF frames per second")
    args = parser.parse_args()

    episodes = (
        [args.episode]
        if args.episode is not None
        else available_episodes(args.exp_dir)
    )
    if not episodes:
        raise SystemExit(f"No detailed stats found under {args.exp_dir}")
    for episode in episodes:
        create_goal_filtering_animation(args.exp_dir, episode=episode, fps=args.fps)


if __name__ == "__main__":
    main()
