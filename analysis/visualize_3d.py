"""Animate what the potato experiment saw and inferred, step by step.

Adapted from the segmentation project's ``visualition_3d.py``, which was
written against the old 3-SM / 2-LM telemetry. This version reads the current
2-SM / 1-LM setup (see ``conf/experiment/potato.yaml``), whose detailed stats
are laid out as::

    stats = {
        "SM_1": {                        # SalienceSM ("view_finder")
            "raw_observations": [{"rgba": (H, W, 4), ...}, ...],
            "sm_properties":    [{"sm_rotation", "sm_location"}, ...],
            "segmentation_maps": [(H, W) mask or None, ...],
            "regions": [[{"location": (3,), "confidence": float, ...}, ...], ...],
        },
        "attention_system": {            # was per-SM "region" telemetry
            "voxel_size": float,
            "voxel_lifetime": int,
            "voxel_grids": [{"voxels": (V, 3), "age": (V,), "count": (V,)}, ...],
        },
        "LM_0": {"evidences": [...], "lm_processed_steps": [...], ...},
    }

SM_0 (the CameraSM "patch") runs with ``save_raw_obs: false``, so it does not
appear in the stats at all. Every telemetry block is optional, so the figure
adapts to what was recorded.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from detailed_stats import available_episodes, extract_rgba, load_episode_stats

# Voxel coordinates are lower corners; offset to centres for plotting.
VOXEL_CENTRE_OFFSET = 0.5

# The ranking table stops here so its rows stay legible with many objects.
MAX_TABLE_ROWS = 15

# Okabe-Ito palette: distinguishable under the common colour-vision deficiencies.
# Yellow and black are ordered last -- yellow is faint on white, and black is
# what the step cursor uses.
OKABE_ITO = (
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#000000",  # black
)

DEFAULT_EXP_DIR = (
    Path(os.environ.get("MONTY_LOGS", "~/tbp/results/monty")).expanduser()
    / "projects"
    / "monty_runs"
    / "potato"
)


class SMTelemetry:
    """Plot-ready views over one SalienceSM's recorded telemetry.

    Attributes:
        rgbas: ``(num_frames, H, W, 4)`` sensor frames.
        n_frames: How many frames the episode holds.
        segmentation_maps: Per frame, the ``(H, W)`` segmentation mask, or None
            on frames where no segmentation strategy ran.
        region_locations: Per frame, the ``(N, 3)`` locations of the region the
            module proposed (the segmented surface points).
        region_salience: Per frame, the ``(N,)`` salience (goal confidence) of
            those points.
    """

    def __init__(self, stats: dict, sensor_module_id: int | str = 1) -> None:
        """Read one sensor module's telemetry out of loaded episode stats.

        Args:
            stats: Loaded episode stats.
            sensor_module_id: Which sensor module to read. Defaults to SM_1,
                the SalienceSM in the 2-SM potato setup.
        """
        if isinstance(sensor_module_id, int):
            sensor_module_id = f"SM_{sensor_module_id}"
        self.sensor_module_id = sensor_module_id
        sm = stats[sensor_module_id]

        self.rgbas = extract_rgba(stats, sensor_module_id)
        self.n_frames = self.rgbas.shape[0]

        self.segmentation_maps: list[np.ndarray | None] = [
            None if mask is None else np.asarray(mask)
            for mask in sm.get("segmentation_maps", [])
        ]

        self.region_locations: list[np.ndarray] = []
        self.region_salience: list[np.ndarray] = []
        for region in sm.get("regions", []):
            goals = [g for g in region if g.get("location") is not None]
            self.region_locations.append(
                np.array([g["location"] for g in goals], dtype=float).reshape(-1, 3)
            )
            self.region_salience.append(
                np.array([g["confidence"] for g in goals], dtype=float)
            )

    @property
    def has_segmentation(self) -> bool:
        """Whether a segmentation strategy ran during this episode."""
        return any(mask is not None for mask in self.segmentation_maps)

    def overlay(self, frame: int) -> np.ndarray:
        """Return the rgba frame with the segmentation tinted in green.

        Args:
            frame: Which frame to render.

        Returns:
            The frame, blended with its segmentation when one was recorded.
        """
        rgba = self.rgbas[frame].copy()
        mask = (
            self.segmentation_maps[frame]
            if frame < len(self.segmentation_maps)
            else None
        )
        if mask is None or not np.any(mask > 0):
            return rgba
        tint = np.zeros_like(rgba)
        tint[..., 1] = 255
        tint[..., 3] = 128
        active = mask > 0
        rgba[active] = (rgba[active] * 0.6 + tint[active] * 0.4).astype(np.uint8)
        return rgba

    def region_at(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the region locations and salience proposed on one frame.

        Args:
            frame: Which frame to read.

        Returns:
            A ``(locations, salience)`` pair, ``(0, 3)`` and ``(0,)`` when the
            frame proposed no region.
        """
        if frame >= len(self.region_locations):
            return np.empty((0, 3)), np.empty(0)
        return self.region_locations[frame], self.region_salience[frame]

    def bounds_points(self) -> list[np.ndarray]:
        """Return every 3D point set the episode touches, for axis limits.

        Returns:
            Point arrays spanning the proposed regions.
        """
        return [points for points in self.region_locations if len(points)]


class AttentionTelemetry:
    """Plot-ready views over the attention system's recorded voxel grids.

    In the old telemetry the voxel grid rode inside the sensor module's
    ``region`` block; it is now recorded by the attention system itself, one
    grid per Monty step, each voxel carrying its remaining ``age`` and its
    proposal ``count``.

    Attributes:
        voxel_size: Edge length of a voxel in meters, or None when no
            attention telemetry was recorded.
        voxel_lifetime: Steps a voxel survives without being re-proposed.
        feature: Which voxel column colours the grid ("age" or "count").
    """

    def __init__(self, stats: dict, feature: str = "age") -> None:
        """Read the attention system's telemetry out of loaded episode stats.

        Args:
            stats: Loaded episode stats.
            feature: Voxel column to expose for colouring.
        """
        attention = stats.get("attention_system", {})
        self.voxel_size = attention.get("voxel_size")
        self.voxel_lifetime = attention.get("voxel_lifetime")
        self.feature = feature

        self.centres: list[np.ndarray] = []
        self.values: list[np.ndarray] = []
        for grid in attention.get("voxel_grids", []):
            # An empty grid serializes as [], so reshape rather than assume (V, 3).
            indices = np.asarray(grid.get("voxels", []), dtype=int).reshape(-1, 3)
            self.centres.append(
                (indices + VOXEL_CENTRE_OFFSET) * (self.voxel_size or 1.0)
            )
            self.values.append(
                np.asarray(grid.get(feature, []), dtype=float).ravel()
            )

    @property
    def has_grids(self) -> bool:
        """Whether any voxel grid was recorded."""
        return self.voxel_size is not None and len(self.centres) > 0

    def voxels_at(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the voxel centres and colouring values recorded on one step.

        Args:
            step: Which step to read.

        Returns:
            A ``(centres, values)`` pair, both empty when the grid was empty.
        """
        if step >= len(self.centres):
            return np.empty((0, 3)), np.empty(0)
        return self.centres[step], self.values[step]

    def bounds_points(self) -> list[np.ndarray]:
        """Return the voxel centres of every step, for axis limits.

        Returns:
            Point arrays spanning the recorded grids.
        """
        return [centres for centres in self.centres if len(centres)]


class LMEvidence:
    """Plot-ready views over one learning module's evidence record.

    Attributes:
        learning_module_id: The module's name, e.g. ``LM_0``.
        max_evidences: Each object's ``(num_steps,)`` max-evidence trace.
        n_hypotheses: ``(num_steps,)`` hypothesis counts per LM step.
        frame_to_lm_step: Per sensor frame, the index of the latest LM step
            (-1 before the LM first processes).
    """

    def __init__(self, stats: dict, learning_module_id: int | str = 0) -> None:
        """Read one learning module's evidence out of loaded episode stats.

        The LM only steps when it processes, so it has fewer steps than the
        sensor has frames. ``lm_processed_steps`` holds one flag per sensor
        frame; its cumulative sum maps every frame to the most recent LM step.

        Args:
            stats: Loaded episode stats.
            learning_module_id: Which learning module to read. Defaults to
                LM_0, the only module in the 1-LM potato setup.
        """
        if isinstance(learning_module_id, int):
            learning_module_id = f"LM_{learning_module_id}"
        self.learning_module_id = learning_module_id

        lm_dict = stats.get(learning_module_id, {})
        all_evidences = lm_dict.get("evidences") or []

        object_names = set()
        for evidences_t in all_evidences:
            object_names.update(evidences_t.keys())

        n_steps = len(all_evidences)
        self.n_hypotheses = np.zeros(n_steps, dtype=int)
        self.max_evidences = {
            name: np.zeros(n_steps) for name in sorted(object_names)
        }
        for step, evidences_t in enumerate(all_evidences):
            for obj_name, trace in self.max_evidences.items():
                obj_t = evidences_t.get(obj_name, [])
                self.n_hypotheses[step] += len(obj_t)
                trace[step] = max(obj_t) if len(obj_t) else 0.0

        processed = np.asarray(lm_dict.get("lm_processed_steps", []), dtype=bool)
        if len(processed):
            self.frame_to_lm_step = np.cumsum(processed) - 1
        else:
            # No alignment recorded; show the full traces on every frame.
            self.frame_to_lm_step = np.full(n_steps, n_steps - 1)

    @property
    def has_evidence(self) -> bool:
        """Whether any evidence was recorded for this module."""
        return self.n_steps > 0

    @property
    def n_steps(self) -> int:
        """How many steps the LM took."""
        return len(self.n_hypotheses)

    @property
    def burst_steps(self) -> np.ndarray:
        """LM steps at which the hypothesis count grew."""
        return np.where(np.diff(self.n_hypotheses) > 0)[0] + 1

    def step_for_frame(self, frame: int) -> int:
        """Return the latest LM step reached by a sensor frame.

        Args:
            frame: The sensor frame.

        Returns:
            The LM step index, -1 before the LM first processes.
        """
        if frame < len(self.frame_to_lm_step):
            return int(self.frame_to_lm_step[frame])
        return self.n_steps - 1

    def ranking_at_step(self, step: int) -> list[tuple[str, float]]:
        """Return objects ranked by their max evidence at one LM step.

        Args:
            step: The LM step to rank at; clipped to the last step.

        Returns:
            ``(name, max_evidence)`` pairs, strongest first; empty before the LM
            first processes.
        """
        if step < 0 or self.n_steps == 0:
            return []
        step = min(step, self.n_steps - 1)
        pairs = [
            (name, float(trace[step]))
            for name, trace in self.max_evidences.items()
        ]
        pairs.sort(key=lambda pair: pair[1], reverse=True)
        return pairs

    def objects_by_peak(self) -> list[str]:
        """Return object names ordered by their peak evidence, strongest first.

        Returns:
            The object names, descending by trace maximum.
        """
        return sorted(
            self.max_evidences,
            key=lambda name: self.max_evidences[name].max(),
            reverse=True,
        )

    def value_range(self) -> tuple[float, float]:
        """Return the range spanned by every evidence trace.

        Returns:
            The ``(low, high)`` pair over all objects and steps.
        """
        all_traces = np.array(list(self.max_evidences.values()))
        return float(all_traces.min()), float(all_traces.max())


def _bounds(point_sets: list[np.ndarray]) -> tuple[list, list, list]:
    """Compute equal-aspect axis limits spanning every given point set.

    Returns:
        An ``(xlim, ylim, zlim)`` triple.
    """
    populated = [p for p in point_sets if p is not None and len(p)]
    if not populated:
        return [-1, 1], [-1, 1], [-1, 1]

    points = np.vstack(populated)
    low, high = points.min(axis=0), points.max(axis=0)
    padding = 0.1 * (high - low)
    low, high = low - padding, high + padding

    centre = (low + high) / 2
    half = (high - low).max() / 2 or 1.0
    return (
        [centre[0] - half, centre[0] + half],
        [centre[1] - half, centre[1] + half],
        [centre[2] - half, centre[2] + half],
    )


def create_segmentation_animation(
    exp_dir: Path,
    episode: int = 0,
    sensor_module_id: str | int = 1,
    fps: int = 2,
    interval: int = 500,
    marker_size: int = 5,
    voxel_feature: str = "age",
    lm_id: int | str = 0,
) -> Path:
    """Animate the segmentation, region, and voxel-grid telemetry.

    Args:
        exp_dir: Experiment directory.
        episode: Episode to visualize.
        sensor_module_id: Sensor module to read; SM_1 is the SalienceSM.
        fps: Frames per second of the saved gif.
        interval: Milliseconds between animation frames.
        marker_size: Scatter marker size in the 3D panels.
        voxel_feature: Voxel column to colour the grid by ("age" or "count").
        lm_id: Learning module whose evidence to plot, when it was recorded.

    Returns:
        Path to the saved gif.
    """
    stats = load_episode_stats(exp_dir, episode=episode)
    sm = SMTelemetry(stats, sensor_module_id)
    attention = AttentionTelemetry(stats, feature=voxel_feature)
    lm = LMEvidence(stats, lm_id)

    n_frames = sm.n_frames
    if not attention.has_grids:
        print(
            "No attention telemetry in this episode - skipping the voxel grid "
            "panel."
        )

    xlim, ylim, zlim = _bounds(sm.bounds_points() + attention.bounds_points())

    # Row 0: frame, the proposed region, and (when recorded) the voxel grid.
    # Row 1 (with evidence): the evidence traces below the frame, and the ranked
    # evidence table beside them.
    ncols = 2 + int(attention.has_grids)
    nrows = 1 + int(lm.has_evidence)
    fig = plt.figure(figsize=(6.5 * ncols, 5.5 * nrows))
    # The 3D panels' colorbars sit between panels; give them room so their labels
    # do not collide with the next panel's y-axis.
    grid = fig.add_gridspec(nrows, ncols, wspace=0.35, hspace=0.3)
    ax_image = fig.add_subplot(grid[0, 0])
    ax_region = fig.add_subplot(grid[0, 1], projection="3d")
    ax_voxels = (
        fig.add_subplot(grid[0, 2], projection="3d")
        if attention.has_grids
        else None
    )
    ax_evidence = fig.add_subplot(grid[1, 0]) if lm.has_evidence else None
    ax_table = fig.add_subplot(grid[1, 1]) if lm.has_evidence else None

    title = "Salience Visualization"
    if attention.voxel_size is not None:
        title += f" (voxel_size={attention.voxel_size})"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    def style_3d(ax, label: str) -> None:
        """Apply shared 3D styling. Re-applied after every ax.clear()."""
        ax.set_title(label)
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
    ax_image.set_title("Frame")
    style_3d(ax_region, "Proposed Region (3D)")
    if ax_voxels is not None:
        style_3d(ax_voxels, "Voxel Grid (3D)")

    image = ax_image.imshow(sm.overlay(0))

    # Empty scatters exist only to anchor the colorbars; update_frame clears and
    # redraws the 3D panels each step.
    region_anchor = ax_region.scatter(
        [], [], [], c=[], cmap="plasma", s=marker_size, alpha=0.8, vmin=0, vmax=1
    )
    region_bar = plt.colorbar(region_anchor, ax=ax_region, fraction=0.046, pad=0.08)
    region_bar.set_label("Salience", rotation=270, labelpad=15)

    # Age is bounded by the voxel lifetime, so its scale is known up front.
    # Count grows without bound, so its scale widens as larger values turn up;
    # it only ever widens, so a colour means the same thing from one frame to
    # the next.
    if voxel_feature == "age" and attention.voxel_lifetime is not None:
        voxel_scale = {"vmin": 0.0, "vmax": float(attention.voxel_lifetime)}
    else:
        voxel_scale = {"vmin": 0.0, "vmax": 1.0}

    def voxel_limits(values: np.ndarray) -> tuple[float, float]:
        """Widen the voxel colour scale to admit ``values``.

        Returns:
            The (vmin, vmax) to draw this frame with.
        """
        if len(values):
            low, high = float(np.min(values)), float(np.max(values))
            voxel_scale["vmin"] = min(voxel_scale["vmin"], low)
            voxel_scale["vmax"] = max(voxel_scale["vmax"], high)
        return voxel_scale["vmin"], voxel_scale["vmax"]

    voxel_bar = None
    if ax_voxels is not None:
        all_values = [v for v in attention.values if len(v)]
        vmin, vmax = voxel_limits(
            np.concatenate(all_values) if all_values else np.empty(0)
        )
        voxel_anchor = ax_voxels.scatter(
            [], [], [], c=[], cmap="viridis", s=marker_size, alpha=0.8,
            vmin=vmin, vmax=vmax,
        )
        voxel_bar = plt.colorbar(voxel_anchor, ax=ax_voxels, fraction=0.046, pad=0.08)
        voxel_bar.set_label(attention.feature, rotation=270, labelpad=15)

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

    # The evidence panel reveals each object's max-evidence trace up to the LM
    # step reached by the current frame. Lines are created once and their data
    # extended per frame; axis limits are fixed up front so nothing rescales.
    evidence_lines: dict[str, object] = {}
    burst_lines: list[tuple[int, object]] = []
    evidence_cursor = None
    if ax_evidence is not None:
        low, high = lm.value_range()
        pad = 0.05 * (high - low or 1.0)
        ax_evidence.set_xlim(0, max(lm.n_steps - 1, 1))
        ax_evidence.set_ylim(low - pad, high + pad)
        ax_evidence.set_xlabel("LM step")
        ax_evidence.set_ylabel("Max evidence")

        # With many objects a full legend is unreadable; label the strongest --
        # one colour-blind-safe colour each -- and draw the rest as thin grey
        # context lines.
        by_peak = lm.objects_by_peak()
        labelled = set(by_peak[: len(OKABE_ITO)])
        for rank, name in enumerate(by_peak):
            if name in labelled:
                (line,) = ax_evidence.plot(
                    [], [], label=name, linewidth=1.5, color=OKABE_ITO[rank]
                )
            else:
                (line,) = ax_evidence.plot(
                    [], [], color="0.8", linewidth=0.8, zorder=1
                )
            evidence_lines[name] = line
        ax_evidence.legend(fontsize=7, loc="upper left")

        # Steps where the hypothesis count grew; revealed as they are reached.
        for burst_step in lm.burst_steps:
            vline = ax_evidence.axvline(
                burst_step, color="gray", linestyle="--", alpha=0.5, visible=False
            )
            burst_lines.append((int(burst_step), vline))

        evidence_cursor = ax_evidence.axvline(
            0, color="black", linewidth=0.8, alpha=0.6, visible=False
        )

    if ax_table is not None:
        ax_table.axis("off")

    def update_frame(step: int):
        image.set_data(sm.overlay(step))
        label = "Frame" if not sm.has_segmentation else "Frame + Segmentation"
        ax_image.set_title(f"{label} (Step {step}/{n_frames - 1})")

        ax_region.clear()
        style_3d(ax_region, "")
        locations, salience = sm.region_at(step)
        if len(locations):
            ax_region.scatter(
                locations[:, 0],
                locations[:, 1],
                locations[:, 2],
                c=salience,
                cmap="plasma",
                s=marker_size,
                alpha=0.8,
                vmin=0,
                vmax=1,
            )
            ax_region.set_title(
                f"Proposed Region ({len(locations)} points, "
                f"Step {step}/{n_frames - 1})"
            )
        else:
            ax_region.set_title(f"Proposed Region (none, Step {step}/{n_frames - 1})")
            ax_region.text2D(0.5, 0.5, "No region", transform=ax_region.transAxes,
                             ha="center")

        if ax_voxels is not None:
            ax_voxels.clear()
            style_3d(ax_voxels, "")
            centres, values = attention.voxels_at(step)
            if len(centres):
                vmin, vmax = voxel_limits(values)
                ax_voxels.scatter(
                    centres[:, 0],
                    centres[:, 1],
                    centres[:, 2],
                    c=values,
                    cmap="viridis",
                    s=marker_size,
                    alpha=0.8,
                    vmin=vmin,
                    vmax=vmax,
                )
                if voxel_bar is not None:
                    # Keep the bar honest if the scale just widened.
                    voxel_bar.mappable.set_clim(vmin, vmax)
                ax_voxels.set_title(
                    f"Voxel Grid ({len(centres)} voxels, "
                    f"Step {step}/{n_frames - 1})"
                )
            else:
                ax_voxels.set_title(f"Voxel Grid (none, Step {step}/{n_frames - 1})")
                ax_voxels.text2D(
                    0.5, 0.5, "No voxels", transform=ax_voxels.transAxes, ha="center"
                )

        if ax_evidence is not None:
            # The LM lags the sensor: reveal traces up to the LM step this frame
            # has reached (-1 before the LM first processes).
            lm_step = lm.step_for_frame(step)
            revealed = np.arange(lm_step + 1)
            for name, line in evidence_lines.items():
                line.set_data(revealed, lm.max_evidences[name][: lm_step + 1])
            for burst_step, vline in burst_lines:
                vline.set_visible(lm_step >= burst_step)
            # An axvline is a two-point Line2D, so both x values must be set.
            evidence_cursor.set_xdata([max(lm_step, 0)] * 2)
            evidence_cursor.set_visible(lm_step >= 0)
            ax_evidence.set_title(
                f"{lm.learning_module_id} Max Evidence "
                f"(LM step {max(lm_step, 0)}/{lm.n_steps - 1})"
            )

        if ax_table is not None:
            lm_step = lm.step_for_frame(step)
            ax_table.clear()
            ax_table.axis("off")
            ax_table.set_title(f"{lm.learning_module_id} Evidence Ranking")
            ranking = lm.ranking_at_step(lm_step)[:MAX_TABLE_ROWS]
            if ranking:
                cells = [[name, f"{value:.2f}"] for name, value in ranking]
                table = ax_table.table(
                    cellText=cells,
                    colLabels=("Object", "Max evidence"),
                    loc="center",
                    cellLoc="left",
                    colWidths=(0.6, 0.4),
                )
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.3)
                # Tint each object's row to match its trace colour; a soft
                # background keeps black text readable for every palette entry.
                for row, (name, _) in enumerate(ranking, start=1):
                    colour = evidence_lines[name].get_color()
                    table[row, 0].set_facecolor(mcolors.to_rgba(colour, alpha=0.3))
            else:
                ax_table.text(
                    0.5, 0.5, "No evidence yet",
                    transform=ax_table.transAxes, ha="center",
                )

        return [image]

    anim = FuncAnimation(fig, update_frame, frames=n_frames, interval=interval,
                         blit=True)

    visualizations_dir = exp_dir / "visualizations"
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    gif_path = visualizations_dir / f"segmentation_{episode}.gif"
    anim.save(gif_path, writer=PillowWriter(fps=fps))

    print(f"Animation saved to: {gif_path}")
    plt.close()

    return gif_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animate an experiment's segmentation and attention telemetry."
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
    parser.add_argument(
        "--voxel-feature",
        choices=("age", "count"),
        default="age",
        help="Voxel column to colour the grid by (default: age)",
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
        create_segmentation_animation(
            args.exp_dir,
            episode=episode,
            voxel_feature=args.voxel_feature,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
