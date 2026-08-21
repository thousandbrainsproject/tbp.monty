# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Animate what a SalienceSM saw and what the AttentionSystem retained, per step.

The detailed stats hold:

    stats["SM_2"] = {
        "raw_observations":  [...],           # via SnapshotTelemetry
        "sm_properties":     [...],
        "segmentation_maps": [(H, W), ...],   # via SalienceSMTelemetry
        "regions":           [[aw, ...]],     # aw: {"location", "weight"}
    }
    stats["attention_system"] = {
        "voxel_size": float,                  # absent in runs before 2026-08-05
        "voxel_grids": [
            {"voxels": (V, 3), "weight": (V,)},
        ],
    }

LM evidence blocks are unchanged from the old format.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from detailed_stats import load_episode_stats
from matplotlib import colors as mcolors
from plots import (
    OKABE_ITO,
    add_fixation_marker,
    add_image_panel,
    add_scatter_colorbar_3d,
    animation_writer,
    equal_aspect_bounds_3d,
    save_animation,
    style_3d,
)

# The ranking table stops here so its rows stay legible with many objects.
MAX_TABLE_ROWS = 12


class SMTelemetry:
    """Plot-ready views over one sensor module's recorded telemetry.

    Attributes:
        rgbas: ``(num_frames, H, W, 4)`` sensor frames.
        n_frames: How many frames the episode holds.
        segmentation_maps: ``(num_frames, H, W)`` segmentation masks, or None
            when none were recorded.
        region_locations: Per frame, the ``(N, 3)`` region point locations.
        region_weights: Per frame, the ``(N,)`` attention weights of those
            points.
    """

    def __init__(self, stats: dict, sensor_module_id: int | str) -> None:
        if isinstance(sensor_module_id, int):
            sensor_module_id = f"SM_{sensor_module_id}"
        self.sensor_module_id = sensor_module_id

        self.rgbas = stats.rgba(sensor_module_id)
        self.n_frames = self.rgbas.shape[0]
        self.segmentation_maps = stats.segmentation_maps(sensor_module_id)
        self.region_locations, self.region_weights = stats.sm_regions(sensor_module_id)

    @property
    def has_segmentation(self) -> bool:
        return self.segmentation_maps is not None

    def overlay(self, frame: int) -> np.ndarray:
        """Return the rgba frame with the segmentation tinted in green."""
        rgba = self.rgbas[frame].copy()
        if self.segmentation_maps is None:
            return rgba
        mask = self.segmentation_maps[frame] > 0
        if not np.any(mask):
            return rgba
        tint = np.zeros_like(rgba)
        tint[..., 1] = 255
        tint[..., 3] = 128
        rgba[mask] = (rgba[mask] * 0.6 + tint[mask] * 0.4).astype(np.uint8)
        return rgba

    def region_at(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the region locations and weights proposed on one frame."""
        if frame >= len(self.region_locations):
            return np.empty((0, 3)), np.empty(0)
        return self.region_locations[frame], self.region_weights[frame]


class AttentionTelemetry:
    """Plot-ready views over the attention system's recorded voxel grids.

    Attributes:
        voxel_size: Edge length of a voxel, in world units.
        centres: Per frame, the ``(V, 3)`` world-frame voxel centres.
        weights: Per frame, the ``(V,)`` voxel weights.
        weight_limits: ``(vmin, vmax)`` spanning every recorded weight and
            zero, so one colour scale serves every frame.
    """

    def __init__(self, stats: dict) -> None:
        self.voxel_size, self.centres, self.weights = stats.voxel_grids()
        recorded = np.concatenate(self.weights) if self.weights else np.empty(0)
        self.weight_limits = (
            float(min(0.0, recorded.min(initial=0.0))),
            float(max(1.0, recorded.max(initial=0.0))),
        )

    @property
    def has_grids(self) -> bool:
        return bool(self.centres)

    def voxels_at(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the voxel centres and weights recorded on one frame."""
        if frame >= len(self.centres):
            return np.empty((0, 3)), np.empty(0)
        return self.centres[frame], self.weights[frame]


class LMEvidence:
    """Plot-ready views over one learning module's evidence record."""

    def __init__(self, stats: dict, learning_module_id: int | str) -> None:
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
        self.max_evidences = {name: np.zeros(n_steps) for name in sorted(object_names)}
        for step, evidences_t in enumerate(all_evidences):
            for obj_name, trace in self.max_evidences.items():
                obj_t = evidences_t.get(obj_name, [])
                self.n_hypotheses[step] += len(obj_t)
                trace[step] = max(obj_t) if len(obj_t) else 0.0

        processed = np.asarray(lm_dict.get("lm_processed_steps", []), dtype=bool)
        if len(processed):
            self.frame_to_lm_step = np.cumsum(processed) - 1
        else:
            self.frame_to_lm_step = np.full(n_steps, n_steps - 1)

    @property
    def has_evidence(self) -> bool:
        return self.n_steps > 0

    @property
    def n_steps(self) -> int:
        return len(self.n_hypotheses)

    @property
    def burst_steps(self) -> np.ndarray:
        return np.where(np.diff(self.n_hypotheses) > 0)[0] + 1

    def step_for_frame(self, frame: int) -> int:
        if frame < len(self.frame_to_lm_step):
            return int(self.frame_to_lm_step[frame])
        return self.n_steps - 1

    def ranking_at_step(self, step: int) -> list[tuple[str, float]]:
        if step < 0 or self.n_steps == 0:
            return []
        step = min(step, self.n_steps - 1)
        pairs = [
            (name, float(trace[step])) for name, trace in self.max_evidences.items()
        ]
        pairs.sort(key=lambda pair: pair[1], reverse=True)
        return pairs

    def objects_by_peak(self) -> list[str]:
        return sorted(
            self.max_evidences,
            key=lambda name: self.max_evidences[name].max(),
            reverse=True,
        )

    def value_range(self) -> tuple[float, float]:
        all_traces = np.array(list(self.max_evidences.values()))
        return float(all_traces.min()), float(all_traces.max())


def create_attention_animation(  # noqa: C901 -- one long figure assembly
    exp_dir: Path,
    episode: int = 0,
    sensor_module_id: str | int = "SM_2",
    fps: int = 2,
    interval: int = 500,
    marker_size: int = 5,
    voxel_marker_size: int | None = None,
    lm_id: int | str = 0,
) -> Path:
    """Animate the segmentation, region proposal, and attention voxel grid.

    Args:
        exp_dir: Experiment directory.
        episode: Episode to visualize.
        sensor_module_id: Sensor module to read.
        fps: Frames per second of the saved gif.
        interval: Milliseconds between animation frames.
        marker_size: Scatter marker size for region points.
        voxel_marker_size: Scatter marker size for attention voxels. Defaults to
            a size inversely proportional to the number of voxels, so coarse
            grids read as blocks and fine grids as point clouds.
        lm_id: Learning module whose evidence to plot, when it was recorded.

    Returns:
        Path to the saved gif.
    """
    stats: dict = load_episode_stats(exp_dir, episode=episode)
    sm = SMTelemetry(stats, sensor_module_id)
    attention = AttentionTelemetry(stats)
    lm = LMEvidence(stats, lm_id)

    if voxel_marker_size is None:
        max_voxels = max((len(c) for c in attention.centres), default=1)
        voxel_marker_size = max(5, min(150, 6000 // max(max_voxels, 1)))

    n_frames = sm.n_frames

    bounds_sets = list(sm.region_locations) + list(attention.centres)
    xlim, ylim, zlim = equal_aspect_bounds_3d(bounds_sets)

    # Row 0: frame, region proposal, and the attention voxel grid.
    # Row 1 (with evidence): evidence traces and the ranked evidence table.
    ncols = 2 + int(attention.has_grids)
    nrows = 1 + int(lm.has_evidence)
    fig = plt.figure(figsize=(6.5 * ncols, 5.5 * nrows))
    grid = fig.add_gridspec(nrows, ncols, wspace=0.35, hspace=0.3)
    ax_image = fig.add_subplot(grid[0, 0])
    ax_region = fig.add_subplot(grid[0, 1], projection="3d")
    ax_voxels = (
        fig.add_subplot(grid[0, 2], projection="3d") if attention.has_grids else None
    )
    ax_evidence = fig.add_subplot(grid[1, 0]) if lm.has_evidence else None
    ax_table = fig.add_subplot(grid[1, 1]) if lm.has_evidence else None

    fig.suptitle(
        f"Attention Visualization (voxel_size={attention.voxel_size})",
        fontsize=14,
        fontweight="bold",
    )

    style_3d(ax_region, xlim, ylim, zlim, "Region Proposal (3D)")
    if ax_voxels is not None:
        style_3d(ax_voxels, xlim, ylim, zlim, "Attention Voxel Grid (3D)")

    image = add_image_panel(ax_image, sm.overlay(0), "Frame")

    add_scatter_colorbar_3d(
        ax_region,
        "weight",
        cmap="plasma",
        vmin=attention.weight_limits[0],
        vmax=attention.weight_limits[1],
        marker_size=marker_size,
    )
    if ax_voxels is not None:
        add_scatter_colorbar_3d(
            ax_voxels,
            "weight",
            cmap="viridis",
            vmin=attention.weight_limits[0],
            vmax=attention.weight_limits[1],
            marker_size=voxel_marker_size,
        )

    add_fixation_marker(
        ax_image, sm.rgbas[0].shape, size=3, edgecolor="black", linewidth=1
    )

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

        # Label the strongest objects, one colour-blind-safe colour each; draw
        # the rest as thin grey context lines.
        by_peak = lm.objects_by_peak()
        labelled = set(by_peak[: len(OKABE_ITO)])
        for rank, name in enumerate(by_peak):
            if name in labelled:
                (line,) = ax_evidence.plot(
                    [], [], label=name, linewidth=1.5, color=OKABE_ITO[rank]
                )
            else:
                (line,) = ax_evidence.plot([], [], color="0.8", linewidth=0.8, zorder=1)
            evidence_lines[name] = line
        ax_evidence.legend(fontsize=7, loc="upper left")

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
        style_3d(ax_region, xlim, ylim, zlim)
        locations, weights = sm.region_at(step)
        if len(locations):
            ax_region.scatter(
                locations[:, 0],
                locations[:, 1],
                locations[:, 2],
                c=weights,
                cmap="plasma",
                s=marker_size,
                alpha=0.8,
                vmin=attention.weight_limits[0],
                vmax=attention.weight_limits[1],
            )
            ax_region.set_title(
                f"Region Proposal ({len(locations)} points, Step {step}/{n_frames - 1})"
            )
        else:
            ax_region.set_title(f"Region Proposal (none, Step {step}/{n_frames - 1})")
            ax_region.text2D(
                0.5, 0.5, "No region", transform=ax_region.transAxes, ha="center"
            )

        if ax_voxels is not None:
            ax_voxels.clear()
            style_3d(ax_voxels, xlim, ylim, zlim)
            centres, weights = attention.voxels_at(step)
            if len(centres):
                ax_voxels.scatter(
                    centres[:, 0],
                    centres[:, 1],
                    centres[:, 2],
                    c=weights,
                    cmap="viridis",
                    s=voxel_marker_size,
                    alpha=0.8,
                    vmin=attention.weight_limits[0],
                    vmax=attention.weight_limits[1],
                )
                ax_voxels.set_title(
                    f"Attention Voxel Grid ({len(centres)} voxels, "
                    f"Step {step}/{n_frames - 1})"
                )
            else:
                ax_voxels.set_title(
                    f"Attention Voxel Grid (none, Step {step}/{n_frames - 1})"
                )
                ax_voxels.text2D(
                    0.5, 0.5, "No voxels", transform=ax_voxels.transAxes, ha="center"
                )

        if ax_evidence is not None:
            lm_step = lm.step_for_frame(step)
            revealed = np.arange(lm_step + 1)
            for name, line in evidence_lines.items():
                line.set_data(revealed, lm.max_evidences[name][: lm_step + 1])
            for burst_step, vline in burst_lines:
                vline.set_visible(lm_step >= burst_step)
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
                table.auto_set_font_size(False)  # noqa: FBT003
                table.set_fontsize(9)
                table.scale(1, 1.3)
                for row, (name, _) in enumerate(ranking, start=1):
                    colour = evidence_lines[name].get_color()
                    table[row, 0].set_facecolor(mcolors.to_rgba(colour, alpha=0.3))
            else:
                ax_table.text(
                    0.5,
                    0.5,
                    "No evidence yet",
                    transform=ax_table.transAxes,
                    ha="center",
                )

        return [image]

    return save_animation(
        fig,
        update_frame,
        n_frames,
        exp_dir / "visualizations" / f"attention_{episode}.gif",
        animation_writer("gif", fps),
        interval=interval,
    )


if __name__ == "__main__":
    exp_dir = Path.home() / "tbp/results/comp_benefits_figures/debug_3lm"
    create_attention_animation(exp_dir, episode=0, sensor_module_id="SM_3", lm_id=1)
