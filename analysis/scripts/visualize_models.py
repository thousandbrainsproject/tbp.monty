# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Draw every object graph a pretrained model holds, one figure per LM.

Each learning module gets a grid of 3D scatters, one per learned object,
with every input channel on the same axes and a marker shape per channel.
Nodes learned from a sensor patch are drawn at their stored color (or
colored by a chosen feature); nodes learned from a lower-level LM are drawn
as bold markers colored by the child object they stand for, in colors that
do not occur on the objects themselves. ``--morphology K`` draws every K-th
node's shape vector: the surface normal for 3D channels, the oriented edge
direction for 2D (edge-based) channels.

Run from the repo root, e.g. ``python -m analysis.scripts.visualize_models
--morphology 20`` for the current chain's newest trained stage (see
``analysis.models.CURRENT_CHAIN``), or name a model: a ``model.pt`` path, a
run directory, or a run name under ``MODELS_DIR``. Figures go to
``<run_dir>/visualizations/``; ``--show`` also opens them in a window (drag
to rotate, scroll to zoom).
"""

from __future__ import annotations

import math
from itertools import cycle
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from analysis.models import (
    LearnedObject,
    child_labels,
    default_model,
    load_learned_objects,
    resolve_model_path,
)
from analysis.panels import (
    add_colorbar,
    attach_scroll_zoom,
    equal_aspect_bounds_3d,
    style_3d,
)

if TYPE_CHECKING:
    import os
    from collections.abc import Sequence

    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D

# Color nodes by their stored color rather than by a feature.
STORED_COLOR = "color"
# Bold colors for child objects; deliberately absent from the objects
# themselves so they cannot be confused with stored patch colors.
CHILD_COLORS = ("#FF2DAF", "#E8000B", "#00B140", "#FFC20A", "#7B2CBF", "#00A6D6")
# Marker shapes: every sensor patch a circle, each LM input channel its own.
PATCH_MARKER = "o"
LM_CHANNEL_MARKERS = ("*", "^", "s", "D", "v", "P")
PATCH_MARKER_SIZE = 4
CHILD_MARKER_SIZE = 40


def create_model_figures(
    model: str | os.PathLike | None = None,
    lms: Sequence[str] | None = None,
    objects: Sequence[str] | None = None,
    color_by: str = STORED_COLOR,
    morphology_every: int = 0,
    columns: int = 4,
    elev: float = 20.0,
    azim: float = -20.0,
    dpi: int = 130,
    out_dir: os.PathLike | None = None,
    show: bool = False,
) -> list[Path]:
    """Draw a pretrained model's object graphs, one figure per learning module.

    Args:
        model: See :func:`analysis.models.resolve_model_path`; the newest
            trained stage of the current chain when None.
        lms: The learning modules to draw (``"LM_0"`` ...); all when None.
        objects: The object ids to draw; all when None.
        color_by: ``"color"`` for the nodes' stored color, else the name of a
            stored feature to color sensor-learned nodes by (a vector
            feature's first component).
        morphology_every: Draw the shape vector of every k-th sensor-learned
            node; 0 for none.
        columns: Objects per figure row, at most.
        elev: Elevation of the 3D view, in degrees.
        azim: Azimuth of the 3D view, in degrees.
        dpi: Resolution of the saved figures.
        out_dir: Where to save the figures; ``<run_dir>/visualizations`` by
            default.
        show: Open the figures in a window after saving (drag to rotate,
            scroll to zoom).

    Returns:
        The saved figure paths, one per learning module drawn.
    """
    model_path = default_model() if model is None else resolve_model_path(model)
    run_dir = model_path.parent.parent
    out_dir = Path(out_dir) if out_dir else run_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    learned = load_learned_objects(model_path)

    saved = []
    figures: list[Figure] = []
    for lm in lms or learned:
        by_object = learned[lm]
        if objects:
            by_object = {name: by_object[name] for name in objects if name in by_object}
        if not by_object:
            continue
        fig = create_lm_figure(
            by_object,
            learned,
            title=f"{run_dir.name} -- {lm}",
            color_by=color_by,
            morphology_every=morphology_every,
            columns=columns,
            elev=elev,
            azim=azim,
        )
        path = out_dir / f"learned_objects_{lm}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved {path}")
        saved.append(path)
        figures.append(fig)
    if show:
        for fig in figures:
            attach_scroll_zoom(fig)
        plt.show()
    for fig in figures:
        plt.close(fig)
    return saved


def create_lm_figure(
    by_object: dict[str, dict[str, LearnedObject]],
    learned: dict[str, dict[str, dict[str, LearnedObject]]],
    title: str,
    color_by: str = STORED_COLOR,
    morphology_every: int = 0,
    columns: int = 4,
    elev: float = 20.0,
    azim: float = -20.0,
) -> Figure:
    """Draw one learning module's objects in a grid, with shared legends.

    Args:
        by_object: The module's graphs, per object id then input channel.
        learned: Everything the model learned, to name child objects by.
        title: The figure title.
        color_by: See :func:`create_model_figures`.
        morphology_every: See :func:`create_model_figures`.
        columns: Objects per row, at most.
        elev: Elevation of the 3D view, in degrees.
        azim: Azimuth of the 3D view, in degrees.

    Returns:
        The figure.
    """
    # Shared across the figure: one marker per channel, one color per
    # child-object label, one color scale for a feature.
    channels = sorted(
        {
            channel
            for object_channels in by_object.values()
            for channel in object_channels
        },
        key=lambda name: (not name.startswith("patch"), name),
    )
    markers = channel_markers(channels)
    label_colors = _label_colors(by_object, learned)
    norm = _shared_norm(by_object, color_by)

    n = len(by_object)
    ncols = min(columns, n)
    nrows = math.ceil(n / ncols)
    # Header: the title, then the legends, at fixed heights above the grid.
    height = 4.6 * nrows + 1.4
    fig = plt.figure(figsize=(4.8 * ncols, height))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1 - 0.15 / height, va="top")
    axes = []
    for index, channels_of_object in enumerate(by_object.values()):
        ax = fig.add_subplot(nrows, ncols, index + 1, projection="3d")
        axes.append(ax)
        draw_learned_object(
            ax,
            channels_of_object,
            learned,
            markers,
            label_colors,
            color_by,
            norm,
            morphology_every,
        )
        ax.view_init(elev=elev, azim=azim, vertical_axis="y")
    if norm is not None:
        add_colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), axes[-1], color_by)

    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markerfacecolor="#666666",
            markeredgewidth=0,
            markersize=6 if marker == PATCH_MARKER else 9,
            label=channel,
        )
        for channel, marker in markers.items()
    ] + [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=8,
            label=label,
        )
        for label, color in label_colors.items()
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1 - 0.55 / height),
        ncols=min(len(handles), 4),
        fontsize=9,
    )
    fig.subplots_adjust(top=1 - 1.25 / height, bottom=0.02)
    return fig


def draw_learned_object(
    ax: Axes3D,
    channels: dict[str, LearnedObject],
    learned: dict[str, dict[str, dict[str, LearnedObject]]],
    markers: dict[str, str],
    label_colors: dict[str, str],
    color_by: str,
    norm: Normalize | None,
    morphology_every: int,
) -> None:
    """Draw one object's graph, every input channel on the same axes.

    Args:
        ax: The 3D axes to draw on.
        channels: The object's graphs, one per input channel.
        learned: Everything the model learned, to name child objects by.
        markers: The marker shape of each channel.
        label_colors: The color of each child-object label
            (``"<channel>: <child>"``), shared across the figure.
        color_by: See :func:`create_model_figures`.
        norm: The shared color scale when coloring by a feature.
        morphology_every: Draw the shape vector of every k-th sensor-learned
            node; 0 for none.
    """
    bounds = equal_aspect_bounds_3d([c.points.locations for c in channels.values()])
    span = max(hi - lo for lo, hi in bounds)
    labels = child_labels(channels, learned)
    for channel, obj in channels.items():
        xyz = obj.points.locations
        if channel in labels:
            # Nodes are child objects the lower LM recognized; name them.
            node_labels = np.array(labels[channel])
            for label in sorted(set(node_labels)):
                ax.scatter(
                    *xyz[node_labels == label].T,
                    s=CHILD_MARKER_SIZE,
                    marker=markers[channel],
                    color=label_colors[label],
                    edgecolors="black",
                    linewidths=0.5,
                    depthshade=False,
                )
            continue
        if norm is None:
            ax.scatter(
                *xyz.T,
                s=PATCH_MARKER_SIZE,
                marker=markers[channel],
                c=obj.colors,
                depthshade=False,
            )
        else:
            ax.scatter(
                *xyz.T,
                s=PATCH_MARKER_SIZE,
                marker=markers[channel],
                c=_feature_column(obj, color_by),
                cmap="viridis",
                norm=norm,
                depthshade=False,
            )
        if morphology_every and "pose_vectors" in obj.points.features:
            every = slice(None, None, morphology_every)
            ax.quiver(
                *xyz[every].T,
                *obj.morphology[every].T,
                length=0.05 * span,
                # An edge direction has no side, so center it on its node.
                pivot="middle" if obj.is_2d else "tail",
                color="#333333",
                linewidth=0.5,
                arrow_length_ratio=0.0,
            )
    counts = " · ".join(f"{channel} {len(obj)}" for channel, obj in channels.items())
    first = next(iter(channels.values()))
    style_3d(ax, *bounds, title=f"{first.object_id}\n{counts}")


def channel_markers(channels: Sequence[str]) -> dict[str, str]:
    """Assign a marker shape to each channel; sensor patches share circles.

    Args:
        channels: The channel names, sensor patches named ``patch_*``.

    Returns:
        Channel name to matplotlib marker.
    """
    lm_markers = cycle(LM_CHANNEL_MARKERS)
    return {
        channel: PATCH_MARKER if channel.startswith("patch") else next(lm_markers)
        for channel in channels
    }


def _feature_column(obj: LearnedObject, feature: str) -> np.ndarray:
    values = obj.points[feature]
    return values[:, 0] if values.ndim == 2 else values


def _label_colors(
    by_object: dict[str, dict[str, LearnedObject]],
    learned: dict[str, dict[str, dict[str, LearnedObject]]],
) -> dict[str, str]:
    labels = sorted(
        {
            label
            for channels in by_object.values()
            for node_labels in child_labels(channels, learned).values()
            for label in node_labels
        }
    )
    return dict(zip(labels, cycle(CHILD_COLORS)))


def _shared_norm(
    by_object: dict[str, dict[str, LearnedObject]], color_by: str
) -> Normalize | None:
    if color_by == STORED_COLOR:
        return None
    columns = [
        _feature_column(obj, color_by)
        for channels in by_object.values()
        for obj in channels.values()
        if color_by in obj.points.features
    ]
    if not columns:
        raise KeyError(f"no learned object stores a {color_by!r} feature")
    values = np.concatenate(columns)
    return Normalize(vmin=float(values.min()), vmax=float(values.max()))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "model",
        nargs="?",
        help="model.pt, run directory, or run name under MODELS_DIR "
        "(default: the current chain's newest trained stage)",
    )
    parser.add_argument(
        "--lm", nargs="+", help="learning modules to draw (default: all)"
    )
    parser.add_argument(
        "--objects", nargs="+", help="object ids to draw (default: all)"
    )
    parser.add_argument(
        "--color-by",
        default=STORED_COLOR,
        help="'color' for the stored color, or a feature name (default: color)",
    )
    parser.add_argument(
        "--morphology",
        type=int,
        default=0,
        metavar="K",
        help="draw every K-th node's normal (3D) or edge direction (2D)",
    )
    parser.add_argument("--columns", type=int, default=4, help="objects per row")
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=-20.0)
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument(
        "--out", help="output directory (default: <run_dir>/visualizations)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="also open a window: drag to rotate, scroll to zoom",
    )
    args = parser.parse_args()
    create_model_figures(
        args.model,
        lms=args.lm,
        objects=args.objects,
        color_by=args.color_by,
        morphology_every=args.morphology,
        columns=args.columns,
        elev=args.elev,
        azim=args.azim,
        dpi=args.dpi,
        out_dir=args.out,
        show=args.show,
    )
