# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Panels: one animated figure panel each, drawing a ``Stream``.

A panel builds its artists once on an axes from a view (see ``views``) and
``update(step)`` moves them to what its streams hold at that episode step;
nothing is cleared and redrawn. Panels know nothing about where data comes
from. The matplotlib helpers the panels build on (axis styling, colorbars,
fixation markers) live at the bottom.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle

from analysis.views import Points

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage
    from matplotlib.lines import Line2D
    from matplotlib.table import Table
    from matplotlib.text import Text
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Path3DCollection

    from analysis.views import Stream

# The ranking table stops here so its rows stay legible with many objects.
MAX_TABLE_ROWS = 12


class Panel(Protocol):
    """One panel of an animated figure: built once, redrawn per step."""

    def update(self, step: int) -> None: ...


class Image(Panel):
    """An image per step: the latest frame at or before the step.

    An ``overlay`` stream (transparent RGBA, e.g. a segmentation mask) is
    drawn on top and alpha-blended by matplotlib. It ticks with the frames,
    so a held frame keeps its own overlay.
    """

    def __init__(
        self,
        stream: Stream[np.ndarray],
        ax: Axes,
        title: str = "",
        cmap: str | None = None,
        vlim: tuple[float, float] | None = None,
        colorbar: str | None = None,
        overlay: Stream[np.ndarray] | None = None,
        fixation_marker: bool = False,
        reserve_colorbar: bool = False,
    ) -> None:
        """Build the panel's artists on an axes.

        Args:
            stream: ``(H, W[, C])`` images; must hold at least one.
            ax: The 2D axes to draw on.
            title: Base title; ``update`` appends the step counter.
            cmap: Colormap name for single-channel images.
            vlim: ``(vmin, vmax)`` of the color scale; defaults to the
                stream's range.
            colorbar: Label of a colorbar to attach, or None for none.
            overlay: Transparent ``(H, W, 4)`` images blended on top,
                ticking with the frames.
            fixation_marker: Whether to mark the fixation at the frame
                center (the sensor patch is centered on what it fixates).
            reserve_colorbar: Whether to reserve (empty) colorbar space so
                the panel lines up with siblings that carry one.
        """
        self._stream: Stream[np.ndarray] = stream
        self._overlay: Stream[np.ndarray] | None = (
            overlay if overlay is not None and len(overlay) else None
        )
        self._ax: Axes = ax
        self._title: str = title
        scale = {}
        if cmap is not None:
            images = np.asarray(list(stream))
            vmin, vmax = vlim or (float(images.min()), float(images.max()))
            scale = {"cmap": cmap, "vmin": vmin, "vmax": vmax}
        first = next(iter(stream))
        ax.axis("off")
        if title:
            ax.set_title(title)
        self._image: AxesImage = ax.imshow(first, **scale)
        self._overlay_image: AxesImage | None = (
            None
            if self._overlay is None
            else ax.imshow(next(iter(self._overlay)), visible=False)
        )
        if colorbar is not None:
            add_colorbar(self._image, ax, colorbar)
        if reserve_colorbar:
            add_invisible_colorbar(ax)
        if fixation_marker:
            add_fixation_marker(ax, first.shape)

    def update(self, step: int) -> None:
        image = self._stream.latest(step)
        if image is not None:
            self._image.set_data(image.value)
        if self._overlay_image is not None:
            overlay = self._overlay.latest(step)
            self._overlay_image.set_visible(overlay is not None)
            if overlay is not None:
                self._overlay_image.set_data(overlay.value)
        self._ax.set_title(f"{self._title} (Step {step}/{self._stream.n_steps - 1})")


class Scatter3D(Panel):
    """A 3D scatter per step of a points stream, colored by one column.

    The axes limits and the color scale are fixed across the episode so
    frames compare. The latest recorded cloud is held between recordings;
    one without points shows ``empty_text``.
    """

    def __init__(
        self,
        stream: Stream[Points],
        ax: Axes3D,
        color: str,
        title: str,
        cmap: str = "viridis",
        vlim: tuple[float, float] | None = None,
        bounds: tuple[list, list, list] | None = None,
        marker_size: int = 5,
        alpha: float = 0.8,
        empty_text: str = "No points",
        fixation: Stream[np.ndarray] | None = None,
    ) -> None:
        """Build the panel's artists on a 3D axes.

        Args:
            stream: Per recorded step, a cloud carrying the ``color``
                feature; an empty cloud is a step that proposed nothing.
            ax: The 3D axes to draw on.
            color: The feature to color points by.
            title: Base title; ``update`` appends the point count and step.
            cmap: Colormap name.
            vlim: ``(vmin, vmax)`` of the color scale; defaults to
                :func:`shared_scale` of the stream alone.
            bounds: ``(xlim, ylim, zlim)``; defaults likewise. Compute both
                once with :func:`shared_scale` to align sibling panels.
            marker_size: Scatter marker size.
            alpha: Marker alpha.
            empty_text: Shown on steps with no points.
            fixation: ``(3,)`` locations to mark per step, or None.
        """
        self._stream: Stream[Points] = stream
        self._ax: Axes3D = ax
        self._color: str = color
        self._title: str = title
        self._fixation: Stream[np.ndarray] | None = fixation
        if bounds is None or vlim is None:
            own_bounds, own_vlim = shared_scale(stream, color=color)
            bounds, vlim = bounds or own_bounds, vlim or own_vlim
        self._vlim: tuple[float, float] = vlim
        style_3d(ax, *bounds)
        self._scatter: Path3DCollection = ax.scatter(
            [],
            [],
            [],
            c=[],
            cmap=cmap,
            vmin=self._vlim[0],
            vmax=self._vlim[1],
            s=marker_size,
            alpha=alpha,
        )
        add_colorbar(self._scatter, ax, color, pad=0.08)
        self._empty: Text = ax.text2D(
            0.5, 0.5, empty_text, transform=ax.transAxes, ha="center", visible=False
        )
        self._fixation_marker: Path3DCollection = ax.scatter(
            [], [], [], marker="X", s=80, color="black", depthshade=False, zorder=10
        )

    def update(self, step: int) -> None:
        tick = self._stream.latest(step)
        points = None if tick is None else tick.value
        n = 0 if points is None else len(points)
        if n:
            self._scatter._offsets3d = tuple(points.locations.T)
            self._scatter.set_array(points[self._color])
        else:
            self._scatter._offsets3d = ([], [], [])
            self._scatter.set_array(np.empty(0))
        self._empty.set_visible(n == 0)
        fixation = self._fixation.latest(step) if self._fixation is not None else None
        self._fixation_marker.set_visible(fixation is not None)
        if fixation is not None:
            self._fixation_marker._offsets3d = tuple(
                [coordinate] for coordinate in fixation.value
            )
        self._ax.set_title(f"{self._title} ({n}, Step {step})")


class EvidenceTraces(Panel):
    """An LM's per-object max-evidence traces, revealed as steps pass.

    The strongest objects get a labelled, color-blind-safe line each; the
    rest are thin gray context lines (see :func:`evidence_colors`). Dashed
    verticals mark resampling bursts once reached, and a cursor marks the
    current LM step.
    """

    def __init__(
        self,
        maxes: Stream[dict[str, float]],
        counts: Stream[int],
        ax: Axes,
        title: str,
        recognized: np.ndarray | None = None,
    ) -> None:
        """Build the panel's artists on an axes.

        Args:
            maxes: Per LM step, each object's max evidence
                (``evidence_stream(...).map(max_evidence)``).
            counts: Per LM step, how many hypotheses were scored; growth
                marks a resampling burst.
            ax: The 2D axes to draw on.
            title: Base title; ``update`` appends the LM step counter.
            recognized: Episode steps at which the module held a recognized
                object (``views.recognized_steps``); shaded green once the
                animation reaches them. None or empty for no shading.
        """
        self._maxes: Stream[dict[str, float]] = maxes
        self._ax: Axes = ax
        self._title: str = title
        # The column view of the stream: each object's trace over LM steps.
        self._traces: dict[str, np.ndarray] = {
            name: np.array([tick.get(name, np.nan) for tick in maxes])
            for name in objects_sorted_by_max_evidence(maxes)
        }
        traces = np.array(list(self._traces.values()))
        low, high = (
            (float(np.nanmin(traces)), float(np.nanmax(traces)))
            if traces.size
            else (0.0, 1.0)
        )
        pad = 0.05 * (high - low or 1.0)
        ax.set_xlim(0, max(len(maxes) - 1, 1))
        ax.set_ylim(low - pad, high + pad)
        ax.set_xlabel("LM step")
        ax.set_ylabel("Max evidence")

        self._lines: dict[str, Line2D] = {}
        for name, color in evidence_colors(maxes).items():
            labelled = color in OKABE_ITO
            (self._lines[name],) = ax.plot(
                [],
                [],
                color=color,
                label=name if labelled else None,
                linewidth=1.5 if labelled else 0.8,
                zorder=2 if labelled else 1,
            )
        self._recognized: list[tuple[int, int, Rectangle]] = []
        for run_start, run_end in _runs(
            np.unique(
                [
                    -1 if (tick := maxes.latest(s)) is None else tick.index
                    for s in recognized
                ]
            )
            if recognized is not None and len(recognized)
            else np.empty(0, dtype=int)
        ):
            span = ax.axvspan(
                run_start - 0.5,
                run_end + 0.5,
                color="#009E73",
                alpha=0.15,
                zorder=0,
                label="recognized" if not self._recognized else None,
            )
            self._recognized.append((run_start, run_end, span))
        ax.legend(fontsize=7, loc="upper left")
        self._bursts: list[tuple[int, Line2D]] = [
            (int(step), ax.axvline(step, color="gray", linestyle="--", alpha=0.5))
            for step in burst_steps(counts)
        ]
        self._cursor: Line2D = ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)

    def update(self, step: int) -> None:
        tick = self._maxes.latest(step)
        lm_step = -1 if tick is None else tick.index
        revealed = np.arange(lm_step + 1)
        for name, line in self._lines.items():
            line.set_data(revealed, self._traces[name][: lm_step + 1])
        for burst_step, vline in self._bursts:
            vline.set_visible(lm_step >= burst_step)
        for run_start, run_end, span in self._recognized:
            span.set_visible(lm_step >= run_start)
            if lm_step >= run_start:
                # Grow with the cursor instead of revealing the whole run.
                span.set_width(min(run_end, lm_step) + 0.5 - span.get_x())
        self._cursor.set_xdata([max(lm_step, 0)] * 2)
        self._cursor.set_visible(lm_step >= 0)
        self._ax.set_title(
            f"{self._title} (LM step {max(lm_step, 0)}/{max(len(self._maxes) - 1, 0)})"
        )


class EvidenceRanking(Panel):
    """A table ranking objects by max evidence at the current LM step.

    Rows are tinted with the color the object's trace is drawn in
    (:func:`evidence_colors`), so the table and the traces read together.
    """

    def __init__(self, maxes: Stream[dict[str, float]], ax: Axes, title: str) -> None:
        """Build the panel's artists on an axes.

        Args:
            maxes: Per LM step, each object's max evidence
                (``evidence_stream(...).map(max_evidence)``).
            ax: The 2D axes to draw on.
            title: The panel title.
        """
        self._maxes: Stream[dict[str, float]] = maxes
        self._colors: dict[str, str] = evidence_colors(maxes)
        self._ax: Axes = ax
        ax.set_title(title)
        ax.axis("off")
        self._note: Text = ax.text(
            0.5, 0.5, "No evidence yet", transform=ax.transAxes, ha="center"
        )
        self._table: Table | None = None

    def update(self, step: int) -> None:
        # A matplotlib table cannot be edited in place; rebuild it.
        if self._table is not None:
            self._table.remove()
            self._table = None
        tick = self._maxes.latest(step)
        ranking = (
            sorted(tick.value.items(), key=lambda pair: pair[1], reverse=True)
            if tick is not None and tick.value
            else []
        )[:MAX_TABLE_ROWS]
        self._note.set_visible(not ranking)
        if not ranking:
            return
        self._table = self._ax.table(
            cellText=[[name, f"{value:.2f}"] for name, value in ranking],
            colLabels=("Object", "Max evidence"),
            loc="center",
            cellLoc="left",
            colWidths=(0.6, 0.4),
        )
        self._table.auto_set_font_size(False)  # noqa: FBT003
        self._table.set_fontsize(9)
        self._table.scale(1, 1.3)
        for row, (name, _) in enumerate(ranking, start=1):
            self._table[row, 0].set_facecolor(
                mcolors.to_rgba(self._colors[name], alpha=0.3)
            )


def _runs(steps: np.ndarray) -> list[tuple[int, int]]:
    # Contiguous runs of sorted non-negative ints as (start, end) pairs.
    steps = steps[steps >= 0]
    if not len(steps):
        return []
    breaks = np.where(np.diff(steps) > 1)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(steps) - 1]])
    return [(int(steps[s]), int(steps[e])) for s, e in zip(starts, ends)]


def burst_steps(counts: Stream[int]) -> np.ndarray:
    """The LM steps at which the hypothesis count grew (a resampling burst).

    Args:
        counts: Per LM step, the hypothesis count
            (``evidence_stream(...).map(hypothesis_count)``).

    Returns:
        The steps, as indices into the stream.
    """
    n_hypotheses = np.fromiter(counts, dtype=int)
    return np.where(np.diff(n_hypotheses) > 0)[0] + 1


def objects_sorted_by_max_evidence(maxes: Stream[dict[str, float]]) -> list[str]:
    """Object names by their peak max evidence over the episode, highest first.

    Args:
        maxes: Per LM step, each object's max evidence
            (``evidence_stream(...).map(max_evidence)``).

    Returns:
        The names.
    """
    peaks: dict[str, float] = {}
    for tick in maxes:
        for name, value in tick.items():
            peaks[name] = max(value, peaks.get(name, -np.inf))
    return sorted(peaks, key=lambda name: peaks[name], reverse=True)


def evidence_colors(maxes: Stream[dict[str, float]]) -> dict[str, str]:
    """The color each object's evidence is drawn in, deterministically.

    The strongest objects (by peak evidence) take the Okabe-Ito palette in
    order; the rest share a light gray. Every panel showing the same
    evidence uses this one assignment.

    Returns:
        Object name to matplotlib color, strongest first.
    """
    return {
        name: OKABE_ITO[rank] if rank < len(OKABE_ITO) else "0.8"
        for rank, name in enumerate(objects_sorted_by_max_evidence(maxes))
    }


def shared_scale(
    *streams: Stream[Points], color: str
) -> tuple[tuple[list, list, list], tuple[float, float]]:
    """One set of axis bounds and one color scale covering several point streams.

    Pass the result's parts as ``bounds`` and ``vlim`` to sibling
    :class:`Scatter3D` panels so they compare visually.

    Args:
        *streams: The point streams to cover.
        color: The feature the color scale covers; extended to include
            ``[0, 1]``.

    Returns:
        ``(bounds, (vmin, vmax))``.
    """
    points = Points.concat(p for stream in streams for p in stream)
    values = points.features.get(color, np.empty(0))
    vlim = (
        float(min(0.0, values.min())) if len(values) else 0.0,
        float(max(1.0, values.max())) if len(values) else 1.0,
    )
    return equal_aspect_bounds_3d([points.locations]), vlim


# Okabe-Ito palette: distinguishable under the common color-vision deficiencies.
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


def equal_aspect_bounds_3d(point_sets: list[np.ndarray]) -> tuple[list, list, list]:
    """Compute equal-aspect axis limits spanning every given point set.

    Args:
        point_sets: Point arrays of shape ``(N, 3)``; empty or None entries
            are ignored.

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

    center = (low + high) / 2
    half = (high - low).max() / 2 or 1.0
    return (
        [center[0] - half, center[0] + half],
        [center[1] - half, center[1] + half],
        [center[2] - half, center[2] + half],
    )


def style_3d(ax: Axes3D, xlim: list, ylim: list, zlim: list, title: str = "") -> None:
    """Apply the shared 3D panel styling, once when a panel is built.

    Sets XYZ labels and limits, an equal box aspect, a top-down view
    (elev=90 looks along Z, azim=-90 orients X/Y conventionally), and hides
    the tick labels. Computed zorder is disabled so overlay markers respect
    their explicit zorder even on the same depth plane, and the title is
    unclipped so the (possibly colorbar-shrunken) axes box does not crop it.

    Args:
        ax: The 3D axes to style.
        xlim: X-axis limits.
        ylim: Y-axis limits.
        zlim: Z-axis limits.
        title: Title to set, if any.
    """
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_box_aspect([1, 1, 1])
    ax.computed_zorder = False
    ax.title.set_clip_on(False)
    ax.view_init(elev=90, azim=-90)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def add_fixation_marker(
    ax: Axes,
    image_shape: tuple[int, ...],
    size: int = 6,
    edgecolor: str = "red",
    linewidth: float = 1.5,
) -> None:
    """Draw a fixation square at the center of an image panel.

    The sensor patch is centered on what it fixates, so the fixation is the
    center of the frame.

    Args:
        ax: The 2D axes showing the image.
        image_shape: The displayed image's shape; only ``(H, W)`` is read.
        size: Side length of the square, in image pixels.
        edgecolor: Edge color of the square.
        linewidth: Line width of the square.
    """
    height, width = image_shape[:2]
    ax.add_patch(
        Rectangle(
            (width // 2 - size / 2, height // 2 - size / 2),
            size,
            size,
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor="none",
        )
    )


def add_colorbar(
    mappable: ScalarMappable, ax: Axes, label: str, pad: float = 0.04
) -> Colorbar:
    """Attach the standard slim colorbar with a rotated label.

    Args:
        mappable: The artist the colorbar reads its scale from.
        ax: The axes to steal space from.
        label: Colorbar label.
        pad: Fraction of the axes to leave between the axes and the bar.

    Returns:
        The colorbar.
    """
    bar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=pad)
    bar.set_label(label, rotation=270, labelpad=15)
    return bar


def add_invisible_colorbar(ax: Axes, pad: float = 0.04) -> Colorbar:
    """Reserve the space a colorbar would take, without showing one.

    Steals the same layout space as ``add_colorbar`` and then hides it, so a
    panel without a color scale keeps the same drawing width as sibling
    panels that carry a real colorbar.

    Args:
        ax: The axes to steal space from.
        pad: Fraction of the axes to leave between the axes and the bar.

    Returns:
        The (hidden) colorbar.
    """
    bar = plt.colorbar(cm.ScalarMappable(), ax=ax, fraction=0.046, pad=pad)
    bar.ax.set_visible(False)
    return bar


def attach_scroll_zoom(fig: Figure, zoom_per_step: float = 1.15) -> None:
    """Make the scroll wheel zoom the 3D axes under the cursor.

    Matplotlib's 3D axes rotate on left-drag and zoom only on right-drag, so
    scroll zoom is wired up by shrinking or growing the axis limits around
    their center.

    Args:
        fig: The figure whose 3D axes should zoom.
        zoom_per_step: Limit scale factor per scroll step; scrolling up
            zooms in.
    """

    def on_scroll(event) -> None:
        ax = event.inaxes
        if ax is None or not hasattr(ax, "get_zlim3d"):
            return
        factor = zoom_per_step ** (-event.step)
        for get_limits, set_limits in (
            (ax.get_xlim3d, ax.set_xlim3d),
            (ax.get_ylim3d, ax.set_ylim3d),
            (ax.get_zlim3d, ax.set_zlim3d),
        ):
            low, high = get_limits()
            center = (low + high) / 2.0
            half_width = (high - low) / 2.0 * factor
            set_limits(center - half_width, center + half_width)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)
