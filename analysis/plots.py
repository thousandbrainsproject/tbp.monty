# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Shared plotting utilities for the analysis scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import (
    AbstractMovieWriter,
    FFMpegWriter,
    FuncAnimation,
    PillowWriter,
)
from matplotlib.colorbar import Colorbar
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle

# Okabe-Ito palette: distinguishable under the common colour-vision deficiencies.
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

    centre = (low + high) / 2
    half = (high - low).max() / 2 or 1.0
    return (
        [centre[0] - half, centre[0] + half],
        [centre[1] - half, centre[1] + half],
        [centre[2] - half, centre[2] + half],
    )


def style_3d(ax, xlim: list, ylim: list, zlim: list, title: str = "") -> None:
    """Apply the shared 3D panel styling. Re-apply after every ``ax.clear()``.

    Sets XYZ labels and limits, an equal box aspect, a top-down view
    (elev=90 looks along Z, azim=-90 orients X/Y conventionally), and hides
    the tick labels. Also works around two Axes3D quirks: computed zorder is
    disabled so overlay markers respect their explicit zorder even on the
    same depth plane, and the title is unclipped -- ``Axes3D.clear()``
    recreates it with clip_on=True, which crops it to the (possibly
    colorbar-shrunken) axes box.

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
    ax,
    image_shape: tuple[int, ...],
    size: int = 6,
    edgecolor: str = "red",
    linewidth: float = 1.5,
) -> None:
    """Draw a fixation square at the centre of an image panel.

    The sensor patch is centred on what it fixates, so the fixation is the
    centre of the frame.

    Args:
        ax: The 2D axes showing the image.
        image_shape: The displayed image's shape; only ``(H, W)`` is read.
        size: Side length of the square, in image pixels.
        edgecolor: Edge colour of the square.
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


def add_fixation_marker_3d(
    ax,
    location: np.ndarray,
    marker: str = "X",
    size: int = 80,
    color: str = "black",
) -> None:
    """Draw a fixation marker at a 3D world location.

    Draws with depth shading off and a high zorder so the marker stays
    visible on top of surrounding point clouds (the axes must have computed
    zorder disabled -- ``style_3d`` does this).

    Args:
        ax: The 3D axes to draw on.
        location: The ``(3,)`` world location to mark.
        marker: Matplotlib marker style.
        size: Marker size.
        color: Marker colour.
    """
    ax.scatter(
        [location[0]],
        [location[1]],
        [location[2]],
        marker=marker,
        s=size,
        color=color,
        depthshade=False,
        zorder=10,
    )


def add_image_panel(
    ax, image: np.ndarray, title: str = "", **imshow_kwargs
) -> AxesImage:
    """Show an image on a 2D panel with the axes hidden.

    Args:
        ax: The 2D axes to draw on.
        image: The initial image; animations update it via
            ``AxesImage.set_data``.
        title: Title to set, if any.
        **imshow_kwargs: Forwarded to ``ax.imshow`` (e.g. cmap, vmin, vmax).

    Returns:
        The image artist, for per-step ``set_data`` updates.
    """
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax.imshow(image, **imshow_kwargs)


def add_colorbar(mappable, ax, label: str, pad: float = 0.04) -> Colorbar:
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


def add_invisible_colorbar(ax, pad: float = 0.04) -> Colorbar:
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


def add_scatter_colorbar_3d(
    ax,
    label: str,
    cmap: str,
    vmin: float,
    vmax: float,
    marker_size: int = 5,
    alpha: float = 0.8,
) -> Colorbar:
    """Attach a colorbar to a 3D panel that is cleared and redrawn each step.

    An empty scatter anchors the colorbar, so it survives ``ax.clear()``;
    per-step scatters must pass the same cmap/vmin/vmax to stay in scale.

    Args:
        ax: The 3D axes to attach to.
        label: Colorbar label.
        cmap: Colormap name shared with the per-step scatters.
        vmin: Lower color limit.
        vmax: Upper color limit.
        marker_size: Marker size of the anchor scatter.
        alpha: Alpha of the anchor scatter.

    Returns:
        The colorbar.
    """
    anchor = ax.scatter(
        [], [], [], c=[], cmap=cmap, s=marker_size, alpha=alpha, vmin=vmin, vmax=vmax
    )
    return add_colorbar(anchor, ax, label, pad=0.08)


def save_animation(
    fig,
    update_frame,
    n_frames: int,
    out_path: Path,
    writer: AbstractMovieWriter,
    interval: int = 500,
) -> Path:
    """Animate a figure and save it, creating the output directory.

    Uses ``blit=False``: saving redraws the whole figure anyway, and blitting
    only complicates which artists appear in the saved frames.

    Args:
        fig: The assembled figure.
        update_frame: Called with the step number to redraw the figure.
        n_frames: How many steps to animate.
        out_path: Where to save the animation.
        writer: The movie writer (see ``animation_writer``).
        interval: Milliseconds between animation frames.

    Returns:
        The saved path.
    """
    anim = FuncAnimation(
        fig, update_frame, frames=n_frames, interval=interval, blit=False
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=writer)

    print(f"Animation saved to: {out_path}")
    plt.close(fig)

    return out_path


def animation_writer(fmt: str, fps: int) -> AbstractMovieWriter:
    """Pick the movie writer for an output format.

    Args:
        fmt: "gif" (PillowWriter) or "mp4" (FFMpegWriter, needs ffmpeg on
            the PATH).
        fps: Frames per second of the saved animation.

    Returns:
        The writer to pass to ``Animation.save``.

    Raises:
        ValueError: If fmt is not "gif" or "mp4".
    """
    if fmt == "gif":
        return PillowWriter(fps=fps)
    if fmt == "mp4":
        return FFMpegWriter(fps=fps)
    raise ValueError(f"fmt must be 'gif' or 'mp4', got {fmt!r}")


class RGBFrame:
    """A camera-view panel wired to one sensor module's rgba frames.

    Build it at figure-assembly time; call ``update(step)`` from the
    animation loop to show that step's frame.
    """

    def __init__(
        self,
        stats: dict,
        sensor_module_id: int | str,
        ax,
        title: str = "",
        fixation_marker: bool = False,
        invisible_colorbar: bool = False,
    ) -> None:
        """Wire the panel to an axes and show the first frame.

        Args:
            stats: Loaded episode stats.
            sensor_module_id: Which sensor module's frames to show.
            ax: The 2D axes to draw on.
            title: Base title; ``update`` appends the step counter.
            fixation_marker: Whether to mark the fixation at the frame
                centre.
            invisible_colorbar: Whether to reserve (empty) colorbar space so
                the frame lines up with sibling panels that carry one.
        """
        self._rgbas = stats.rgba(sensor_module_id)
        self._ax = ax
        self._title = title
        self._image = add_image_panel(ax, self._rgbas[0], title)
        if invisible_colorbar:
            add_invisible_colorbar(ax)
        if fixation_marker:
            add_fixation_marker(ax, self._rgbas[0].shape)

    @property
    def n_frames(self) -> int:
        """How many frames the episode holds."""
        return self._rgbas.shape[0]

    def update(self, step: int) -> None:
        """Show one step's frame.

        Args:
            step: The animation step to show.
        """
        self._image.set_data(self._rgbas[step])
        self._ax.set_title(f"{self._title} (Step {step}/{self.n_frames - 1})")


class SalienceFrame:
    """A salience-map panel wired to one sensor module's recorded maps.

    The colormap is scaled to the episode-wide value range so brightness is
    comparable across steps. Call ``update(step)`` from the animation loop.
    """

    def __init__(
        self,
        stats: dict,
        sensor_module_id: int | str | None,
        ax,
        title: str = "",
        cmap: str = "inferno",
        fixation_marker: bool = False,
        colorbar: bool = True,
    ) -> None:
        """Wire the panel to an axes.

        Args:
            stats: Loaded episode stats.
            sensor_module_id: Which sensor module's salience maps to show.
            ax: The 2D axes to draw on.
            title: Base title; ``update`` appends the step counter.
            cmap: Colormap name.
            fixation_marker: Whether to mark the fixation at the frame
                centre.
            colorbar: Whether to attach a colorbar.
        """
        self._maps = stats.salience_maps(sensor_module_id)
        self._ax = ax
        self._title = title
        first = self._maps[0] if self._maps else np.zeros((2, 2))
        self._image = add_image_panel(
            ax,
            first,
            title,
            cmap=cmap,
            vmin=min((m.min() for m in self._maps), default=0.0),
            vmax=max((m.max() for m in self._maps), default=1.0),
        )
        if colorbar:
            add_colorbar(self._image, ax, "salience")
        if fixation_marker:
            add_fixation_marker(ax, first.shape)

    def update(self, step: int) -> None:
        """Show one step's salience map.

        Args:
            step: The animation step to show.
        """
        if step < len(self._maps):
            self._image.set_data(self._maps[step])
            title = self._title
        else:
            title = f"{self._title} (none)"
        self._ax.set_title(f"{title} (Step {step}/{max(len(self._maps) - 1, 0)})")


class GoalColorer(Protocol):
    def __call__(self, goals: list[dict]) -> np.ndarray: ...


class ConfidenceColorer(GoalColorer):
    def __init__(
        self,
        cmap: str = "inferno",
        vlim: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.cmap = cmap
        self.vlim = vlim
        self._colormap = mpl.colormaps[cmap]

    def __call__(self, goals: list[dict]) -> np.ndarray:
        confidences = np.array([goal["confidence"] for goal in goals])
        vmin, vmax = self.vlim
        # Rescale so vlim spans the full colormap, not just a clipped band.
        span = (vmax - vmin) or 1.0
        normed = (np.clip(confidences, vmin, vmax) - vmin) / span
        return self._colormap(normed)


class FixedColorer(GoalColorer):
    def __init__(self, color: str | tuple[float, float, float, float]) -> None:
        if isinstance(color, str):
            self._color = mpl.colors.to_rgba(color)
        else:
            self._color = color

    def __call__(self, goals: list[dict]) -> np.ndarray:
        return np.array([self._color for _ in goals])


class GoalsScatter3D:
    """A 3D scatter panel of the attention system's goals, per step.

    Goals are colored by confidence; optionally the centre of a sensor's
    FOV is marked in world space. The axes limits span every step's goals.
    Call ``update(step)`` from the animation loop; steps beyond the recorded
    goal stream clamp to its last entry.
    """

    def __init__(
        self,
        stats: dict,
        ax,
        stage: str = "post",
        fov_sensor_module_id: int | str | None = None,
        title: str = "Goals",
        marker_size: int = 5,
        colorbar: bool = True,
        goal_colorer: GoalColorer | None = None,
    ) -> None:
        """Wire the panel to a 3D axes.

        Args:
            stats: Loaded episode stats.
            ax: The 3D axes to draw on.
            stage: Which goals to show, "pre" or "post" filtering.
            fov_sensor_module_id: Sensor module whose FOV centre to mark, or
                None for no marker.
            title: Base title; ``update`` appends the goal count.
            marker_size: Scatter marker size for goal points.
            colorbar: Whether to attach a confidence colorbar.
            goal_colorer: Optional callable to determine goal colors. If
                None, defaults to coloring by confidence.
        """
        self._goal_steps = stats.goals(stage)
        self._fov_centres = (
            stats.fov_centres(fov_sensor_module_id)
            if fov_sensor_module_id is not None
            else []
        )
        self._ax = ax
        self._title = title
        self._marker_size = marker_size
        self._bounds = equal_aspect_bounds_3d(
            [locations for locations, _ in self._goal_steps]
        )
        style_3d(ax, *self._bounds)
        self._goal_colorer = goal_colorer or ConfidenceColorer()
        if colorbar:
            # Match the colorer's scale so the legend reflects the points;
            # colorers without a cmap (e.g. FixedColorer) fall back to
            # the ConfidenceColorer defaults.
            cmap = getattr(self._goal_colorer, "cmap", "plasma")
            vmin, vmax = getattr(self._goal_colorer, "vlim", (0.0, 1.0))
            add_scatter_colorbar_3d(
                ax,
                "confidence",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                marker_size=marker_size,
                alpha=0.6,
            )

    def update(self, step: int) -> None:
        """Redraw the goal scatter for one step.

        Args:
            step: The animation step to show.
        """
        ax = self._ax
        ax.clear()
        style_3d(ax, *self._bounds)
        goal_step = min(step, len(self._goal_steps) - 1)
        locations, confidences = (
            self._goal_steps[goal_step]
            if self._goal_steps
            else (np.empty((0, 3)), np.empty(0))
        )
        if len(locations):
            ax.scatter(
                locations[:, 0],
                locations[:, 1],
                locations[:, 2],
                c=self._goal_colorer([{"confidence": c} for c in confidences]),
                s=self._marker_size,
                alpha=0.6,
            )
            ax.set_title(f"{self._title} ({len(locations)})")
        else:
            ax.set_title(f"{self._title} (0)")
            ax.text2D(0.5, 0.5, "No goals", transform=ax.transAxes, ha="center")

        fov_centre = self._fov_centres[step] if step < len(self._fov_centres) else None
        if fov_centre is not None:
            add_fixation_marker_3d(ax, fov_centre)
