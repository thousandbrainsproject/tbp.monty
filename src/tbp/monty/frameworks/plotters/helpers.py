# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt
from matplotlib.collections import LineCollection

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def is_interactive_backend() -> bool:
    """Whether the active matplotlib backend can run a blocking event loop.

    Returns:
        True if the current backend is an interactive (GUI) backend.
    """
    return mpl.get_backend() in mpl.rcsetup.interactive_bk


def unit(vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return `vec` normalized to unit length, or unchanged if near zero.

    Args:
        vec: The vector to normalize.

    Returns:
        The unit vector, or the original vector when its norm is negligible.
    """
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-9 else vec


def color_kwarg(color: object) -> dict:
    """Map a group color to the right scatter color keyword.

    Args:
        color: `None` to use the axis color cycle, a per-point `(M, 3)` RGB array, or a
            single matplotlib color.

    Returns:
        `{}` for the cycle, `{"c": color}` for a per-point array, else
        `{"color": color}` for a single color.
    """
    if color is None:
        return {}
    if isinstance(color, np.ndarray) and color.ndim == 2:
        return {"c": color}
    return {"color": color}


def is_3d(pts: npt.NDArray[np.float64]) -> bool:
    """Whether a point cloud has a real third dimension.

    A 2D sensor module pins every location to the `z = 0` plane, so a non-zero spread
    in z marks a genuine 3D graph. Only meaningful once all points are known
    (inference); a partially built graph may look planar by chance.

    Args:
        pts: The `(M, 3)` locations to inspect.

    Returns:
        True when the points vary in z, False otherwise.
    """
    return pts.shape[1] >= 3 and not np.allclose(pts[:, 2], 0.0)


def frame_center_half(
    pts: npt.NDArray[np.float64],
    base_size: float = 0.05,
    step: float = 0.05,
) -> tuple[npt.NDArray[np.float64], float]:
    """Center and half-side of the square/cube enclosing `pts`.

    The frame starts at `base_size` and grows in `step` increments, so it only ever
    changes size when points cross a step boundary rather than rescaling continuously
    with every new observation. Works for 2D `(M, 2)` or 3D `(M, 3)` points, so the flat
    2D channel view and the 3D buffer view share it.

    Args:
        pts: The `(M, 2)` or `(M, 3)` points the frame must enclose.
        base_size: Side length in meters of the smallest square/cube frame.
        step: Increment in meters by which the frame grows when points exceed it.

    Returns:
        The per-axis center and the frame's half side length.
    """
    low = pts[:, :3].min(axis=0)
    high = pts[:, :3].max(axis=0)
    center = (low + high) / 2

    span = float((high - low).max())
    size = base_size
    if span > size:
        size += step * math.ceil((span - base_size) / step)

    return center, size / 2


def corner_rect(
    bbox,
    width_frac: float,
    height: float,
    top_pad: float = 0.0,
) -> list[float]:
    """Figure-coordinate rectangle for a corner inset over a host panel.

    Args:
        bbox: The host panel's figure-coordinate bounding box.
        width_frac: The inset width as a fraction of the host panel's width.
        height: The inset height in figure coordinates.
        top_pad: Gap in figure coordinates between the host's top and the inset.

    Returns:
        The `[left, bottom, width, height]` rectangle in the panel's top-left corner.
    """
    width = (bbox.x1 - bbox.x0) * width_frac
    top = bbox.y1 - top_pad
    return [bbox.x0 + 0.005, top - height, width, height]


def planar_style(
    n: int,
    hsv: npt.NDArray[np.float64] | None,
    flags: npt.NDArray[np.float64] | None,
    pose: npt.NDArray[np.float64] | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], npt.NDArray | None]:
    """Resolve per-point colors, an edge mask, and edge tangents for a planar cloud.

    Shared by the inference MLH view (reading graph features) and the training buffer
    view of a 2D sensor-module channel (reading buffer features), so both render the
    same way from whichever source supplies the raw feature arrays.

    Args:
        n: The number of points.
        hsv: The `(n, >=3)` hsv feature, or `None` to fall back to dark gray.
        flags: The `(n, 1)` pose_fully_defined feature, or `None` for no edges.
        pose: The `(n, 9)` flattened pose vectors, or `None` for no tangents.

    Returns:
        The `(n, 3)` colors, the `(n,)` edge mask, and the `(E, 2)` edge tangents of the
        masked points (or `None` when no edge defines a pose).
    """
    if hsv is not None and hsv.shape[0] == n and hsv.shape[1] >= 3:
        colors = mcolors.hsv_to_rgb(np.clip(hsv[:, :3], 0.0, 1.0))
    else:
        colors = np.full((n, 3), 0.2)
    if flags is not None and flags.shape[0] == n:
        edge_mask = flags[:, 0].astype(bool)
    else:
        edge_mask = np.zeros(n, dtype=bool)
    if edge_mask.any() and pose is not None and pose.shape[0] == n:
        tangents = pose[edge_mask, 3:5]
    else:
        tangents = None
    return colors, edge_mask, tangents


def draw_buffer_series(
    main_ax: Axes,
    proj_axes: list[Axes],
    groups: list[tuple[npt.NDArray[np.float64], object, str | None]],
    title: str,
    title_fontsize: int | None,
    show_ticks: bool = True,
) -> None:
    """Draw point-cloud groups in a 3D axis plus its three 2D projections.

    The 3D cube and all three projections share one stepped frame, so each head-on view
    uses the same width and height as the corresponding cube faces. A legend is drawn
    only when at least one group carries a label.

    Args:
        main_ax: The 3D axis for the point cloud.
        proj_axes: The three 2D axes for the XY/XZ/YZ projections.
        groups: The `(points, color, label)` groups to overlay, where `color` is `None`
            (cycle), a single color, or a per-point `(M, 3)` RGB array, and `label` is
            the legend entry or `None` to omit it.
        title: The title for the 3D axis.
        title_fontsize: Font size for the 3D title, or `None` for the default.
        show_ticks: Whether to draw axis ticks; `False` drops them on every panel so the
            small stacked Details plots stay readable.
    """
    main_ax.cla()
    main_ax.set_title(title, fontsize=title_fontsize)
    all_points = [pts for pts, _, _ in groups]
    stacked = np.concatenate(all_points)
    center, half = frame_center_half(stacked)
    for pts, color, label in groups:
        main_ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=6,
            label=label,
            **color_kwarg(color),
        )
    main_ax.set_xlim(center[0] - half, center[0] + half)
    main_ax.set_ylim(center[1] - half, center[1] + half)
    main_ax.set_zlim(center[2] - half, center[2] + half)
    main_ax.set_box_aspect((1, 1, 1))
    if not show_ticks:
        main_ax.set_xticks([])
        main_ax.set_yticks([])
        main_ax.set_zticks([])
    if any(label is not None for _, _, label in groups):
        main_ax.legend(fontsize=8, loc="best")

    # The three head-on 2D projections, as (x_dim, y_dim, label).
    projections = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    for ax, (a, b, name) in zip(proj_axes, projections):
        ax.cla()
        for pts, color, _ in groups:
            ax.scatter(pts[:, a], pts[:, b], s=4, **color_kwarg(color))
        ax.set_xlim(center[a] - half, center[a] + half)
        ax.set_ylim(center[b] - half, center[b] + half)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=7)
        if show_ticks:
            ax.tick_params(labelsize=6)
        else:
            ax.set_xticks([])
            ax.set_yticks([])


def draw_2d_segments(
    ax: Axes,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    colors: npt.NDArray[np.float64],
    edge_mask: npt.NDArray[np.bool_],
    tangents: npt.NDArray[np.float64] | None,
) -> None:
    """Draw a planar cloud as hsv-colored dots with edge-oriented segments.

    Nodes where an edge defines the pose are drawn as short dashed segments along the
    edge tangent; the rest are drawn as dots. Shared by the inference MLH view and the
    training buffer view of a 2D sensor-module channel. The view uses the same stepped
    square frame as the 3D buffer view, so it starts at the base size and grows in steps
    as points accumulate rather than rescaling continuously.

    Args:
        ax: The 2D axis to draw into.
        x: The x coordinates, shape `(N,)`.
        y: The y coordinates, shape `(N,)`.
        colors: The per-point `(N, 3)` RGB colors.
        edge_mask: The boolean `(N,)` mask of points where an edge defines the pose.
        tangents: The `(E, 2)` edge tangents of the masked points, or `None`.
    """
    ax.scatter(x[~edge_mask], y[~edge_mask], color=colors[~edge_mask], s=6, zorder=1)
    center, half = frame_center_half(np.stack([x, y], axis=1))
    if edge_mask.any() and tangents is not None and len(tangents):
        normed = tangents / np.clip(
            np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9, None
        )
        seg_half = 0.04 * half
        centers = np.stack([x[edge_mask], y[edge_mask]], axis=1)
        segments = np.stack(
            [centers - seg_half * normed, centers + seg_half * normed], axis=1
        )
        ax.add_collection(
            LineCollection(
                segments,
                colors=colors[edge_mask],
                linestyles="--",
                linewidths=1.2,
                zorder=2,
            )
        )
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
