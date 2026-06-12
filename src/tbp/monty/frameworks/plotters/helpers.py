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
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import quaternion as qt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tbp.monty.frameworks.models.evidence_matching.burst_sampling import (
    BurstSamplingHypothesesUpdater,
)
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.two_d_sensor_module import TwoDSensorModule

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from tbp.monty.frameworks.models.abstract_monty_classes import (
        LearningModule,
        Monty,
        SensorModule,
    )


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


def draw_section_dividers(fig: Figure, sim_spec, monty_spec, details_spec) -> None:
    """Draw vertical separators in the gaps between the three column sections.

    Args:
        fig: The figure to draw the separators on.
        sim_spec: The Simulator column's subplot spec.
        monty_spec: The Monty column's subplot spec.
        details_spec: The Details column's subplot spec.
    """
    sim = sim_spec.get_position(fig)
    monty = monty_spec.get_position(fig)
    details = details_spec.get_position(fig)
    label_margin = 0.05
    gaps = [
        min((sim.x1 + monty.x0) / 2, monty.x0 - label_margin),
        min((monty.x1 + details.x0) / 2, details.x0 - label_margin),
    ]
    for x in gaps:
        fig.add_artist(
            Line2D(
                [x, x],
                [sim.y0, sim.y1],
                transform=fig.transFigure,
                color="0.7",
                linewidth=1,
            )
        )


class EvidenceHistory:
    """Per-LM accumulation of object evidence and hypothesis counts over an episode.

    Holds one record per learning module so the displayed LM can switch mid-episode and
    still reveal a fully populated history. Pure data: the Details section reads it to
    draw the evidence and number-of-hypotheses line plots.
    """

    def __init__(self) -> None:
        self.steps_by_lm: dict[str, list[int]] = {}
        self.evidence_by_lm: dict[str, dict[str, list[float]]] = {}
        self.num_hyp_by_lm: dict[str, dict[str, list[float]]] = {}
        self.burst_steps_by_lm: dict[str, list[int]] = {}
        self._last_accumulated_step: int | None = None

    def accumulate(self, learning_modules: list[LearningModule], step: int) -> None:
        """Record this step's evidence history for every evidence-supporting LM.

        Accumulates once per step regardless of how many times the frame is redrawn, so
        switching the displayed LM mid-episode reveals a fully populated history.

        Args:
            learning_modules: All of the model's learning modules.
            step: The index of the current step within the episode.
        """
        if self._last_accumulated_step == step:
            return
        self._last_accumulated_step = step
        for lm in learning_modules:
            if isinstance(lm, EvidenceGraphLM):
                self._append(lm, step)

    def _append(self, lm: EvidenceGraphLM, step: int) -> None:
        """Append one learning module's evidence and hypothesis counts to its history.

        One series per object id with a non-empty hypothesis space; objects appearing
        late are NaN-backfilled so every series aligns to the LM's step list. Steps with
        a sampling burst are recorded for vertical markers.

        Args:
            lm: The learning module whose current state is recorded.
            step: The index of the current step within the episode.
        """
        mlh = lm.get_current_mlh()
        if not mlh or mlh.get("graph_id") == "no_observations_yet":
            return

        graph_ids, evidences = lm.evidence_for_each_graph()
        _, num_hyps = lm.num_hypotheses_for_each_graph()
        evidence_by_id = dict(zip(graph_ids, evidences))
        num_hyp_by_id = dict(zip(graph_ids, num_hyps))

        lm_id = lm.learning_module_id
        steps = self.steps_by_lm.setdefault(lm_id, [])
        evidence_history = self.evidence_by_lm.setdefault(lm_id, {})
        num_hyp_history = self.num_hyp_by_lm.setdefault(lm_id, {})
        burst_steps = self.burst_steps_by_lm.setdefault(lm_id, [])

        steps.append(step)
        n = len(steps)
        for graph_id in graph_ids:
            if graph_id not in evidence_history:
                evidence_history[graph_id] = [np.nan] * (n - 1)
                num_hyp_history[graph_id] = [np.nan] * (n - 1)
        for graph_id in evidence_history:
            evidence_history[graph_id].append(evidence_by_id.get(graph_id, np.nan))
            num_hyp_history[graph_id].append(num_hyp_by_id.get(graph_id, np.nan))
        if self._in_burst(lm):
            burst_steps.append(step)

    @staticmethod
    def _in_burst(learning_module: EvidenceGraphLM) -> bool:
        """Whether the LM's burst-sampling updater is currently in a burst.

        Args:
            learning_module: The learning module whose updater is inspected.

        Returns:
            True when the LM uses a `BurstSamplingHypothesesUpdater` and a burst is in
            progress, False otherwise.
        """
        updater = learning_module.hypotheses_updater
        if not isinstance(updater, BurstSamplingHypothesesUpdater):
            return False
        return updater.sampling_burst_steps > 0


class ChannelView:
    """The selected learning module and input channel, with channel resolution.

    Owns the runtime selection (which learning module is displayed and which of its
    input channels is selected) and answers the resolution and feature queries the
    panels read from that selection: the channels in the displayed LM, the sensor or
    learning module feeding a channel, the sensor module driving the Simulator, and the
    per-channel buffer features aligned to their valid points. Both the displayed LM and
    the selected channel cycle at runtime through the two selector buttons.
    """

    def __init__(self, model: Monty) -> None:
        """Select the first learning module and defer the channel default.

        The selected channel starts unset and is defaulted on first use by
        `ensure_channel`, once the displayed LM's buffer has channels.

        Args:
            model: The Monty model whose sensor and learning modules are read.
        """
        self.model = model
        self.lm_index = 0
        self.lm = model.learning_modules[0]
        self.channel: str | None = None
        self.supports_evidence = isinstance(self.lm, EvidenceGraphLM)

    def cycle_lm(self) -> None:
        """Advance to the next learning module, resetting the channel to its default.

        Switching the displayed LM recomputes whether the inference panels are supported
        and selects that LM's default channel.
        """
        self.lm_index = (self.lm_index + 1) % len(self.model.learning_modules)
        self.lm = self.model.learning_modules[self.lm_index]
        self.supports_evidence = isinstance(self.lm, EvidenceGraphLM)
        self.channel = self.default_channel()

    def cycle_channel(self) -> bool:
        """Advance to the next input channel of the displayed LM.

        Returns:
            True when the channel advanced, False when the displayed LM has no channels.
        """
        channels = self.lm_channels()
        if not channels:
            return False
        if self.channel in channels:
            index = (channels.index(self.channel) + 1) % len(channels)
        else:
            index = 0
        self.channel = channels[index]
        return True

    def ensure_channel(self) -> bool:
        """Default the selected channel on first use once the buffer has channels.

        Returns:
            True when the channel was just defaulted, False when it was already set.
        """
        if self.channel is None:
            self.channel = self.default_channel()
            return True
        return False

    def labels(self) -> tuple[str, str]:
        """Return the current `(learning module, channel)` selector button labels.

        Returns:
            The displayed-LM label and the selected-channel label.
        """
        return (
            f"LM: {self.lm.learning_module_id}",
            f"ch: {self.channel or '-'}",
        )

    def lm_channels(self) -> list[str]:
        """Return the displayed LM's input channels in observation order.

        Returns:
            The channel ids (sender ids) seen in the displayed LM's buffer.
        """
        return list(self.lm.buffer.channel_sender_types)

    def default_channel(self) -> str | None:
        """Return the displayed LM's default channel: the first sensor-module channel.

        Returns:
            The first SM channel, else the first channel of any type, else `None` when
            the buffer holds no channels yet.
        """
        channels = self.lm_channels()
        sender_types = self.lm.buffer.channel_sender_types
        sm_channels = [c for c in channels if sender_types.get(c) == "SM"]
        if sm_channels:
            return sm_channels[0]
        return channels[0] if channels else None

    def resolve_sm_channel(self, channel: str | None) -> SensorModule | None:
        """Resolve an SM channel id to its sensor module instance.

        Args:
            channel: The channel id to resolve.

        Returns:
            The matching sensor module, or `None` when the channel is not a sensor
            module channel or no module carries that id.
        """
        if channel is None:
            return None
        if self.lm.buffer.channel_sender_types.get(channel) != "SM":
            return None
        return next(
            (s for s in self.model.sensor_modules if s.sensor_module_id == channel),
            None,
        )

    def resolve_lm_channel(self, channel: str | None) -> LearningModule | None:
        """Resolve an LM channel id to its source learning module instance.

        Args:
            channel: The channel id to resolve.

        Returns:
            The matching learning module, or `None` when the channel is not a
            learning-module channel or no module carries that id.
        """
        if channel is None:
            return None
        if self.lm.buffer.channel_sender_types.get(channel) != "LM":
            return None
        return next(
            (m for m in self.model.learning_modules if m.learning_module_id == channel),
            None,
        )

    def simulator_sm(self) -> tuple[SensorModule | None, str | None]:
        """Return the sensor module to drive the Simulator section.

        Follows the selected channel when it is a sensor module; otherwise falls back to
        the displayed LM's first sensor-module channel so the view finder and RGB patch
        stay meaningful even when an LM channel is selected.

        Returns:
            The `(sensor_module, sensor_module_id)` pair, or `(None, None)` when the
            displayed LM has no sensor-module channel yet.
        """
        sm = self.resolve_sm_channel(self.channel)
        if sm is not None:
            return sm, self.channel
        sender_types = self.lm.buffer.channel_sender_types
        for channel in self.lm_channels():
            if sender_types.get(channel) == "SM":
                sm = self.resolve_sm_channel(channel)
                if sm is not None:
                    return sm, channel
        return None, None

    @staticmethod
    def channel_points(
        locations: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return the non-NaN-padded location rows for one buffer channel.

        Args:
            locations: The padded `(N, 3)` location buffer for a single channel.

        Returns:
            The `(M, 3)` rows whose first coordinate is not NaN, or an empty `(0, 3)`
            array when the buffer is empty or not yet shaped.
        """
        if locations.ndim != 2 or locations.shape[0] == 0 or locations.shape[1] < 3:
            return np.empty((0, 3))
        return locations[~np.isnan(locations[:, 0])]

    def aligned_feature(
        self, channel: str, attr: str
    ) -> npt.NDArray[np.float64] | None:
        """Return one buffer feature aligned row-for-row with a channel's valid points.

        The buffer pads every per-channel feature to the location length, so the rows
        kept by `channel_points` (non-NaN location) index the feature identically.

        Args:
            channel: The buffer input channel to read.
            attr: The feature name (e.g. `"hsv"` or `"object_id"`).

        Returns:
            The `(M, K)` feature rows for the channel's valid points, or `None` when the
            feature is absent, mis-shaped, or missing at any of those points.
        """
        channel_feats = self.lm.buffer.features.get(channel)
        if not channel_feats or attr not in channel_feats:
            return None
        arr = np.asarray(channel_feats[attr], dtype=float)
        locations = np.asarray(self.lm.buffer.locations[channel])
        if arr.ndim != 2 or arr.shape[0] != locations.shape[0]:
            return None
        valid = arr[~np.isnan(locations[:, 0])]
        if valid.size == 0 or np.isnan(valid).any():
            return None
        return valid

    def object_id_names(self, channel: str) -> dict[int, str]:
        """Map an LM channel's numeric object ids back to their object names.

        A learning-module channel carries each point's object as a numeric `object_id`
        feature (a hash of the object name). The source learning module knows the names
        of the objects it has learned, so re-hashing each known name inverts the feature
        and recovers the human-readable name shown in the legend, matching the text in
        the "Input Feature" inset.

        Args:
            channel: The buffer input channel being colored.

        Returns:
            A `{numeric object id: object name}` mapping, empty when the channel is not
            a learning-module channel feeding object ids.
        """
        source_lm = self.resolve_lm_channel(channel)
        if not isinstance(source_lm, EvidenceGraphLM):
            return {}
        return {
            sum(ord(c) for c in graph_id): graph_id
            for graph_id in source_lm.graph_memory.get_memory_ids()
        }

    def channel_groups(
        self, channel: str, pts: npt.NDArray[np.float64]
    ) -> list[tuple[npt.NDArray[np.float64], object, str | None]]:
        """Resolve the per-channel coloring for a point cloud.

        Patch channels carry an `hsv` feature, so their points are colored by hue with
        no legend. Learning-module channels instead carry an `object_id` feature, so
        their points are grouped and colored by object id with a legend. When both are
        present `hsv` wins; when neither is, the points fall back to the color cycle.

        Args:
            channel: The buffer input channel being drawn.
            pts: The channel's `(M, 3)` valid points (as returned by `channel_points`).

        Returns:
            The `(points, color, label)` groups to draw for this channel.
        """
        hsv = self.aligned_feature(channel, "hsv")
        if hsv is not None and hsv.shape[0] == pts.shape[0] and hsv.shape[1] >= 3:
            colors = mcolors.hsv_to_rgb(np.clip(hsv[:, :3], 0.0, 1.0))
            return [(pts, colors, None)]

        object_id = self.aligned_feature(channel, "object_id")
        if object_id is not None and object_id.shape[0] == pts.shape[0]:
            ids = object_id[:, 0]
            names = self.object_id_names(channel)
            unique_ids = np.unique(ids)
            cmap = plt.get_cmap("tab10" if len(unique_ids) <= 10 else "tab20")
            groups: list[tuple[npt.NDArray[np.float64], object, str | None]] = []
            for i, uid in enumerate(unique_ids):
                selected = pts[ids == uid]
                label = names.get(int(uid), f"object {int(uid)}")
                groups.append((selected, cmap(i % cmap.N), label))
            return groups

        return [(pts, None, None)]


class FeatureInset:
    """A figure-level "Input Feature" corner inset over a host panel.

    Owns its axis, white border box, and title independently of any gridspec, so it
    survives the host axes being removed and re-added. It repositions when its host
    rectangle moves and re-creates its axis only when the matplotlib projection flips
    between 3D and non-3D (an axis cannot switch projection in place).

    `draw` renders the selected channel's live feature into the inset, reading the
    channel resolution from a `ChannelView`: a 2D sensor module's detected edge, a 3D
    sensor module's local surface and normal, or the name of the object passed on a
    learning-module channel.
    """

    def __init__(self, fig: Figure, channel_view: ChannelView) -> None:
        """Initialize an empty inset bound to a figure and channel view.

        Args:
            fig: The figure the inset draws on.
            channel_view: The selection and channel resolution the drawn feature reads.
        """
        self.fig = fig
        self.channel_view = channel_view
        self.ax: Axes | None = None
        self.projection: str | None = None
        self._border = None
        self._title = None
        self._rect: tuple[float, ...] | None = None

    def ensure(self, projection: str, rect: list[float]) -> Axes:
        """Return the inset axis with the requested projection, positioned at `rect`.

        Args:
            projection: The semantic mode `"2d"`, `"3d"`, or `"text"`.
            rect: The `[left, bottom, width, height]` figure-coordinate rectangle.

        Returns:
            The inset axis.
        """
        self._ensure_frame(rect)
        mpl_projection = "3d" if projection == "3d" else None
        current_mpl = "3d" if self.projection == "3d" else None
        if self.ax is not None and current_mpl == mpl_projection:
            self.ax.set_position(rect)
            self.projection = projection
            return self.ax
        if self.ax is not None:
            self.ax.remove()
        self.ax = self.fig.add_axes(rect, projection=mpl_projection, zorder=10)
        self.projection = projection
        return self.ax

    def _ensure_frame(self, rect: list[float]) -> None:
        """Create or reposition the inset's white background, border, and title.

        The background and border are a single figure-level rectangle covering the
        inset rect, so they always align regardless of the inset's projection (a 3D
        axis renders its scene within a smaller region than its bounding box, so an
        axis-level frame would not match the visible box). The inset axis is transparent
        and sits above this rectangle; the title is a figure-level label above the box.

        Args:
            rect: The `[left, bottom, width, height]` figure-coordinate rectangle.
        """
        left, bottom, width, height = rect
        if self._border is None:
            self._border = plt.Rectangle(
                (left, bottom),
                width,
                height,
                transform=self.fig.transFigure,
                facecolor="white",
                edgecolor="black",
                linewidth=1.0,
                zorder=9,
            )
            self.fig.add_artist(self._border)
            self._title = self.fig.text(
                left + width / 2,
                bottom + height + 0.005,
                "Input Feature",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        elif self._rect != tuple(rect):
            self._border.set_bounds(left, bottom, width, height)
            self._title.set_position((left + width / 2, bottom + height + 0.005))
        self._rect = tuple(rect)

    def clear(self) -> Axes:
        """Clear the inset to a transparent, axis-off surface.

        Returns:
            The cleared inset axis.
        """
        self.ax.cla()
        self.ax.set_axis_off()
        self.ax.patch.set_visible(False)
        return self.ax

    def remove(self) -> None:
        """Remove the inset axis, border, and title from the figure."""
        for artist in (self.ax, self._border, self._title):
            if artist is not None:
                artist.remove()
        self.ax = self._border = self._title = None
        self.projection = self._rect = None

    def draw(self, channel: str | None, rect: list[float]) -> None:
        """Draw one channel's live input feature into the inset.

        The content depends on the channel's source: a 2D sensor module draws its
        detected edge, a 3D sensor module draws its local surface and normal, and a
        learning-module channel shows the name of the object currently being passed.

        Args:
            channel: The channel whose live feature is drawn.
            rect: The `[left, bottom, width, height]` inset rectangle.
        """
        sender_type = (
            self.channel_view.lm.buffer.channel_sender_types.get(channel)
            if channel is not None
            else None
        )
        if sender_type == "SM":
            sm = self.channel_view.resolve_sm_channel(channel)
            self._draw_from_sm(rect, sm)
        elif channel is not None and sender_type == "LM":
            self._draw_lm_name(rect, channel)
        else:
            self.ensure("text", rect)
            self._draw_message("no channel")

    def _draw_from_sm(self, rect: list[float], sm: SensorModule | None) -> None:
        """Draw a sensor module's detected feature, colored by the sensed hsv.

        For a 3D sensor module the inset is a small 3D axis showing the local surface as
        a tilted square plane with an arrow along its outward normal. For a 2D sensor
        module it is a flat axis showing the detected edge as a line. Off-object or
        degraded observations carry no pose, so every access is guarded and the inset
        shows a message instead.

        Args:
            rect: The `[left, bottom, width, height]` inset rectangle.
            sm: The sensor module feeding the channel.
        """
        is_2d = isinstance(sm, TwoDSensorModule)
        self.ensure("2d" if is_2d else "3d", rect)
        processed = sm.processed_obs if sm is not None else []
        if not processed:
            self._draw_message("no pose")
            return
        obs = processed[-1]
        morph, non_morph = (
            obs["morphological_features"],
            obs["non_morphological_features"],
        )
        pose_vectors = morph.get("pose_vectors")
        if pose_vectors is None:
            self._draw_message("off object")
            return
        pose_vectors = np.asarray(pose_vectors)

        hsv = non_morph.get("hsv")
        if hsv is not None:
            color = mcolors.hsv_to_rgb(np.clip(np.asarray(hsv, dtype=float), 0.0, 1.0))
        else:
            color = np.array([0.5, 0.5, 0.5])

        if is_2d:
            pose_fully_defined = bool(morph.get("pose_fully_defined", False))
            self._draw_edge(pose_vectors, pose_fully_defined, color)
        else:
            rotation = sm.state.rotation if sm.state is not None else None
            self._draw_surface(pose_vectors, color, rotation)

    def _draw_lm_name(self, rect: list[float], channel: str) -> None:
        """Show the name of the object being passed on a learning-module channel.

        Args:
            rect: The `[left, bottom, width, height]` inset rectangle.
            channel: The learning-module channel.
        """
        self.ensure("text", rect)
        ax = self.clear()
        source_lm = self.channel_view.resolve_lm_channel(channel)
        name = "-"
        if source_lm is not None:
            mlh = source_lm.get_current_mlh()
            graph_id = mlh.get("graph_id") if mlh else None
            if graph_id and graph_id != "no_observations_yet":
                name = str(graph_id)
        ax.text(0.5, 0.5, name, ha="center", va="center", transform=ax.transAxes)

    def _draw_message(self, message: str) -> None:
        """Clear the inset and show a centered message.

        Args:
            message: The text to display (e.g. "off object").
        """
        ax = self.clear()
        text = ax.text2D if self.projection == "3d" else ax.text
        text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)

    def _draw_surface(
        self,
        pose_vectors: npt.NDArray[np.float64],
        color: npt.NDArray[np.float64],
        rotation: qt.quaternion | None,
    ) -> None:
        """Draw the local 3D surface as a tilted square with its outward normal.

        The pose vectors arrive in the world frame, so they are first rotated into the
        sensor's local frame by the inverse of the sensor's world rotation. The square
        then lies in the tangent plane spanned by the two principal-curvature directions
        and is colored by the sensed hsv; the arrow points along the surface normal.

        Args:
            pose_vectors: The `(3, 3)` pose vectors (normal, then two tangents).
            color: The face color (the sensed hsv as RGB, or gray when absent).
            rotation: The sensor's world rotation, used to map the pose vectors into the
                agent's frame; `None` leaves them in the world frame.
        """
        ax = self.clear()
        if rotation is not None:
            pose_vectors = qt.rotate_vectors(rotation.inverse(), pose_vectors)
        normal = unit(pose_vectors[0])
        tangent_u = unit(pose_vectors[1])
        tangent_v = unit(pose_vectors[2])
        corners = np.array(
            [
                0.5 * (tangent_u + tangent_v),
                0.5 * (tangent_u - tangent_v),
                0.5 * (-tangent_u - tangent_v),
                0.5 * (-tangent_u + tangent_v),
            ]
        )
        ax.add_collection3d(
            Poly3DCollection(
                [corners], facecolors=[color], edgecolors="black", linewidths=0.5
            )
        )
        ax.quiver(
            0,
            0,
            0,
            normal[0],
            normal[1],
            normal[2],
            length=0.8,
            color="black",
            linewidth=2,
            arrow_length_ratio=0.3,
        )
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=90, azim=-90)

    def _draw_edge(
        self,
        pose_vectors: npt.NDArray[np.float64],
        pose_fully_defined: bool,
        color: npt.NDArray[np.float64],
    ) -> None:
        """Draw the detected 2D edge as a centered line colored by the sensed hsv.

        Args:
            pose_vectors: The `(3, 3)` pose vectors (normal, edge tangent, edge perp).
            pose_fully_defined: Whether an edge defines the pose this step.
            color: The line color (the sensed hsv as RGB, or gray when absent).
        """
        ax = self.clear()
        tangent = np.asarray(pose_vectors[1][:2], dtype=float)
        norm = np.linalg.norm(tangent)
        if not pose_fully_defined or norm < 1e-9:
            ax.text(
                0.5,
                0.5,
                "No edge detected",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return
        vec = tangent / norm * 0.4
        ax.plot(
            [0.5 - vec[0], 0.5 + vec[0]],
            [0.5 - vec[1], 0.5 + vec[1]],
            color=color,
            lw=3,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
