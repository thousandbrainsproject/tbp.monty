# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.gridspec import GridSpecFromSubplotSpec

from tbp.monty.frameworks.models.two_d_sensor_module import TwoDSensorModule
from tbp.monty.frameworks.plotters.helpers import (
    ChannelView,
    EvidenceHistory,
    FeatureInset,
    corner_rect,
    draw_2d_segments,
    draw_buffer_series,
    is_3d,
    planar_style,
)
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.frameworks.utils.plot_utils import add_patch_outline_to_view_finder

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from tbp.monty.frameworks.agents import AgentID
    from tbp.monty.frameworks.models.abstract_monty_classes import (
        AgentObservations,
        Observations,
        SensorModule,
    )
    from tbp.monty.frameworks.models.object_model import GraphObjectModel


class SimulatorPanel:
    """The Simulator section: a view finder above the RGB patch the sensor sees.

    Owns its two axes within the figure's left column and redraws them each frame from
    the driving sensor module's observations. Which sensor module drives the section is
    resolved by the plotter (it follows the selected channel, or the displayed LM's
    first sensor-module channel) and passed in per draw.
    """

    def __init__(self, fig: Figure, spec) -> None:
        """Lay out the view-finder and RGB-patch axes in the Simulator column.

        Args:
            fig: The figure to draw on.
            spec: The Simulator column's gridspec subplot spec.
        """
        self.fig = fig
        sim_grid = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=spec, height_ratios=[2, 1], hspace=0.3
        )
        self.ax_view = fig.add_subplot(sim_grid[0, 0])
        self.ax_rgb = fig.add_subplot(sim_grid[1, 0])

    def draw(
        self,
        observations: Observations,
        sm: SensorModule | None,
        sm_id: str | None,
    ) -> None:
        """Draw the view finder and RGB patch for the driving sensor module.

        Args:
            observations: The observations from the most recent step.
            sm: The sensor module driving the section, or `None` when the displayed LM
                has no sensor-module channel yet.
            sm_id: The id of that sensor module, or `None`.
        """
        agent_id = (
            self._resolve_obs_agent_id(observations, sm_id)
            if sm_id is not None
            else None
        )
        agent_obs: AgentObservations = (
            observations.get(agent_id, {}) if agent_id is not None else {}
        )
        self._draw_view_finder(agent_obs, sm, sm_id)
        self._draw_rgb_patch(agent_obs, sm_id)

    @staticmethod
    def _resolve_obs_agent_id(
        observations: Observations, sensor_module_id: str
    ) -> AgentID | None:
        """Find the agent id whose observations contain the given sensor module.

        Args:
            observations: The observations from the most recent step.
            sensor_module_id: The sensor module id to search for.

        Returns:
            The matching agent id, or `None` if no agent carries the sensor module.
        """
        for agent_id, agent_observations in observations.items():
            if sensor_module_id in agent_observations:
                return agent_id
        return None

    def _draw_view_finder(
        self,
        agent_obs: AgentObservations,
        sm: SensorModule | None,
        sm_id: str | None,
    ) -> None:
        """Draw the view finder, outlining the patch when a pixel location is known.

        Args:
            agent_obs: The observing agent's per-sensor observations this step.
            sm: The sensor module driving the Simulator (for its raw-pixel telemetry).
            sm_id: The id of that sensor module, used to read its patch observation.
        """
        ax = self.ax_view
        ax.cla()
        ax.set_title("View finder")
        ax.set_axis_off()
        if SensorID("view_finder") not in agent_obs:
            return
        rgba = np.asarray(agent_obs[SensorID("view_finder")]["rgba"])
        raw = sm._snapshot_telemetry.raw_observations if sm is not None else []
        patch_obs = agent_obs.get(sm_id) if sm_id is not None else None
        has_outline = (
            len(raw) > 0
            and "pixel_loc" in raw[-1]
            and patch_obs is not None
            and "depth" in patch_obs
        )
        if has_outline:
            patch_size = np.asarray(patch_obs["depth"]).shape[0]
            rgba = add_patch_outline_to_view_finder(
                rgba, np.array(raw[-1]["pixel_loc"]), patch_size
            )
        ax.imshow(rgba, zorder=-99)
        if not has_outline:
            shape = rgba.shape
            ax.add_patch(
                plt.Rectangle(
                    (shape[1] * 4.5 // 10, shape[0] * 4.5 // 10),
                    shape[1] / 10,
                    shape[0] / 10,
                    fc="none",
                    ec="white",
                )
            )

    def _draw_rgb_patch(self, agent_obs: AgentObservations, sm_id: str | None) -> None:
        """Draw what the sensor module sees, preferring RGB over depth.

        Args:
            agent_obs: The observing agent's per-sensor observations this step.
            sm_id: The id of the sensor module driving the Simulator.
        """
        ax = self.ax_rgb
        ax.cla()
        ax.set_axis_off()
        patch = agent_obs.get(sm_id) if sm_id is not None else None
        if patch is not None and "rgba" in patch:
            ax.imshow(np.asarray(patch["rgba"]))


class MontyPanel:
    """The Monty column: the selected channel's main graph and its feature inset.

    Owns the Monty column's single physical region, the axis-layout state machine, and
    the "Input Feature" corner inset stacked over it. During an exploratory step it
    draws the selected channel's buffered points (a flat edge cloud for a 2D sensor
    module, else a 3D cube with three head-on projections); during a matching step it
    draws the displayed LM's most likely hypothesis graph and location marker; otherwise
    a centered placeholder. Axes are rebuilt only when the layout or projection changes
    (a matplotlib axis cannot switch between 2D and 3D in place), to avoid flicker. The
    selection and per-channel features come from a `ChannelView`.
    """

    def __init__(self, fig: Figure, spec, channel_view: ChannelView) -> None:
        """Bind the Monty column to its figure region and selection.

        Lays out the initial single 2D axis, matching the layout the placeholder and
        inference 2D views start from.

        Args:
            fig: The figure to draw on.
            spec: The Monty column's gridspec subplot spec.
            channel_view: The selected LM/channel and per-channel feature accessors.
        """
        self.fig = fig
        self.spec = spec
        self.channel_view = channel_view
        self._ax = fig.add_subplot(spec)
        self._mode: str | None = "single"
        self._projection: str | None = None
        self._proj_axes: list[Axes] = []
        self._inset: FeatureInset | None = None

    def draw_placeholder(self, message: str) -> None:
        """Draw a centered placeholder message in the Monty panel.

        Args:
            message: The text to display.
        """
        ax = self._ensure_projection(None)
        ax.cla()
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            wrap=True,
            transform=ax.transAxes,
        )

    def draw_selected_channel(self, channel: str, pts: npt.NDArray[np.float64]) -> None:
        """Draw the selected channel's buffered points (exploratory step).

        A 2D sensor-module channel is drawn as a flat edge cloud mirroring the inference
        2D view; every other channel is drawn as a 3D cube with three head-on
        projections, since a partially built 3D graph may look planar by chance.

        Args:
            channel: The selected input channel.
            pts: The channel's `(M, 3)` non-padded location points.
        """
        if isinstance(self.channel_view.resolve_sm_channel(channel), TwoDSensorModule):
            self._draw_selected_channel_2d(channel, pts)
            return
        main_ax, proj_axes = self._ensure_projected()
        groups = self.channel_view.channel_groups(channel, pts)
        draw_buffer_series(
            main_ax,
            proj_axes,
            groups,
            title="",
            title_fontsize=None,
        )

    def draw_mlh(self) -> None:
        """Render the most likely hypothesis graph and its location marker.

        Uses the selected channel's stored graph when that channel is part of the MLH
        graph, otherwise the graph's first sensor-module channel. Falls back to a "No
        MLH" placeholder when there is no current hypothesis or the graph cannot be
        retrieved. Planar graphs are drawn as edge-oriented segments; all other graphs
        as a 3D point cloud.
        """
        lm = self.channel_view.lm
        graph = None
        mlh = lm.get_current_mlh()
        if mlh and mlh.get("graph_id") not in (None, "no_observations_yet"):
            graph_id = mlh["graph_id"]
            if graph_id in lm.graph_memory.get_memory_ids():
                channel = self._mlh_channel(graph_id)
                if channel is not None:
                    graph = lm.graph_memory.get_graph(graph_id, channel)

        if graph is None or getattr(graph, "pos", None) is None:
            self.draw_placeholder("No MLH")
            return
        pos = np.asarray(graph.pos)
        if len(pos) == 0:
            self.draw_placeholder("No MLH")
            return

        color = self._mlh_marker_color(mlh, lm.object_evidence_threshold)
        if is_3d(pos):
            ax = self._ensure_projection("3d")
            self._show_mlh_3d(ax, mlh, pos, color)
        else:
            ax = self._ensure_projection(None)
            self._show_mlh_2d(ax, mlh, graph, pos, color)

    def draw_feature_inset(self) -> None:
        """Draw the Monty section's "Input Feature" inset for the selected channel."""
        if self._inset is None:
            self._inset = FeatureInset(self.fig, self.channel_view)
        monty = self.spec.get_position(self.fig)
        rect = corner_rect(monty, width_frac=0.32, height=0.18, top_pad=0.07)
        self._inset.draw(self.channel_view.channel, rect)

    def _clear_axes(self) -> None:
        """Remove the main Monty axis and any projection axes below it."""
        self._ax.remove()
        for ax in self._proj_axes:
            ax.remove()
        self._proj_axes = []

    def _ensure_projection(self, projection: str | None) -> Axes:
        """Recreate the Monty axis as a single panel with the given projection.

        A matplotlib axis cannot switch between 2D and 3D in place, so the axis is
        removed and re-added whenever the layout or projection changes.

        Args:
            projection: The desired projection (`"3d"` or `None` for 2D).

        Returns:
            The current single Monty axis with the requested projection.
        """
        if self._mode == "single" and self._projection == projection:
            return self._ax
        self._clear_axes()
        self._ax = self.fig.add_subplot(self.spec, projection=projection)
        self._mode = "single"
        self._projection = projection
        return self._ax

    def _ensure_projected(self) -> tuple[Axes, list[Axes]]:
        """Lay the Monty column out as a 3D cloud over a row of three projections.

        Returns:
            The 3D main axis and the three 2D projection axes (XY, XZ, YZ).
        """
        if self._mode == "projected":
            return self._ax, self._proj_axes
        self._clear_axes()
        grid = GridSpecFromSubplotSpec(
            2,
            3,
            subplot_spec=self.spec,
            height_ratios=[3, 1],
            hspace=0.3,
            wspace=0.35,
        )
        self._ax = self.fig.add_subplot(grid[0, :], projection="3d")
        self._proj_axes = [self.fig.add_subplot(grid[1, j]) for j in range(3)]
        self._mode = "projected"
        self._projection = "3d"
        return self._ax, self._proj_axes

    def _draw_selected_channel_2d(
        self, channel: str, pts: npt.NDArray[np.float64]
    ) -> None:
        """Draw a 2D sensor-module channel's buffer as a planar edge cloud.

        Mirrors the inference 2D MLH view: a single flat rectilinear axis of
        hsv-colored dots with dashed segments along the edge tangent where the pose is
        fully defined, rather than the 3D cube and projections used for 3D channels.

        Args:
            channel: The selected 2D sensor-module channel.
            pts: The channel's `(M, 3)` non-padded location points.
        """
        ax = self._ensure_projection(None)
        ax.cla()
        x, y = pts[:, 0], pts[:, 1]
        colors, edge_mask, tangents = planar_style(
            len(x),
            self.channel_view.aligned_feature(channel, "hsv"),
            self.channel_view.aligned_feature(channel, "pose_fully_defined"),
            self.channel_view.aligned_feature(channel, "pose_vectors"),
        )
        draw_2d_segments(ax, x, y, colors, edge_mask, tangents)
        ax.set_title("")

    def _mlh_channel(self, graph_id: str) -> str | None:
        """Choose which channel's stored graph to render for the MLH.

        Args:
            graph_id: The MLH graph id.

        Returns:
            The selected channel when it is part of the graph, else the graph's first
            sensor-module channel, else `None` when the graph has no sensor channel.
        """
        lm = self.channel_view.lm
        channels = lm.get_input_channels_in_graph(graph_id)
        if self.channel_view.channel in channels:
            return self.channel_view.channel
        sender_types = lm.buffer.channel_sender_types
        sm_channels = [c for c in channels if sender_types.get(c) == "SM"]
        return sm_channels[0] if sm_channels else None

    @staticmethod
    def _mlh_marker_color(mlh: dict, evidence_threshold: float | None) -> str:
        """Color the MLH marker by whether its evidence clears the threshold.

        Args:
            mlh: The current most likely hypothesis.
            evidence_threshold: The object evidence threshold, or `None`.

        Returns:
            `"red"` when above threshold (or threshold unknown), else `"gray"`.
        """
        if evidence_threshold is None:
            return "red"
        return "red" if mlh["evidence"] > evidence_threshold else "gray"

    def _show_mlh_3d(
        self, ax: Axes, mlh: dict, pos: npt.NDArray[np.float64], mlh_color: str
    ) -> None:
        """Render a 3D graph as a point cloud with the MLH location marked.

        Args:
            ax: The 3D Monty axis.
            mlh: The current most likely hypothesis.
            pos: The graph node positions, shape `(N, 3)`.
            mlh_color: The MLH location marker color.
        """
        ax.cla()
        ax.scatter(pos[:, 1], pos[:, 0], pos[:, 2], c="black", s=2)
        ax.scatter(
            mlh["location"][1],
            mlh["location"][0],
            mlh["location"][2],
            c=mlh_color,
            s=15,
        )
        ax.set_title(f"MLH ({mlh['graph_id']})")
        ax.set_axis_off()
        ax.set_aspect("equal")

    def _show_mlh_2d(
        self,
        ax: Axes,
        mlh: dict,
        graph: GraphObjectModel,
        pos: npt.NDArray[np.float64],
        mlh_color: str,
    ) -> None:
        """Render a 2D SM graph as hsv-colored, edge-oriented segments.

        Args:
            ax: The 2D Monty axis.
            mlh: The current most likely hypothesis.
            graph: The MLH graph object model.
            pos: The graph node positions, shape `(N, >=2)`.
            mlh_color: The MLH location marker color.
        """
        ax.cla()
        x, y = pos[:, 0], pos[:, 1]
        fm = graph.feature_mapping
        graph_feature = {
            name: np.asarray(graph.get_values_for_feature(name))
            for name in ("hsv", "pose_fully_defined", "pose_vectors")
            if name in fm
        }
        colors, edge_mask, tangents = planar_style(
            len(x),
            graph_feature.get("hsv"),
            graph_feature.get("pose_fully_defined"),
            graph_feature.get("pose_vectors"),
        )
        draw_2d_segments(ax, x, y, colors, edge_mask, tangents)
        ax.plot(
            mlh["location"][0],
            mlh["location"][1],
            "x",
            color=mlh_color,
            markersize=8,
            markeredgewidth=2,
            zorder=3,
        )
        ax.set_title(f"MLH ({mlh['graph_id']})")


class DetailsPanel:
    """The Details column: per-channel buffer stacks, or inference line plots.

    Owns the Details column's single physical region and its mode state machine: a
    per-channel buffer grid during exploratory steps, the displayed LM's evidence and
    number-of-hypotheses line plots during matching steps, or a centered placeholder.
    Axes are rebuilt only when the active mode or channel count changes, to avoid
    flicker. The selection and per-channel features come from a `ChannelView` and the
    accumulated line-plot series from an `EvidenceHistory`. The "Input Feature" inset
    stacked over each buffer plot is a `FeatureInset`, the same widget the Monty section
    uses.
    """

    def __init__(
        self,
        fig: Figure,
        spec,
        channel_view: ChannelView,
        history: EvidenceHistory,
    ) -> None:
        """Bind the Details column to its figure region and data sources.

        Args:
            fig: The figure to draw on.
            spec: The Details column's gridspec subplot spec.
            channel_view: The selected LM/channel and per-channel feature accessors.
            history: The per-LM evidence and hypothesis-count history.
        """
        self.fig = fig
        self.spec = spec
        self.channel_view = channel_view
        self.history = history
        self._insets: dict[str, FeatureInset] = {}
        self._axes: list[Axes] = []
        self._proj_axes: list[list[Axes]] = []
        self._mode: str | None = None
        self._channel_count: int | None = None
        self._inference_legend = None
        self._evidence_steps: list[int] = []
        self._evidence_history: dict[str, list[float]] = {}
        self._num_hyp_history: dict[str, list[float]] = {}
        self._burst_steps: list[int] = []

    def draw_buffer_grid(
        self, channels: list[str], points: dict[str, npt.NDArray[np.float64]]
    ) -> None:
        """Draw one stacked plot group per channel's buffer (exploratory step).

        Args:
            channels: The channel ids with at least one location.
            points: The per-channel non-padded location points.
        """
        self._ensure_grid(len(channels))
        self._sync_insets(channels)
        for main_ax, proj_axes, channel in zip(self._axes, self._proj_axes, channels):
            groups = self.channel_view.channel_groups(channel, points[channel])
            draw_buffer_series(
                main_ax,
                proj_axes,
                groups,
                title=str(channel),
                title_fontsize=8,
                show_ticks=False,
            )
            cell = main_ax.get_position(self.fig)
            height = (cell.y1 - cell.y0) * 0.45
            rect = corner_rect(cell, width_frac=0.4, height=height)
            self._insets[channel].draw(channel, rect)

    def draw_inference(self) -> None:
        """Draw the displayed LM's evidence and hypothesis-count line plots."""
        self._select_active_history()
        self._ensure_lines()
        self._draw_object_series(
            self._evidence_ax,
            self._evidence_history,
            "Highest evidence per object",
            "evidence",
        )
        self._draw_object_series(
            self._num_hyp_ax,
            self._num_hyp_history,
            "Number of hypotheses per object",
            "hypotheses",
        )
        self._draw_inference_legend()

    def draw_placeholder(self, message: str) -> None:
        """Draw a centered placeholder message spanning the Details column.

        Args:
            message: The text to display.
        """
        if self._mode != "placeholder":
            self._clear_insets()
            self._clear_axes()
            self._axes = [self.fig.add_subplot(self.spec)]
            self._mode = "placeholder"
            self._channel_count = None

        ax = self._axes[0]
        ax.cla()
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            wrap=True,
            transform=ax.transAxes,
        )

    def _clear_axes(self) -> None:
        """Remove every main and projection axis in the Details column."""
        for ax in self._axes:
            ax.remove()
        for axes in self._proj_axes:
            for ax in axes:
                ax.remove()
        self._axes = []
        self._proj_axes = []

    def _sync_insets(self, channels: list[str]) -> None:
        """Match the feature insets to the channels currently stacked in the column.

        Drops insets for channels no longer shown and creates one per new channel, so
        each stacked Details panel carries the same "Input Feature" inset as the Monty
        section.

        Args:
            channels: The channel ids stacked in the Details column this frame.
        """
        wanted = set(channels)
        for channel in list(self._insets):
            if channel not in wanted:
                self._insets.pop(channel).remove()
        for channel in channels:
            if channel not in self._insets:
                self._insets[channel] = FeatureInset(self.fig, self.channel_view)

    def _clear_insets(self) -> None:
        """Remove every feature inset (when leaving the per-channel layout)."""
        for inset in self._insets.values():
            inset.remove()
        self._insets = {}

    def _ensure_grid(self, n: int) -> None:
        """Rebuild the column as `n` stacked per-channel plot groups.

        Each channel group is a 3D cloud over a row of three 2D projections. Rebuilds
        only when the channel count changes, to avoid flicker.

        Args:
            n: The number of stacked per-channel plot groups.
        """
        if self._mode == "grid" and self._channel_count == n:
            return
        self._clear_axes()
        outer = GridSpecFromSubplotSpec(n, 1, subplot_spec=self.spec, hspace=0.25)
        for i in range(n):
            inner = GridSpecFromSubplotSpec(
                2,
                3,
                subplot_spec=outer[i, 0],
                height_ratios=[3, 1],
                hspace=0.1,
                wspace=0.15,
            )
            self._axes.append(self.fig.add_subplot(inner[0, :], projection="3d"))
            self._proj_axes.append(
                [self.fig.add_subplot(inner[1, j]) for j in range(3)]
            )
        self._mode = "grid"
        self._channel_count = n

    def _ensure_lines(self) -> None:
        """Rebuild the column as two line plots with the legend between them."""
        if self._mode == "lines":
            return
        self._clear_insets()
        self._clear_axes()
        grid = GridSpecFromSubplotSpec(
            3,
            1,
            subplot_spec=self.spec,
            height_ratios=[1, 0.5, 1],
            hspace=0.4,
        )
        self._evidence_ax = self.fig.add_subplot(grid[0, 0])
        self._legend_ax = self.fig.add_subplot(grid[1, 0])
        self._legend_ax.set_axis_off()
        self._num_hyp_ax = self.fig.add_subplot(grid[2, 0])
        self._axes = [self._evidence_ax, self._legend_ax, self._num_hyp_ax]
        self._mode = "lines"
        self._channel_count = None

    def _select_active_history(self) -> None:
        """Point the line-plot history fields at the displayed LM's accumulated data."""
        lm_id = self.channel_view.lm.learning_module_id
        self._evidence_steps = self.history.steps_by_lm.setdefault(lm_id, [])
        self._evidence_history = self.history.evidence_by_lm.setdefault(lm_id, {})
        self._num_hyp_history = self.history.num_hyp_by_lm.setdefault(lm_id, {})
        self._burst_steps = self.history.burst_steps_by_lm.setdefault(lm_id, [])

    def _draw_object_series(
        self,
        ax: Axes,
        history: dict[str, list[float]],
        title: str,
        ylabel: str,
    ) -> None:
        """Redraw one per-object line plot with burst markers.

        Iterating `self._evidence_history` key order in both plots keeps each object's
        color consistent across them. The shared legend is drawn separately beneath the
        MLH panel by `_draw_inference_legend`.

        Args:
            ax: The axis to redraw.
            history: The per-object value history to plot.
            title: The plot title.
            ylabel: The y-axis label.
        """
        ax.cla()
        for graph_id in self._evidence_history:
            ax.plot(self._evidence_steps, history[graph_id], label=graph_id)
        for i, burst_step in enumerate(self._burst_steps):
            ax.axvline(
                burst_step,
                linestyle="--",
                color="red",
                label="burst" if i == 0 else None,
            )
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)

    def _draw_inference_legend(self) -> None:
        """Place the shared per-object legend between the two line plots.

        The legend would otherwise run off the figure's right edge, so it is drawn in a
        dedicated axis squeezed between the evidence and hypotheses plots, with the
        handles gathered from the line plots.
        """
        if self._inference_legend is not None:
            self._inference_legend.remove()
            self._inference_legend = None
        handles, labels = self._num_hyp_ax.get_legend_handles_labels()
        if not handles:
            return
        self._inference_legend = self._legend_ax.legend(
            handles,
            labels,
            loc="center",
            ncol=2,
            fontsize=10,
            borderaxespad=0.0,
        )
