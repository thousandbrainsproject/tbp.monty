# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Callable

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tbp.monty.frameworks.actions.action_samplers import ActionSampler
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.abstract_monty_classes import (
    AgentObservations,
    Monty,
    Observations,
)
from tbp.monty.frameworks.models.evidence_matching.burst_sampling import (
    BurstSamplingHypothesesUpdater,
)
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.two_d_sensor_module import TwoDSensorModule
from tbp.monty.frameworks.plotters.helpers import (
    draw_2d_segments,
    draw_buffer_series,
    is_3d,
    is_interactive_backend,
    planar_style,
    unit,
)
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.frameworks.utils.plot_utils import add_patch_outline_to_view_finder

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.random import RandomState

    from tbp.monty.frameworks.actions.actions import Action
    from tbp.monty.frameworks.agents import AgentID
    from tbp.monty.frameworks.models.abstract_monty_classes import (
        LearningModule,
        SensorModule,
    )
    from tbp.monty.frameworks.models.object_model import GraphObjectModel


class FeatureInset:
    """A figure-level "Input Feature" corner inset over a host panel.

    Owns its axis, white border box, and title independently of any gridspec, so it
    survives the host axes being removed and re-added. It repositions when its host
    rectangle moves and re-creates its axis only when the matplotlib projection flips
    between 3D and non-3D (an axis cannot switch projection in place).
    """

    def __init__(self, fig: Figure) -> None:
        """Initialize an empty inset bound to a figure.

        Args:
            fig: The figure the inset draws on.
        """
        self.fig = fig
        self.ax: Axes | None = None
        self.projection: str | None = None
        self._border = None
        self._title = None
        self._rect: tuple[float, float, float, float] | None = None

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
        ax.set_title("Patch")
        patch = agent_obs.get(sm_id) if sm_id is not None else None
        if patch is not None and "rgba" in patch:
            ax.imshow(np.asarray(patch["rgba"]))
            ax.set_title("Patch (RGB)")


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

    def _append(self, lm: LearningModule, step: int) -> None:
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
    def _in_burst(learning_module: LearningModule) -> bool | None:
        """Whether the LM's burst-sampling updater is currently in a burst.

        Args:
            learning_module: The learning module whose updater is inspected.

        Returns:
            `None` when the LM does not use a `BurstSamplingHypothesesUpdater` (so the
            caller can hide the info), otherwise whether a burst is in progress.
        """
        updater = learning_module.hypotheses_updater
        if not isinstance(updater, BurstSamplingHypothesesUpdater):
            return None
        return updater.sampling_burst_steps > 0


class ChannelView:
    """The selected learning module and input channel, with channel resolution.

    Owns the runtime selection — which learning module is displayed and which of its
    input channels is selected — and answers the resolution and feature queries the
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
            groups = []
            for i, uid in enumerate(unique_ids):
                selected = pts[ids == uid]
                label = names.get(int(uid), f"object {int(uid)}")
                groups.append((selected, cmap(i % cmap.N), label))
            return groups

        return [(pts, None, None)]


class LivePlotter:
    """Live plotter implementing the `Plotter` Protocol for training and inference.

    Renders a 3-section view of the *selected input channel* of the *displayed*
    learning module. Both the displayed learning module and the selected input channel
    are switchable at runtime through two cycling buttons placed under the `Step N`
    title; the learning module and channel are discovered from the model rather than
    configured by id.

    - Simulator: view finder and RGB patch of the sensor module feeding the selected
      (or, for an LM channel, the displayed LM's first sensor) channel.
    - Monty: the main graph of the selected channel, with an "Input Feature" corner
      inset showing the live feature on that channel (a 2D edge, a 3D surface, or, for a
      learning-module channel, the name of the object being passed). During an
      exploratory step the main graph is the channel's buffered points; during a
      matching step it is the most likely hypothesis graph plus its location marker.
    - Details: during an exploratory step, a per-channel stack of the *other* channels'
      buffers, each carrying the same "Input Feature" inset as the Monty section; during
      a matching step, the displayed LM's per-object evidence and number-of-hypotheses
      line plots.

    A non-interactive plotter exposes a `Speed` slider; an interactive plotter exposes
    one button per action the policy's sampler can produce (plus "End episode") and
    overrides the executed action via `override_action`.

    When the displayed learning module lacks the evidence-LM inference API, the Monty
    and Details matching-step panels degrade to a placeholder rather than raising.
    """

    def __init__(
        self,
        interactive: bool = False,
        max_delay: float = 2.0,
        figsize: tuple[float, float] = (16, 8),
    ) -> None:
        """Initialize the plotter.

        Args:
            interactive: Whether to render action buttons and override the executed
                action with the user's choice (otherwise render a speed slider).
            max_delay: Maximum non-interactive pause in seconds at the slowest speed.
            figsize: Figure size in inches.
        """
        # Turn interactive plotting off so the plotter controls when figures are
        # drawn and when execution blocks, via its own canvas event loop.
        plt.ioff()

        self.interactive = interactive
        self.max_delay = max_delay
        self.figsize = figsize

        self.fig = None

    def _lm_building_graph(self, lm: LearningModule) -> bool:
        """Whether a learning module is building a graph this step.

        The model's `step_type` is shared across all learning modules, so it can't
        distinguish per-LM phases in a heterarchy. On an exploratory step every LM is
        building. On a matching step (the default, and what partially-supervised
        heterarchy training stays on) an LM is still building its graph when it is one
        of the supervised LMs in a training episode: its buffer is converted to a graph
        from the ground-truth label at episode end. Every other LM is matching.

        Args:
            lm: The learning module whose phase decides which panels to draw.

        Returns:
            True when `lm` is building a graph (exploratory panels), False when it is
            matching (inference panels).
        """
        if self.model.step_type == "exploratory_step":
            return True
        return (
            self.model.experiment_mode is ExperimentMode.TRAIN
            and lm.learning_module_id in self.model.supervised_lm_ids
        )

    def _pause_seconds(self, speed: float) -> float | None:
        """Map a speed slider value in [0, 1] to a pause duration.

        Args:
            speed: The slider value; 1 = no delay, 0 = halt indefinitely.

        Returns:
            `None` to halt (speed 0), `0.0` for full speed (speed >= 1), else a
            positive pause bounded by `max_delay`.
        """
        if speed <= 0.0:
            return None
        if speed >= 1.0:
            return 0.0
        return self.max_delay * (1.0 - speed)

    def initialize(self, model: Monty) -> None:
        """Resolve the displayed LM and build the figure, axes, and widgets.

        Detects whether the displayed learning module supports the inference and buffer
        panels and resets the per-episode history. When interactive, derives the action
        buttons from the policy's action sampler. Then builds the matplotlib figure for
        this episode, closing the previous episode's figure first so figures don't
        accumulate across a multi-object / multi-rotation run. Must be called once per
        episode.

        Args:
            model: The Monty model whose sensor and learning modules are plotted.
        """
        self.model = model
        self._resolve_episode_targets(model)
        self._setup_interactive_sampler(model)
        self._reset_episode_state()
        self._build_figure()

    def _resolve_episode_targets(self, model: Monty) -> None:
        """Build the channel view selecting the default displayed LM and channel.

        Args:
            model: The Monty model whose learning modules are plotted.
        """
        self._channel_view = ChannelView(model)

    def _setup_interactive_sampler(self, model: Monty) -> None:
        """Derive the action sampler and button names from the policy.

        A non-interactive plotter clears the sampler state and returns.

        Args:
            model: The Monty model whose motor system exposes the policy.

        Raises:
            ValueError: When interactive but the policy exposes no usable sampler.
        """
        self._sampler: ActionSampler | None = None
        self._agent_id: str | None = None
        self._action_names: list[str] = []
        if not self.interactive:
            return
        policy = model.motor_system.policy
        if policy is None:
            raise ValueError(
                "Interactive plotter requires a policy exposing an action sampler, "
                "but the motor system's selector exposes no policy."
            )
        if not hasattr(policy, "action_sampler") or not hasattr(policy, "agent_id"):
            raise ValueError(
                "Interactive plotter requires a policy with an `action_sampler` "
                f"and `agent_id`; {type(policy).__name__} exposes neither."
            )
        self._sampler = policy.action_sampler
        self._agent_id = policy.agent_id
        self._action_names = list(self._sampler.action_names)
        if not self._action_names:
            raise ValueError(
                "Interactive plotter requires a policy whose action sampler can "
                "produce at least one action, but it produced none."
            )

    def _reset_episode_state(self) -> None:
        """Reset the per-episode history, selection, and axis-layout bookkeeping."""
        self._history = EvidenceHistory()
        self._evidence_steps: list[int] = []
        self._evidence_history: dict[str, list[float]] = {}
        self._num_hyp_history: dict[str, list[float]] = {}
        self._burst_steps: list[int] = []
        self._selected: str | None = None
        self._buttons: list[Button] = []
        self._slider: Slider | None = None
        self._lm_button: Button | None = None
        self._channel_button: Button | None = None
        self._monty_mode: str | None = "single"
        self._monty_projection: str | None = None
        self._monty_proj_axes: list[Axes] = []
        self._monty_inset: FeatureInset | None = None
        self._details_insets: dict[str, FeatureInset] = {}
        self._details_axes: list[Axes] = []
        self._details_proj_axes: list[list[Axes]] = []
        self._details_mode: str | None = None
        self._details_channel_count: int | None = None
        self._inference_legend = None
        self._last_observations: Observations | None = None
        self._last_step: int | None = None

    def _build_figure(self) -> None:
        """Build this episode's figure, axes, and widgets.

        Closes the previous episode's figure first so figures don't accumulate across a
        multi-object / multi-rotation run.

        Raises:
            RuntimeError: When interactive but the active backend cannot run the
                blocking event loop (e.g. the headless `Agg` backend).
        """
        if self.interactive and not is_interactive_backend():
            raise RuntimeError(
                "Interactive plotter requires a GUI matplotlib backend (e.g. TkAgg / "
                f"QtAgg); the current backend {mpl.get_backend()!r} cannot run "
                "the blocking event loop."
            )

        if self.fig is not None:
            plt.close(self.fig)

        self.fig = plt.figure(figsize=self.figsize)
        self.fig.subplots_adjust(
            bottom=0.16, top=0.9, left=0.04, right=0.97, wspace=0.25
        )
        outer = self.fig.add_gridspec(1, 3)
        self._sim_spec = outer[0, 0]
        self._monty_spec = outer[0, 1]
        self._details_spec = outer[0, 2]

        self._simulator = SimulatorPanel(self.fig, self._sim_spec)
        self._ax_monty = self.fig.add_subplot(self._monty_spec)
        self._draw_section_dividers()
        self._build_selector_buttons()

        if self.interactive:
            self._build_action_buttons()
        else:
            self._build_speed_slider()

        if is_interactive_backend():
            self.fig.show()

    def update(self, observations: Observations, step: int) -> None:
        """Draw the current state, never blocking for an action.

        The frame is stashed so selector-button clicks can repaint it without new
        observations. A non-interactive plotter applies its speed-slider pause/halt
        here; interactive blocking lives only in `override_action`.

        Args:
            observations: The observations from the most recent step.
            step: The index of the current step within the episode.
        """
        if self.fig is None:
            return
        self._last_observations = observations
        self._last_step = step
        self._history.accumulate(self.model.learning_modules, step)
        self._render(observations, step)
        if not self.interactive:
            self._apply_speed_pause()

    def _render(self, observations: Observations, step: int) -> None:
        """Draw every section for one frame.

        Args:
            observations: The observations to draw.
            step: The index of the current step within the episode.
        """
        self.fig.suptitle(f"Step {step}")
        if self._channel_view.ensure_channel():
            self._refresh_selector_labels()

        sm, sm_id = self._channel_view.simulator_sm()
        self._simulator.draw(observations, sm, sm_id)
        if self._lm_building_graph(self._channel_view.lm):
            self._draw_training()
        else:
            self._draw_inference()
        self._draw_feature_inset()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _redraw(self) -> None:
        """Repaint the last frame after a selection change, without re-accumulating.

        Called by the selector buttons so a new learning module or channel is visible
        immediately, even while a blocking event loop is running.
        """
        if self.fig is None or self._last_observations is None:
            return
        self._render(self._last_observations, self._last_step)

    def override_action(self, rng: RandomState) -> list[Action]:
        """Block until a button is clicked, then return the user's chosen action.

        The wait is guarded on `self._selected`, so a pre-set selection skips the event
        loop entirely. Selector-button clicks repaint the figure without setting
        `self._selected`, so they do not end the wait.

        Args:
            rng: The random state used to sample the chosen action.

        Returns:
            The actions to execute next, built from the user's button choice.

        Raises:
            StopIteration: When the user clicks "End episode", after setting any LMs
                that have not reached a terminal state to time_out so the episode logs
                cleanly.
        """
        while self._selected is None:
            self.fig.canvas.start_event_loop(0.1)
        selected, self._selected = self._selected, None
        if selected == "End episode":
            self.model.deal_with_time_out()
            raise StopIteration

        action_method: Callable[[str, RandomState], Action] = getattr(
            self._sampler, f"sample_{selected}"
        )

        return [action_method(self._agent_id, rng)]

    def close(self) -> None:
        """Close the final figure and drop widget references."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
        self._buttons = []
        self._slider = None
        self._lm_button = None
        self._channel_button = None
        self._simulator = None
        self._monty_inset = None
        self._details_insets = {}

    def _build_action_buttons(self) -> None:
        """Add one button per action name plus an "End episode" button."""
        labels = self._action_names + ["End episode"]
        n = len(labels)
        for i, label in enumerate(labels):
            # [left, bottom, width, height]
            ax_btn = self.fig.add_axes(
                [0.02 + i * (0.96 / n), 0.03, 0.96 / n - 0.01, 0.07]
            )
            btn = Button(ax_btn, label)
            btn.on_clicked(lambda _event, lbl=label: self._on_click(lbl))
            self._buttons.append(btn)

    def _on_click(self, label: str) -> None:
        """Record the clicked action and stop the blocking event loop.

        Args:
            label: The clicked button's label (an action name or "End episode").
        """
        self._selected = label
        with contextlib.suppress(Exception):
            self.fig.canvas.stop_event_loop()

    def _build_speed_slider(self) -> None:
        """Add the non-interactive speed slider."""
        ax_slider = self.fig.add_axes([0.07, 0.05, 0.86, 0.03])
        self._slider = Slider(ax_slider, "Speed", 0.0, 1.0, valinit=1.0)

    def _apply_speed_pause(self) -> None:
        """Pause (or halt) between steps according to the speed slider."""
        if self._slider is None:
            return
        delay = self._pause_seconds(float(self._slider.val))
        if delay is None:
            while float(self._slider.val) <= 0.0:
                self.fig.canvas.start_event_loop(0.1)
        elif delay > 0.0:
            plt.pause(delay)

    def _build_selector_buttons(self) -> None:
        """Add the displayed-LM and selected-channel cycling buttons.

        The buttons sit in the top margin above the Monty column, under the `Step N`
        title. Their clicks cycle the selection and repaint the figure but never set
        `self._selected`, so they are inert with respect to the action wait.
        """
        monty = self._monty_spec.get_position(self.fig)
        width = monty.x1 - monty.x0
        bottom, height = 0.91, 0.03
        lm_label, channel_label = self._channel_view.labels()

        ax_lm = self.fig.add_axes([monty.x0, bottom, width * 0.48, height])
        self._lm_button = Button(ax_lm, lm_label)
        self._lm_button.on_clicked(self._on_cycle_lm)

        ax_channel = self.fig.add_axes(
            [monty.x0 + width * 0.52, bottom, width * 0.48, height]
        )
        self._channel_button = Button(ax_channel, channel_label)
        self._channel_button.on_clicked(self._on_cycle_channel)

    def _refresh_selector_labels(self) -> None:
        """Update both selector button captions to the current selection."""
        lm_label, channel_label = self._channel_view.labels()
        if self._lm_button is not None:
            self._lm_button.label.set_text(lm_label)
        if self._channel_button is not None:
            self._channel_button.label.set_text(channel_label)

    def _on_cycle_lm(self, _event: object) -> None:
        """Advance to the next learning module and repaint.

        Args:
            _event: The matplotlib button event (unused).
        """
        self._channel_view.cycle_lm()
        self._refresh_selector_labels()
        self._redraw()

    def _on_cycle_channel(self, _event: object) -> None:
        """Advance to the next input channel of the displayed LM and repaint.

        Args:
            _event: The matplotlib button event (unused).
        """
        if not self._channel_view.cycle_channel():
            return
        self._refresh_selector_labels()
        self._redraw()

    def _draw_section_dividers(self) -> None:
        """Draw vertical separators in the gaps between the three column sections."""
        sim = self._sim_spec.get_position(self.fig)
        monty = self._monty_spec.get_position(self.fig)
        details = self._details_spec.get_position(self.fig)
        label_margin = 0.05
        gaps = [
            min((sim.x1 + monty.x0) / 2, monty.x0 - label_margin),
            min((monty.x1 + details.x0) / 2, details.x0 - label_margin),
        ]
        for x in gaps:
            self.fig.add_artist(
                Line2D(
                    [x, x],
                    [sim.y0, sim.y1],
                    transform=self.fig.transFigure,
                    color="0.7",
                    linewidth=1,
                )
            )

    def _draw_feature_inset(self) -> None:
        """Draw the Monty section's "Input Feature" inset for the selected channel."""
        if self._monty_inset is None:
            self._monty_inset = FeatureInset(self.fig)
        monty = self._monty_spec.get_position(self.fig)
        rect = self._corner_rect(monty, width_frac=0.32, height=0.18, top_pad=0.07)
        self._draw_channel_feature(self._monty_inset, self._channel_view.channel, rect)

    def _corner_rect(
        self,
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
            The `[left, bottom, width, height]` rectangle in the panel's top-left
            corner.
        """
        width = (bbox.x1 - bbox.x0) * width_frac
        top = bbox.y1 - top_pad
        return [bbox.x0 + 0.005, top - height, width, height]

    def _draw_channel_feature(
        self, inset: FeatureInset, channel: str | None, rect: list[float]
    ) -> None:
        """Draw one channel's live input feature into a corner inset.

        The content depends on the channel's source: a 2D sensor module draws its
        detected edge, a 3D sensor module draws its local surface and normal, and a
        learning-module channel shows the name of the object currently being passed.

        Args:
            inset: The inset to draw into.
            channel: The channel whose live feature is drawn.
            rect: The `[left, bottom, width, height]` inset rectangle.
        """
        sender_type = (
            self._channel_view.lm.buffer.channel_sender_types.get(channel)
            if channel is not None
            else None
        )
        if sender_type == "SM":
            sm = self._channel_view.resolve_sm_channel(channel)
            self._draw_feature_from_sm(inset, rect, sm)
        elif sender_type == "LM":
            self._draw_feature_lm_name(inset, rect, channel)
        else:
            inset.ensure("text", rect)
            self._draw_feature_message(inset, "no channel")

    def _draw_feature_from_sm(
        self, inset: FeatureInset, rect: list[float], sm: SensorModule | None
    ) -> None:
        """Draw a sensor module's detected feature, colored by the sensed hsv.

        For a 3D sensor module the inset is a small 3D axis showing the local surface as
        a tilted square plane with an arrow along its outward normal. For a 2D sensor
        module it is a flat axis showing the detected edge as a line. Off-object or
        degraded observations carry no pose, so every access is guarded and the inset
        shows a message instead.

        Args:
            inset: The inset to draw into.
            rect: The `[left, bottom, width, height]` inset rectangle.
            sm: The sensor module feeding the channel.
        """
        is_2d = isinstance(sm, TwoDSensorModule)
        inset.ensure("2d" if is_2d else "3d", rect)
        processed = sm.processed_obs if sm is not None else []
        if not processed:
            self._draw_feature_message(inset, "no pose")
            return
        obs = processed[-1]
        morph, non_morph = (
            obs["morphological_features"],
            obs["non_morphological_features"],
        )
        pose_vectors = morph.get("pose_vectors")
        if pose_vectors is None:
            self._draw_feature_message(inset, "off object")
            return
        pose_vectors = np.asarray(pose_vectors)

        hsv = non_morph.get("hsv")
        if hsv is not None:
            color = mcolors.hsv_to_rgb(np.clip(np.asarray(hsv, dtype=float), 0.0, 1.0))
        else:
            color = np.array([0.5, 0.5, 0.5])

        if is_2d:
            pose_fully_defined = bool(morph.get("pose_fully_defined", False))
            self._draw_feature_edge(inset, pose_vectors, pose_fully_defined, color)
        else:
            self._draw_feature_surface(inset, pose_vectors, color)

    def _draw_feature_lm_name(
        self, inset: FeatureInset, rect: list[float], channel: str
    ) -> None:
        """Show the name of the object being passed on a learning-module channel.

        Args:
            inset: The inset to draw into.
            rect: The `[left, bottom, width, height]` inset rectangle.
            channel: The learning-module channel.
        """
        inset.ensure("text", rect)
        ax = inset.clear()
        source_lm = self._channel_view.resolve_lm_channel(channel)
        name = "-"
        if source_lm is not None:
            mlh = source_lm.get_current_mlh()
            graph_id = mlh.get("graph_id") if mlh else None
            if graph_id and graph_id != "no_observations_yet":
                name = str(graph_id)
        ax.text(0.5, 0.5, name, ha="center", va="center", transform=ax.transAxes)

    def _draw_feature_message(self, inset: FeatureInset, message: str) -> None:
        """Clear an inset and show a centered message.

        Args:
            inset: The inset to draw into.
            message: The text to display (e.g. "off object").
        """
        ax = inset.clear()
        text = ax.text2D if inset.projection == "3d" else ax.text
        text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)

    def _draw_feature_surface(
        self,
        inset: FeatureInset,
        pose_vectors: npt.NDArray[np.float64],
        color: npt.NDArray[np.float64],
    ) -> None:
        """Draw the local 3D surface as a tilted square with its outward normal.

        The square lies in the tangent plane spanned by the two principal-curvature
        directions and is colored by the sensed hsv; the arrow points along the surface
        normal. The camera looks straight down `+z` (the XY-plane view), so that it
        mirrors the agent's pose in the viewfinder.

        Args:
            inset: The inset to draw into.
            pose_vectors: The `(3, 3)` pose vectors (normal, then two tangents).
            color: The face color (the sensed hsv as RGB, or gray when absent).
        """
        ax = inset.clear()
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

    def _draw_feature_edge(
        self,
        inset: FeatureInset,
        pose_vectors: npt.NDArray[np.float64],
        pose_fully_defined: bool,
        color: npt.NDArray[np.float64],
    ) -> None:
        """Draw the detected 2D edge as a centered line colored by the sensed hsv.

        Args:
            inset: The inset to draw into.
            pose_vectors: The `(3, 3)` pose vectors (normal, edge tangent, edge perp).
            pose_fully_defined: Whether an edge defines the pose this step.
            color: The line color (the sensed hsv as RGB, or gray when absent).
        """
        ax = inset.clear()
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

    def _clear_monty_axes(self) -> None:
        """Remove the main Monty axis and any projection axes below it."""
        self._ax_monty.remove()
        for ax in self._monty_proj_axes:
            ax.remove()
        self._monty_proj_axes = []

    def _ensure_monty_projection(self, projection: str | None) -> Axes:
        """Recreate the Monty axis as a single panel with the given projection.

        A matplotlib axis cannot switch between 2D and 3D in place, so the axis is
        removed and re-added whenever the layout or projection changes.

        Args:
            projection: The desired projection (`"3d"` or `None` for 2D).

        Returns:
            The current single Monty axis with the requested projection.
        """
        if self._monty_mode == "single" and self._monty_projection == projection:
            return self._ax_monty
        self._clear_monty_axes()
        self._ax_monty = self.fig.add_subplot(self._monty_spec, projection=projection)
        self._monty_mode = "single"
        self._monty_projection = projection
        return self._ax_monty

    def _ensure_monty_projected(self) -> tuple[Axes, list[Axes]]:
        """Lay the Monty column out as a 3D cloud over a row of three projections.

        Returns:
            The 3D main axis and the three 2D projection axes (XY, XZ, YZ).
        """
        if self._monty_mode == "projected":
            return self._ax_monty, self._monty_proj_axes
        self._clear_monty_axes()
        grid = GridSpecFromSubplotSpec(
            2,
            3,
            subplot_spec=self._monty_spec,
            height_ratios=[3, 1],
            hspace=0.3,
            wspace=0.35,
        )
        self._ax_monty = self.fig.add_subplot(grid[0, :], projection="3d")
        self._monty_proj_axes = [self.fig.add_subplot(grid[1, j]) for j in range(3)]
        self._monty_mode = "projected"
        self._monty_projection = "3d"
        return self._ax_monty, self._monty_proj_axes

    def _draw_monty_placeholder(self, message: str) -> None:
        """Draw a centered placeholder message in the Monty panel.

        Args:
            message: The text to display.
        """
        ax = self._ensure_monty_projection(None)
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

    def _clear_details_axes(self) -> None:
        """Remove every main and projection axis in the Details column."""
        for ax in self._details_axes:
            ax.remove()
        for axes in self._details_proj_axes:
            for ax in axes:
                ax.remove()
        self._details_axes = []
        self._details_proj_axes = []

    def _sync_details_insets(self, channels: list[str]) -> None:
        """Match the Details feature insets to the channels currently stacked there.

        Drops insets for channels no longer shown and creates one per new channel, so
        each stacked Details panel carries the same "Input Feature" inset as the Monty
        section.

        Args:
            channels: The channel ids stacked in the Details column this frame.
        """
        wanted = set(channels)
        for channel in list(self._details_insets):
            if channel not in wanted:
                self._details_insets.pop(channel).remove()
        for channel in channels:
            if channel not in self._details_insets:
                self._details_insets[channel] = FeatureInset(self.fig)

    def _clear_details_insets(self) -> None:
        """Remove every Details feature inset (when leaving the per-channel layout)."""
        for inset in self._details_insets.values():
            inset.remove()
        self._details_insets = {}

    def _ensure_details_grid(self, n: int) -> None:
        """Rebuild the Details column as `n` stacked per-channel plot groups.

        Each channel group is a 3D cloud over a row of three 2D projections. Rebuilds
        only when the channel count changes, to avoid flicker.

        Args:
            n: The number of stacked per-channel plot groups.
        """
        if self._details_mode == "grid" and self._details_channel_count == n:
            return
        self._clear_details_axes()
        outer = GridSpecFromSubplotSpec(
            n, 1, subplot_spec=self._details_spec, hspace=0.25
        )
        for i in range(n):
            inner = GridSpecFromSubplotSpec(
                2,
                3,
                subplot_spec=outer[i, 0],
                height_ratios=[3, 1],
                hspace=0.1,
                wspace=0.15,
            )
            self._details_axes.append(
                self.fig.add_subplot(inner[0, :], projection="3d")
            )
            self._details_proj_axes.append(
                [self.fig.add_subplot(inner[1, j]) for j in range(3)]
            )
        self._details_mode = "grid"
        self._details_channel_count = n

    def _ensure_details_lines(self) -> None:
        """Rebuild the Details column as two line plots with the legend between them."""
        if self._details_mode == "lines":
            return
        self._clear_details_insets()
        self._clear_details_axes()
        grid = GridSpecFromSubplotSpec(
            3,
            1,
            subplot_spec=self._details_spec,
            height_ratios=[1, 0.5, 1],
            hspace=0.4,
        )
        self._evidence_ax = self.fig.add_subplot(grid[0, 0])
        self._legend_ax = self.fig.add_subplot(grid[1, 0])
        self._legend_ax.set_axis_off()
        self._num_hyp_ax = self.fig.add_subplot(grid[2, 0])
        self._details_axes = [self._evidence_ax, self._legend_ax, self._num_hyp_ax]
        self._details_mode = "lines"
        self._details_channel_count = None

    def _draw_details_placeholder(self, message: str) -> None:
        """Draw a centered placeholder message spanning the Details column.

        Args:
            message: The text to display.
        """
        if self._details_mode != "placeholder":
            self._clear_details_insets()
            self._clear_details_axes()
            self._details_axes = [self.fig.add_subplot(self._details_spec)]
            self._details_mode = "placeholder"
            self._details_channel_count = None

        ax = self._details_axes[0]
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

    def _draw_training(self) -> None:
        """Draw the exploratory-step panels from the LM buffer.

        The Monty panel shows the selected channel's buffered points; the Details panel
        stacks one plot group per *other* channel. While the graph is still being built
        we cannot tell whether the points are planar, so every buffer view is drawn in
        3D with three head-on 2D projections beneath it. Degrades to a placeholder when
        there are no observations yet.
        """
        locations = self._channel_view.lm.buffer.locations
        points = {
            c: self._channel_view.channel_points(np.asarray(locs))
            for c, locs in locations.items()
        }
        channels = [c for c in points if points[c].size]
        if not channels:
            self._draw_monty_placeholder("no observations yet")
            self._draw_details_placeholder("no observations yet")
            return

        selected = self._channel_view.channel
        selected_pts = points.get(selected) if selected is not None else None
        if selected_pts is None or not selected_pts.size:
            label = selected if selected is not None else "channel"
            self._draw_monty_placeholder(f"no observations on {label}")
        else:
            self._draw_selected_channel(selected, selected_pts)

        others = [c for c in channels if c != selected]
        if others:
            self._draw_buffer_per_channel(others, points)
        else:
            self._draw_details_placeholder("no other channels")

    def _draw_selected_channel(
        self, channel: str, pts: npt.NDArray[np.float64]
    ) -> None:
        """Draw the selected channel's buffered points in the Monty panel.

        A 2D sensor-module channel is drawn as a flat edge cloud mirroring the inference
        2D view; every other channel is drawn as a 3D cube with three head-on
        projections, since a partially built 3D graph may look planar by chance.

        Args:
            channel: The selected input channel.
            pts: The channel's `(M, 3)` non-padded location points.
        """
        if isinstance(self._channel_view.resolve_sm_channel(channel), TwoDSensorModule):
            self._draw_selected_channel_2d(channel, pts)
            return
        main_ax, proj_axes = self._ensure_monty_projected()
        groups = self._channel_view.channel_groups(channel, pts)
        draw_buffer_series(
            main_ax,
            proj_axes,
            groups,
            title="",
            title_fontsize=None,
        )

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
        ax = self._ensure_monty_projection(None)
        ax.cla()
        x, y = pts[:, 0], pts[:, 1]
        colors, edge_mask, tangents = planar_style(
            len(x),
            self._channel_view.aligned_feature(channel, "hsv"),
            self._channel_view.aligned_feature(channel, "pose_fully_defined"),
            self._channel_view.aligned_feature(channel, "pose_vectors"),
        )
        draw_2d_segments(ax, x, y, colors, edge_mask, tangents)
        ax.set_title("")

    def _draw_buffer_per_channel(
        self, channels: list[str], points: dict[str, npt.NDArray[np.float64]]
    ) -> None:
        """Draw one Details plot group per channel's buffer.

        Args:
            channels: The channel ids with at least one location.
            points: The per-channel non-padded location points.
        """
        self._ensure_details_grid(len(channels))
        self._sync_details_insets(channels)
        for main_ax, proj_axes, channel in zip(
            self._details_axes, self._details_proj_axes, channels
        ):
            groups = self._channel_view.channel_groups(channel, points[channel])
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
            rect = self._corner_rect(cell, width_frac=0.4, height=height)
            self._draw_channel_feature(self._details_insets[channel], channel, rect)

    def _draw_inference(self) -> None:
        """Draw the matching-step panels (MLH graph and line plots).

        Renders the displayed LM's MLH graph and line plots from the per-LM evidence
        history accumulated in `update`. Degrades to placeholders when the displayed LM
        lacks the evidence-LM inference API.
        """
        if not self._channel_view.supports_evidence:
            lm_name = type(self._channel_view.lm).__name__
            placeholder = f"Inference view not available for {lm_name}"
            self._draw_monty_placeholder(placeholder)
            self._draw_details_placeholder(placeholder)
            return
        self._select_active_history()
        self._draw_mlh()
        self._draw_evidence_plots()
        self._draw_inference_legend()

    def _select_active_history(self) -> None:
        """Point the line-plot history fields at the displayed LM's accumulated data."""
        lm_id = self._channel_view.lm.learning_module_id
        self._evidence_steps = self._history.steps_by_lm.setdefault(lm_id, [])
        self._evidence_history = self._history.evidence_by_lm.setdefault(lm_id, {})
        self._num_hyp_history = self._history.num_hyp_by_lm.setdefault(lm_id, {})
        self._burst_steps = self._history.burst_steps_by_lm.setdefault(lm_id, [])

    def _draw_mlh(self) -> None:
        """Render the most likely hypothesis graph and its location marker.

        Uses the selected channel's stored graph when that channel is part of the MLH
        graph, otherwise the graph's first sensor-module channel. Falls back to a "No
        MLH" placeholder when there is no current hypothesis or the graph cannot be
        retrieved. Planar graphs are drawn as edge-oriented segments; all other graphs
        as a 3D point cloud.
        """
        lm = self._channel_view.lm
        graph = None
        mlh = lm.get_current_mlh()
        if mlh and mlh.get("graph_id") not in (None, "no_observations_yet"):
            graph_id = mlh["graph_id"]
            if graph_id in lm.graph_memory.get_memory_ids():
                channel = self._mlh_channel(graph_id)
                if channel is not None:
                    graph = lm.graph_memory.get_graph(graph_id, channel)

        if graph is None or getattr(graph, "pos", None) is None:
            self._draw_monty_placeholder("No MLH")
            return
        pos = np.asarray(graph.pos)
        if len(pos) == 0:
            self._draw_monty_placeholder("No MLH")
            return

        color = self._mlh_marker_color(mlh, lm.object_evidence_threshold)
        if is_3d(pos):
            ax = self._ensure_monty_projection("3d")
            self._show_mlh_3d(ax, mlh, pos, color)
        else:
            ax = self._ensure_monty_projection(None)
            self._show_mlh_2d(ax, mlh, graph, pos, color)

    def _mlh_channel(self, graph_id: str) -> str | None:
        """Choose which channel's stored graph to render for the MLH.

        Args:
            graph_id: The MLH graph id.

        Returns:
            The selected channel when it is part of the graph, else the graph's first
            sensor-module channel, else `None` when the graph has no sensor channel.
        """
        lm = self._channel_view.lm
        channels = lm.get_input_channels_in_graph(graph_id)
        if self._channel_view.channel in channels:
            return self._channel_view.channel
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

    def _draw_evidence_plots(self) -> None:
        """Redraw both inference line plots from the accumulated history."""
        self._ensure_details_lines()
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
