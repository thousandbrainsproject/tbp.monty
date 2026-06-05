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
import math
from typing import TYPE_CHECKING, Callable

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tbp.monty.frameworks.actions.action_samplers import ActionSampler
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
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.frameworks.utils.plot_utils import add_patch_outline_to_view_finder

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.random import RandomState

    from tbp.monty.frameworks.actions.actions import Action
    from tbp.monty.frameworks.agents import AgentID
    from tbp.monty.frameworks.models.abstract_monty_classes import (
        LearningModule,
        SensorModule,
    )
    from tbp.monty.frameworks.models.object_model import GraphObjectModel

# turn interactive plotting off. Call plt.show() to open all figures
plt.ioff()

_END_EPISODE = "__end_episode__"
"""Internal sentinel for the "End episode" button (distinct from action names)."""

_BASE_FRAME_SIZE = 0.1
"""Side length in meters of the smallest square/cube buffer-view frame."""

_FRAME_STEP = 0.05
"""Increment in meters by which the buffer-view frame grows when points exceed it."""

_PROJECTIONS = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
"""The three head-on 2D projections drawn below a 3D buffer view, as (x_dim, y_dim,
label) over the location columns."""


def _is_interactive_backend() -> bool:
    """Whether the active matplotlib backend can run a blocking event loop.

    Returns:
        True if the current backend is an interactive (GUI) backend.
    """
    return mpl.get_backend() in mpl.rcsetup.interactive_bk


class LivePlotter:
    """Live plotter implementing the `Plotter` Protocol for training and inference.

    Renders a 3-section view of one sensor module and one learning module:

    - Simulator: view finder, RGB patch, and the feature patch (2D edge / 3D normal),
      always drawn for any learning module.
    - Monty and Details: phase-dependent on `model.step_type`. During an exploratory
      step they show the graph being learned from the LM buffer (channels overlaid plus
      per-channel subplots); during a matching step they show inference (the MLH graph
      plus evidence and number-of-hypotheses line plots).

    A non-interactive plotter exposes a `Speed` slider; an interactive plotter exposes
    one button per action the policy's sampler can produce (plus "End episode") and
    overrides the executed action via `override_action`.

    When the target learning module lacks the evidence-LM inference API (or a buffer),
    the corresponding Monty/Details panels degrade to a placeholder rather than raising.
    """

    def __init__(
        self,
        sensor_module_id: str = "patch",
        learning_module_id: str = "learning_module_0",
        interactive: bool = False,
        max_delay: float = 2.0,
        figsize: tuple[float, float] = (16, 8),
    ) -> None:
        """Initialize the plotter.

        Args:
            sensor_module_id: Id of the sensor module to plot.
            learning_module_id: Id of the learning module to plot.
            interactive: Whether to render action buttons and override the executed
                action with the user's choice (otherwise render a speed slider).
            max_delay: Maximum non-interactive pause in seconds at the slowest speed.
            figsize: Figure size in inches.
        """
        self.sensor_module_id = sensor_module_id
        self.learning_module_id = learning_module_id
        self.interactive = interactive
        self.max_delay = max_delay
        self.figsize = figsize

        self.fig = None

    @staticmethod
    def _resolve_targets(
        model: Monty, sensor_module_id: str, learning_module_id: str
    ) -> tuple[SensorModule, LearningModule]:
        """Resolve the target sensor and learning modules by string id.

        Args:
            model: The Monty model to search.
            sensor_module_id: Id of the sensor module to plot.
            learning_module_id: Id of the learning module to plot.

        Returns:
            The matching `(sensor_module, learning_module)` pair.

        Raises:
            ValueError: If either id is not found, listing the available ids.
        """
        sm = next(
            (s for s in model.sensor_modules if s.sensor_module_id == sensor_module_id),
            None,
        )
        if sm is None:
            available = [s.sensor_module_id for s in model.sensor_modules]
            raise ValueError(
                f"Sensor module '{sensor_module_id}' not found. Available: {available}"
            )
        lm = next(
            (
                m
                for m in model.learning_modules
                if m.learning_module_id == learning_module_id
            ),
            None,
        )
        if lm is None:
            available = [m.learning_module_id for m in model.learning_modules]
            raise ValueError(
                f"Learning module '{learning_module_id}' not found. "
                f"Available: {available}"
            )
        return sm, lm

    @staticmethod
    def _is_building_graph(model: Monty) -> bool:
        """Whether the learning module is currently building a graph.

        Args:
            model: The Monty model whose current step type decides the panel.

        Returns:
            True for an exploratory step, False for any other step (e.g. matching).
        """
        return model.step_type == "exploratory_step"

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

    @staticmethod
    def _channel_points(locations: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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

    def _aligned_feature(
        self, channel: str, attr: str
    ) -> npt.NDArray[np.float64] | None:
        """Return one buffer feature aligned row-for-row with a channel's valid points.

        The buffer pads every per-channel feature to the location length, so the rows
        kept by `_channel_points` (non-NaN location) index the feature identically.

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

    def _details_groups(
        self, channel: str, pts: npt.NDArray[np.float64]
    ) -> list[tuple[npt.NDArray[np.float64], object, str | None]]:
        """Resolve the per-channel coloring for a Details cell.

        Patch channels carry an `hsv` feature, so their points are colored by hue with
        no legend. Learning-module channels instead carry an `object_id` feature, so
        their points are grouped and colored by object id with a legend. When both are
        present `hsv` wins; when neither is, the points fall back to the color cycle.

        Args:
            channel: The buffer input channel being drawn.
            pts: The channel's `(M, 3)` valid points (as returned by `_channel_points`).

        Returns:
            The `(points, color, label)` groups to draw for this channel.
        """
        hsv = self._aligned_feature(channel, "hsv")
        if hsv is not None and hsv.shape[0] == pts.shape[0] and hsv.shape[1] >= 3:
            colors = mcolors.hsv_to_rgb(np.clip(hsv[:, :3], 0.0, 1.0))
            return [(pts, colors, None)]

        object_id = self._aligned_feature(channel, "object_id")
        if object_id is not None and object_id.shape[0] == pts.shape[0]:
            ids = object_id[:, 0]
            unique_ids = np.unique(ids)
            cmap = plt.get_cmap("tab10" if len(unique_ids) <= 10 else "tab20")
            groups = []
            for i, uid in enumerate(unique_ids):
                selected = pts[ids == uid]
                groups.append((selected, cmap(i % cmap.N), f"object {int(uid)}"))
            return groups

        return [(pts, None, None)]

    @staticmethod
    def _is_3d(pts: npt.NDArray[np.float64]) -> bool:
        """Whether a point cloud has a real third dimension.

        A 2D sensor module pins every location to the `z = 0` plane, so a non-zero
        spread in z marks a genuine 3D graph. Only meaningful once all points are known
        (inference); a partially built graph may look planar by chance.

        Args:
            pts: The `(M, 3)` locations to inspect.

        Returns:
            True when the points vary in z, False otherwise.
        """
        return pts.shape[1] >= 3 and not np.allclose(pts[:, 2], 0.0)

    def _frame_center_half(
        self, pts: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Center and half-side of the cube enclosing `pts`.

        The frame starts at `_BASE_FRAME_SIZE` and grows in `_FRAME_STEP` increments,
        so it only ever changes size when points cross a step boundary rather than
        rescaling continuously with every new observation.

        Args:
            pts: The `(M, 3)` points the frame must enclose.

        Returns:
            The `(3,)` cube center and the cube's half side length.
        """
        low = pts[:, :3].min(axis=0)
        high = pts[:, :3].max(axis=0)
        center = (low + high) / 2

        span = float((high - low).max())
        size = _BASE_FRAME_SIZE
        if span > size:
            size += _FRAME_STEP * math.ceil((span - _BASE_FRAME_SIZE) / _FRAME_STEP)

        return center, size / 2

    def _draw_buffer_series(
        self,
        main_ax: Axes,
        proj_axes: list[Axes],
        groups: list[tuple[npt.NDArray[np.float64], object, str | None]],
        title: str,
        title_fontsize: int | None,
    ) -> None:
        """Draw point-cloud groups in a 3D axis plus its three 2D projections.

        The 3D cube and all three projections share one stepped frame, so each head-on
        view uses the same width and height as the corresponding cube faces. A legend is
        drawn only when at least one group carries a label.

        Args:
            main_ax: The 3D axis for the point cloud.
            proj_axes: The three 2D axes for the XY/XZ/YZ projections.
            groups: The `(points, color, label)` groups to overlay, where `color` is
                `None` (cycle), a single color, or a per-point `(M, 3)` RGB array, and
                `label` is the legend entry or `None` to omit it.
            title: The title for the 3D axis.
            title_fontsize: Font size for the 3D title, or `None` for the default.
        """
        main_ax.cla()
        main_ax.set_title(title, fontsize=title_fontsize)
        all_points = [pts for pts, _, _ in groups]
        stacked = np.concatenate(all_points)
        center, half = self._frame_center_half(stacked)
        for pts, color, label in groups:
            main_ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                s=6,
                label=label,
                **self._color_kwarg(color),
            )
        main_ax.set_xlim(center[0] - half, center[0] + half)
        main_ax.set_ylim(center[1] - half, center[1] + half)
        main_ax.set_zlim(center[2] - half, center[2] + half)
        main_ax.set_box_aspect((1, 1, 1))
        if any(label is not None for _, _, label in groups):
            main_ax.legend(fontsize=8, loc="best")
        for ax, (a, b, name) in zip(proj_axes, _PROJECTIONS):
            ax.cla()
            for pts, color, _ in groups:
                ax.scatter(pts[:, a], pts[:, b], s=4, **self._color_kwarg(color))
            ax.set_xlim(center[a] - half, center[a] + half)
            ax.set_ylim(center[b] - half, center[b] + half)
            ax.set_aspect("equal")
            ax.set_title(name, fontsize=7)
            ax.tick_params(labelsize=6)

    @staticmethod
    def _color_kwarg(color: object) -> dict:
        """Map a group color to the right scatter color keyword.

        Args:
            color: `None` to use the axis color cycle, a per-point `(M, 3)` RGB array,
                or a single matplotlib color.

        Returns:
            `{}` for the cycle, `{"c": color}` for a per-point array, else
            `{"color": color}` for a single color.
        """
        if color is None:
            return {}
        if isinstance(color, np.ndarray) and color.ndim == 2:
            return {"c": color}
        return {"color": color}

    def _update_evidence_history(self, step: int) -> None:
        """Append this step's per-object evidence and hypothesis counts to the history.

        One series per object id with a non-empty hypothesis space; objects appearing
        late are NaN-backfilled so every series aligns to `self._evidence_steps`. Steps
        with a sampling burst are recorded for vertical markers.

        Args:
            step: The index of the current step within the episode.
        """
        mlh = self.lm.get_current_mlh()
        if not mlh or mlh.get("graph_id") == "no_observations_yet":
            return

        graph_ids, evidences = self.lm.evidence_for_each_graph()
        _, num_hyps = self.lm.num_hypotheses_for_each_graph()
        evidence_by_id = dict(zip(graph_ids, evidences))
        num_hyp_by_id = dict(zip(graph_ids, num_hyps))

        self._evidence_steps.append(step)
        n = len(self._evidence_steps)
        for graph_id in graph_ids:
            if graph_id not in self._evidence_history:
                self._evidence_history[graph_id] = [np.nan] * (n - 1)
                self._num_hyp_history[graph_id] = [np.nan] * (n - 1)
        for graph_id in self._evidence_history:
            self._evidence_history[graph_id].append(
                evidence_by_id.get(graph_id, np.nan)
            )
            self._num_hyp_history[graph_id].append(num_hyp_by_id.get(graph_id, np.nan))
        if self._in_burst(self.lm):
            self._burst_steps.append(step)

    def initialize(self, model: Monty) -> None:
        """Resolve the target SM/LM and build the figure, axes, and widgets.

        Detects whether the sensor module is 2D, whether the learning module supports
        the inference and buffer panels, and resets the per-episode history. When
        interactive, derives the action buttons from the policy's action sampler. Then
        builds the matplotlib figure for this episode, closing the previous episode's
        figure first so figures don't accumulate across a multi-object / multi-rotation
        run. Must be called once per episode.

        Args:
            model: The Monty model whose sensor and learning modules are plotted.
        """
        self.model = model
        self._resolve_episode_targets(model)
        self._setup_interactive_sampler(model)
        self._reset_episode_state()
        self._build_figure()

    def _resolve_episode_targets(self, model: Monty) -> None:
        """Resolve the target SM/LM and detect which panels they support.

        Args:
            model: The Monty model whose sensor and learning modules are plotted.
        """
        self.sm, self.lm = self._resolve_targets(
            model, self.sensor_module_id, self.learning_module_id
        )
        self.is_2d = isinstance(self.sm, TwoDSensorModule)
        self._supports_evidence = isinstance(self.lm, EvidenceGraphLM)

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
        """Reset the per-episode history and axis-layout bookkeeping."""
        self._evidence_steps: list[int] = []
        self._evidence_history: dict[str, list[float]] = {}
        self._num_hyp_history: dict[str, list[float]] = {}
        self._burst_steps: list[int] = []
        self._selected: str | None = None
        self._buttons: list[Button] = []
        self._slider: Slider | None = None
        self._monty_mode: str | None = "single"
        self._monty_projection: str | None = None
        self._monty_proj_axes: list[Axes] = []
        self._details_axes: list[Axes] = []
        self._details_proj_axes: list[list[Axes]] = []
        self._details_mode: str | None = None
        self._details_channel_count: int | None = None
        self._inference_legend = None

    def _build_figure(self) -> None:
        """Build this episode's figure, axes, and widgets.

        Closes the previous episode's figure first so figures don't accumulate across a
        multi-object / multi-rotation run.

        Raises:
            RuntimeError: When interactive but the active backend cannot run the
                blocking event loop (e.g. the headless `Agg` backend).
        """
        if self.interactive and not _is_interactive_backend():
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

        self._build_simulator_axes()
        self._ax_monty = self.fig.add_subplot(self._monty_spec)
        self._draw_section_dividers()

        if self.interactive:
            self._build_action_buttons()
        else:
            self._build_speed_slider()

        if _is_interactive_backend():
            self.fig.show()

    def update(self, observations: Observations, step: int) -> None:
        """Draw the current state, never blocking for an action.

        A non-interactive plotter applies its speed-slider pause/halt here; interactive
        blocking lives only in `override_action`.

        Args:
            observations: The observations from the most recent step.
            step: The index of the current step within the episode.
        """
        if self.fig is None:
            return
        self.fig.suptitle(f"Step {step}")

        # Simulator
        self._draw_simulator(observations)

        # Monty and Detail
        if self._is_building_graph(self.model):
            self._draw_training()
        else:
            self._draw_inference(step)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        if not self.interactive:
            self._apply_speed_pause()

    def override_action(self, rng: RandomState) -> list[Action]:
        """Block until a button is clicked, then return the user's chosen action.

        The wait is guarded on `self._selected`, so a pre-set selection skips the event
        loop entirely.

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
        if selected == _END_EPISODE:
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
        self._selected = _END_EPISODE if label == "End episode" else label
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

    def _build_simulator_axes(self) -> None:
        """Lay out the view-finder, RGB-patch, and feature-patch axes."""
        sim_grid = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=self._sim_spec, height_ratios=[2, 1], hspace=0.3
        )
        self._ax_view = self.fig.add_subplot(sim_grid[0, :])
        self._ax_rgb = self.fig.add_subplot(sim_grid[1, 0])
        feat_projection = None if self.is_2d else "3d"
        self._ax_feat = self.fig.add_subplot(sim_grid[1, 1], projection=feat_projection)

    def _draw_section_dividers(self) -> None:
        """Draw vertical separators in the gaps between the three column sections."""
        sim = self._sim_spec.get_position(self.fig)
        monty = self._monty_spec.get_position(self.fig)
        details = self._details_spec.get_position(self.fig)
        label_margin = 0.05
        gaps = [
            max((sim.x1 + monty.x0) / 2, sim.x1 + label_margin),
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

    def _resolve_obs_agent_id(self, observations: Observations) -> AgentID | None:
        """Find the agent id whose observations contain the target sensor module.

        Args:
            observations: The observations from the most recent step.

        Returns:
            The matching agent id, or `None` if no agent carries the sensor module.
        """
        for agent_id, agent_observations in observations.items():
            if self.sensor_module_id in agent_observations:
                return agent_id
        return None

    def _draw_simulator(self, observations: Observations) -> None:
        """Draw the Simulator section (view finder, RGB patch, feature patch).

        Args:
            observations: The observations from the most recent step.
        """
        agent_id: AgentID | None = self._resolve_obs_agent_id(observations)
        agent_obs: AgentObservations = (
            observations.get(agent_id, {}) if agent_id is not None else {}
        )
        self._draw_view_finder(agent_obs)
        self._draw_rgb_patch(agent_obs)
        self._draw_feature_patch()

    def _draw_view_finder(self, agent_obs: AgentObservations) -> None:
        """Draw the view finder, outlining the patch when a pixel location is known.

        Args:
            agent_obs: The observing agent's per-sensor observations this step.
        """
        ax = self._ax_view
        ax.cla()
        ax.set_title("View finder")
        ax.set_axis_off()
        if SensorID("view_finder") not in agent_obs:
            return
        rgba = np.asarray(agent_obs[SensorID("view_finder")]["rgba"])
        raw = self.sm._snapshot_telemetry.raw_observations
        patch_obs = agent_obs.get(self.sensor_module_id)
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

    def _draw_rgb_patch(self, agent_obs: AgentObservations) -> None:
        """Draw what the sensor module sees, preferring RGB over depth.

        Args:
            agent_obs: The observing agent's per-sensor observations this step.
        """
        ax = self._ax_rgb
        ax.cla()
        ax.set_axis_off()
        ax.set_title("Patch")
        patch = agent_obs.get(self.sensor_module_id)
        if patch is not None and "rgba" in patch:
            ax.imshow(np.asarray(patch["rgba"]))
            ax.set_title("Patch (RGB)")

    def _draw_feature_patch(self) -> None:
        """Draw the detected feature, colored by the sensed hsv.

        For a 3D sensor module the panel is a small 3D axis showing the local surface as
        a tilted square plane with an arrow along its outward normal. For a 2D sensor
        module it is a flat axis showing the detected edge as a line (or "No edge
        detected" when no edge defines the pose). Off-object or degraded observations
        carry no pose, so every access is guarded and the panel shows a message instead.
        """
        processed = self.sm.processed_obs
        if not processed:
            self._draw_feature_message("no pose")
            return
        obs = processed[-1]
        morph, non_morph = (
            obs["morphological_features"],
            obs["non_morphological_features"],
        )
        pose_vectors = morph.get("pose_vectors")
        if pose_vectors is None:
            self._draw_feature_message("off object")
            return
        pose_vectors = np.asarray(pose_vectors)

        hsv = non_morph.get("hsv")
        if hsv is not None:
            color = mcolors.hsv_to_rgb(np.clip(np.asarray(hsv, dtype=float), 0.0, 1.0))
        else:
            color = np.array([0.5, 0.5, 0.5])

        if self.is_2d:
            pose_fully_defined = bool(morph.get("pose_fully_defined", False))
            self._draw_feature_edge(pose_vectors, pose_fully_defined, color)
        else:
            self._draw_feature_surface(pose_vectors, color)

    def _clear_feature_ax(self) -> Axes:
        """Clear the feature panel and reset its title and axes.

        Returns:
            The cleared feature axis.
        """
        ax = self._ax_feat
        ax.cla()
        ax.set_axis_off()
        ax.set_title("Feature")
        return ax

    def _draw_feature_message(self, message: str) -> None:
        """Clear the feature panel and show a centered message.

        Args:
            message: The text to display (e.g. "off object").
        """
        ax = self._clear_feature_ax()
        text = ax.text if self.is_2d else ax.text2D
        text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)

    def _draw_feature_surface(
        self, pose_vectors: npt.NDArray[np.float64], color: npt.NDArray[np.float64]
    ) -> None:
        """Draw the local 3D surface as a tilted square with its outward normal.

        The square lies in the tangent plane spanned by the two principal-curvature
        directions and is colored by the sensed hsv; the arrow points along the surface
        normal. The camera looks straight down `+z` (the XY-plane view), so that it
        mirrors the agent's pose in the viewfinder.

        Args:
            pose_vectors: The `(3, 3)` pose vectors (normal, then two tangents).
            color: The face color (the sensed hsv as RGB, or gray when absent).
        """
        ax = self._clear_feature_ax()
        normal = self._unit(pose_vectors[0])
        tangent_u = self._unit(pose_vectors[1])
        tangent_v = self._unit(pose_vectors[2])
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
        ax = self._clear_feature_ax()
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

    @staticmethod
    def _unit(vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return `vec` normalized to unit length, or unchanged if near zero.

        Args:
            vec: The vector to normalize.

        Returns:
            The unit vector, or the original vector when its norm is negligible.
        """
        vec = np.asarray(vec, dtype=float)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-9 else vec

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
            n, 1, subplot_spec=self._details_spec, hspace=0.6
        )
        for i in range(n):
            inner = GridSpecFromSubplotSpec(
                2,
                3,
                subplot_spec=outer[i, 0],
                height_ratios=[3, 1],
                hspace=0.35,
                wspace=0.4,
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

        The Monty panel overlays all channels; the Details panel shows one plot group
        per channel. While the graph is still being built we cannot tell whether the
        points are planar, so every buffer view is drawn in 3D with three head-on 2D
        projections beneath it. Degrades to a placeholder when there are no observations
        yet.
        """
        locations = self.lm.buffer.locations
        points = {
            c: self._channel_points(np.asarray(locs)) for c, locs in locations.items()
        }
        channels = [c for c in points if points[c].size]
        if not channels:
            self._draw_monty_placeholder("no observations yet")
            self._draw_details_placeholder("no observations yet")
            return
        self._draw_buffer_overlay(channels, points)
        self._draw_buffer_per_channel(channels, points)

    def _draw_buffer_overlay(
        self, channels: list[str], points: dict[str, npt.NDArray[np.float64]]
    ) -> None:
        """Overlay every channel's buffered locations in the Monty panel.

        Args:
            channels: The channel ids with at least one buffered location.
            points: The per-channel non-padded location points.
        """
        main_ax, proj_axes = self._ensure_monty_projected()
        groups = [(points[c], None, str(c)) for c in channels]
        self._draw_buffer_series(
            main_ax,
            proj_axes,
            groups,
            title="Graph being learned",
            title_fontsize=None,
        )

    def _draw_buffer_per_channel(
        self, channels: list[str], points: dict[str, npt.NDArray[np.float64]]
    ) -> None:
        """Draw one Details plot group per channel's buffer.

        Args:
            channels: The channel ids with at least one location.
            points: The per-channel non-padded location points.
        """
        self._ensure_details_grid(len(channels))
        for main_ax, proj_axes, channel in zip(
            self._details_axes, self._details_proj_axes, channels
        ):
            groups = self._details_groups(channel, points[channel])
            self._draw_buffer_series(
                main_ax,
                proj_axes,
                groups,
                title=str(channel),
                title_fontsize=8,
            )

    def _draw_inference(self, step: int) -> None:
        """Draw the matching-step panels (MLH graph and line plots).

        Degrades to placeholders when the LM lacks the evidence-LM inference API.

        Args:
            step: The index of the current step within the episode.
        """
        if not self._supports_evidence:
            placeholder = f"Inference view not available for {type(self.lm).__name__}"
            self._draw_monty_placeholder(placeholder)
            self._draw_details_placeholder(placeholder)
            return
        self._draw_mlh()
        self._update_evidence_history(step)
        self._draw_evidence_plots()
        self._draw_inference_legend()

    def _draw_mlh(self) -> None:
        """Render the most likely hypothesis graph and its location marker.

        Falls back to a "No MLH" placeholder when there is no current hypothesis or the
        graph cannot be retrieved. Planar graphs are drawn as edge-oriented segments;
        all other graphs as a 3D point cloud.
        """
        graph = None
        mlh = self.lm.get_current_mlh()
        if mlh and mlh.get("graph_id") not in (None, "no_observations_yet"):
            graph_id = mlh["graph_id"]
            if graph_id in self.lm.graph_memory.get_memory_ids():
                channels = self.lm.get_input_channels_in_graph(graph_id)
                sender_types = self.lm.buffer.channel_sender_types
                sm_channels = [c for c in channels if sender_types.get(c) == "SM"]
                if sm_channels:
                    graph = self.lm.graph_memory.get_graph(graph_id, sm_channels[0])

        if graph is None or getattr(graph, "pos", None) is None:
            self._draw_monty_placeholder("No MLH")
            return
        pos = np.asarray(graph.pos)
        if len(pos) == 0:
            self._draw_monty_placeholder("No MLH")
            return

        color = self._mlh_marker_color(mlh, self.lm.object_evidence_threshold)
        if self._is_3d(pos):
            ax = self._ensure_monty_projection("3d")
            self._show_mlh_3d(ax, mlh, pos, color)
        else:
            ax = self._ensure_monty_projection(None)
            self._show_mlh_2d(ax, mlh, graph, pos, color)

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

        Nodes where an edge defines the pose are drawn as short dashed segments along
        the edge tangent; the rest are drawn as dots. Both are colored by the node's
        stored `hsv` feature.

        Args:
            ax: The 2D Monty axis.
            mlh: The current most likely hypothesis.
            graph: The MLH graph object model.
            pos: The graph node positions, shape `(N, >=2)`.
            mlh_color: The MLH location marker color.
        """
        ax.cla()
        x, y = pos[:, 0], pos[:, 1]
        feature_mapping = graph.feature_mapping

        if "hsv" in feature_mapping:
            hsv = np.asarray(graph.get_values_for_feature("hsv"))
            colors = mcolors.hsv_to_rgb(np.clip(hsv, 0.0, 1.0))
        else:
            colors = np.full((len(x), 3), 0.2)

        if "pose_fully_defined" in feature_mapping:
            flags = np.asarray(graph.get_values_for_feature("pose_fully_defined"))
            edge_mask = flags[:, 0].astype(bool)
        else:
            edge_mask = np.zeros(len(x), dtype=bool)

        ax.scatter(
            x[~edge_mask], y[~edge_mask], color=colors[~edge_mask], s=6, zorder=1
        )

        if edge_mask.any() and "pose_vectors" in feature_mapping:
            pose = np.asarray(graph.get_values_for_feature("pose_vectors"))
            tangents = pose[edge_mask, 3:5]
            tangents = tangents / np.clip(
                np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9, None
            )
            seg_half = 0.02 * max(float(np.ptp(x)), float(np.ptp(y)))
            centers = pos[edge_mask, :2]
            segments = np.stack(
                [centers - seg_half * tangents, centers + seg_half * tangents], axis=1
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

        ax.plot(
            mlh["location"][0],
            mlh["location"][1],
            "x",
            color=mlh_color,
            markersize=8,
            markeredgewidth=2,
            zorder=3,
        )

        margin = 0.05 * max(float(np.ptp(x)), float(np.ptp(y)), 1e-6)
        ax.set_xlim(x.min() - margin, x.max() + margin)
        ax.set_ylim(y.min() - margin, y.max() + margin)
        ax.set_title(f"MLH ({mlh['graph_id']})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")

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
