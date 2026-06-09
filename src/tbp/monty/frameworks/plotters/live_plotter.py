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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.plotters.controls import (
    ActionButtons,
    SelectorBar,
    SpeedSlider,
)
from tbp.monty.frameworks.plotters.helpers import (
    ChannelView,
    EvidenceHistory,
    draw_section_dividers,
    is_interactive_backend,
)
from tbp.monty.frameworks.plotters.panels import (
    DetailsPanel,
    MontyPanel,
    SimulatorPanel,
)

if TYPE_CHECKING:
    from numpy.random import RandomState

    from tbp.monty.frameworks.actions.actions import Action
    from tbp.monty.frameworks.models.abstract_monty_classes import (
        LearningModule,
        Monty,
        Observations,
    )


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
        self._controls: ActionButtons | SpeedSlider | None = None

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
        self._channel_view = ChannelView(model)
        self._controls = (
            ActionButtons(model) if self.interactive else SpeedSlider(self.max_delay)
        )
        self._history = EvidenceHistory()
        self._last_observations: Observations | None = None
        self._last_step: int | None = None

        self._build_figure()

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
        self._monty = MontyPanel(self.fig, self._monty_spec, self._channel_view)
        self._details = DetailsPanel(
            self.fig,
            self._details_spec,
            self._channel_view,
            self._history,
        )
        draw_section_dividers(
            self.fig, self._sim_spec, self._monty_spec, self._details_spec
        )
        self._selector = SelectorBar(
            self.fig, self._monty_spec, self._channel_view, self._redraw
        )
        self._controls.build(self.fig)

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
            self._controls.pause()

    def _render(self, observations: Observations, step: int) -> None:
        """Draw every section for one frame.

        Args:
            observations: The observations to draw.
            step: The index of the current step within the episode.
        """
        self.fig.suptitle(f"Step {step}")
        if self._channel_view.ensure_channel():
            self._selector.refresh_labels()

        sm, sm_id = self._channel_view.simulator_sm()
        self._simulator.draw(observations, sm, sm_id)

        if self._lm_building_graph(self._channel_view.lm):
            self._draw_training()
        else:
            self._draw_inference()

        self._monty.draw_feature_inset()

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

        Delegates to the interactive `ActionButtons` control, which blocks on the
        figure's event loop, samples the chosen action, and raises `StopIteration` on
        "End episode".

        Args:
            rng: The random state used to sample the chosen action.

        Returns:
            The actions to execute next, built from the user's button choice.
        """
        return self._controls.override_action(rng)

    def close(self) -> None:
        """Close the final figure and drop widget references."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
        if self._controls is not None:
            self._controls.close()
        self._selector = None
        self._simulator = None
        self._monty = None
        self._details = None

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
            self._monty.draw_placeholder("no observations yet")
            self._details.draw_placeholder("no observations yet")
            return

        selected = self._channel_view.channel
        selected_pts = points.get(selected) if selected is not None else None
        if selected_pts is None or not selected_pts.size:
            label = selected if selected is not None else "channel"
            self._monty.draw_placeholder(f"no observations on {label}")
        else:
            self._monty.draw_selected_channel(selected, selected_pts)

        others = [c for c in channels if c != selected]
        if others:
            self._details.draw_buffer_grid(others, points)
        else:
            self._details.draw_placeholder("no other channels")

    def _draw_inference(self) -> None:
        """Draw the matching-step panels (MLH graph and line plots).

        Renders the displayed LM's MLH graph and line plots from the per-LM evidence
        history accumulated in `update`. Degrades to placeholders when the displayed LM
        lacks the evidence-LM inference API.
        """
        if not self._channel_view.supports_evidence:
            lm_name = type(self._channel_view.lm).__name__
            placeholder = f"Inference view not available for {lm_name}"
            self._monty.draw_placeholder(placeholder)
            self._details.draw_placeholder(placeholder)
            return
        self._monty.draw_mlh()
        self._details.draw_inference()
