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

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.random import RandomState

    from tbp.monty.frameworks.actions.action_samplers import ActionSampler
    from tbp.monty.frameworks.actions.actions import Action
    from tbp.monty.frameworks.models.abstract_monty_classes import Monty
    from tbp.monty.frameworks.plotters.helpers import ChannelView


class ActionButtons:
    """The interactive stepping control: one button per action sampler action.

    Renders one button per action the policy's action sampler can produce, plus an
    "End episode" button, along the bottom of the figure. `override_action` blocks on
    the figure's event loop until a button is clicked, then samples and returns the
    chosen action. The policy and its sampler are validated up front, so an unusable
    policy fails at construction rather than mid-episode.
    """

    def __init__(self, model: Monty) -> None:
        """Derive and validate the action sampler from the model's policy.

        Args:
            model: The Monty model whose motor system exposes the policy.

        Raises:
            ValueError: When the policy is missing or exposes no usable sampler.
        """
        self.model = model
        self.fig: Figure | None = None
        self._selected: str | None = None
        self._buttons: list[Button] = []
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
        self._sampler: ActionSampler = policy.action_sampler
        self._agent_id: str = policy.agent_id
        self.action_names: list[str] = list(self._sampler.action_names)
        if not self.action_names:
            raise ValueError(
                "Interactive plotter requires a policy whose action sampler can "
                "produce at least one action, but it produced none."
            )

    def build(self, fig: Figure) -> None:
        """Add one button per action name plus an "End episode" button.

        Args:
            fig: The figure to place the buttons along the bottom of.
        """
        self.fig = fig
        labels = self.action_names + ["End episode"]
        n = len(labels)
        for i, label in enumerate(labels):
            # [left, bottom, width, height]
            ax_btn = fig.add_axes([0.02 + i * (0.96 / n), 0.03, 0.96 / n - 0.01, 0.07])
            btn = Button(ax_btn, label)
            btn.on_clicked(lambda _event, lbl=label: self._on_click(lbl))
            self._buttons.append(btn)

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
        """Drop the button references."""
        self._buttons = []

    def _on_click(self, label: str) -> None:
        """Record the clicked action and stop the blocking event loop.

        Args:
            label: The clicked button's label (an action name or "End episode").
        """
        self._selected = label
        with contextlib.suppress(Exception):
            self.fig.canvas.stop_event_loop()


class SpeedSlider:
    """The non-interactive stepping control: a speed slider that paces playback.

    Renders a `Speed` slider along the bottom of the figure and, after each step,
    pauses for a duration derived from the slider value: full speed runs without delay,
    an intermediate value pauses proportionally up to `max_delay`, and zero halts until
    the slider is moved.
    """

    def __init__(self, max_delay: float) -> None:
        """Initialize the slider control.

        Args:
            max_delay: Maximum pause in seconds at the slowest non-halting speed.
        """
        self.max_delay = max_delay
        self.fig: Figure | None = None
        self._slider: Slider | None = None

    def build(self, fig: Figure) -> None:
        """Add the speed slider along the bottom of the figure.

        Args:
            fig: The figure to place the slider on.
        """
        self.fig = fig
        ax_slider = fig.add_axes([0.07, 0.05, 0.86, 0.03])
        self._slider = Slider(ax_slider, "Speed", 0.0, 1.0, valinit=1.0)

    def pause(self) -> None:
        """Pause (or halt) between steps according to the speed slider."""
        if self._slider is None:
            return
        delay = self._pause_seconds(float(self._slider.val))
        if delay is None:
            while float(self._slider.val) <= 0.0:
                self.fig.canvas.start_event_loop(0.1)
        elif delay > 0.0:
            plt.pause(delay)

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

    def close(self) -> None:
        """Drop the slider reference."""
        self._slider = None


class SelectorBar:
    """The displayed-LM and selected-channel cycling buttons above the Monty column.

    Two buttons in the figure's top margin, under the `Step N` title, that cycle the
    `ChannelView`'s displayed learning module and selected input channel. Each click
    advances the selection, updates the button captions, and triggers a repaint through
    the `on_change` callback so the new selection is visible immediately, even while a
    blocking event loop is running. The clicks never set the action selection, so they
    are inert with respect to the interactive action wait.
    """

    def __init__(
        self,
        fig: Figure,
        spec,
        channel_view: ChannelView,
        on_change: Callable[[], None],
    ) -> None:
        """Lay out the two cycling buttons over the Monty column's top margin.

        Args:
            fig: The figure to draw on.
            spec: The Monty column's subplot spec, used to position the buttons.
            channel_view: The selection the buttons cycle.
            on_change: Repaints the last frame after a selection change.
        """
        self.fig = fig
        self.channel_view = channel_view
        self._on_change = on_change
        monty = spec.get_position(fig)
        width = monty.x1 - monty.x0
        bottom, height = 0.91, 0.03
        lm_label, channel_label = channel_view.labels()

        ax_lm = fig.add_axes([monty.x0, bottom, width * 0.48, height])
        self._lm_button = Button(ax_lm, lm_label)
        self._lm_button.on_clicked(self._on_cycle_lm)

        ax_channel = fig.add_axes(
            [monty.x0 + width * 0.52, bottom, width * 0.48, height]
        )
        self._channel_button = Button(ax_channel, channel_label)
        self._channel_button.on_clicked(self._on_cycle_channel)

    def refresh_labels(self) -> None:
        """Update both button captions to the current selection."""
        lm_label, channel_label = self.channel_view.labels()
        self._lm_button.label.set_text(lm_label)
        self._channel_button.label.set_text(channel_label)

    def _on_cycle_lm(self, _event: object) -> None:
        """Advance to the next learning module and repaint.

        Args:
            _event: The matplotlib button event (unused).
        """
        self.channel_view.cycle_lm()
        self.refresh_labels()
        self._on_change()

    def _on_cycle_channel(self, _event: object) -> None:
        """Advance to the next input channel of the displayed LM and repaint.

        Args:
            _event: The matplotlib button event (unused).
        """
        if not self.channel_view.cycle_channel():
            return
        self.refresh_labels()
        self._on_change()
