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
from typing import TYPE_CHECKING, Callable, ClassVar

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from tbp.monty.frameworks.actions.actions import SetAgentPose
from tbp.monty.frameworks.plotters.policies import (
    HEADINGS,
    STEP_SCALE_INIT,
    STEP_SCALE_MAX,
    STEP_SCALE_MIN,
    interactive_policy_for,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import KeyEvent
    from matplotlib.figure import Figure

    from tbp.monty.context import RuntimeContext
    from tbp.monty.frameworks.actions.actions import Action
    from tbp.monty.frameworks.models.abstract_monty_classes import Monty
    from tbp.monty.frameworks.plotters.helpers import ChannelView
    from tbp.monty.frameworks.plotters.policies import InteractivePolicy


END_EPISODE = "End episode"
JUMP = "jump"


class ActionButtons:
    """The interactive stepping control: a heading D-pad plus jump / end episode.

    The buttons are rebuilt each time the user is asked to choose, so only the choices
    that can act this step are shown rather than greyed out. The four exploration
    headings (up / down / left / right) wrap around the RGB patch as a D-pad, each
    pointing the way it moves the sensor. A centered "jump" button appears whenever the
    model proposes a hypothesis-testing jump, and a centered "End episode" button is
    always present.

    `override_action` blocks on the figure's event loop until a button is clicked or its
    keyboard shortcut is pressed (WASD or arrow keys for the headings, space for jump,
    delete for end episode). The matplotlib quit keys (q) still close the figure window.
    The interactive policy is built up front, so an unsupported policy fails at
    construction rather than mid-episode.

    A persistent `Step Multiplier` slider centered beneath the D-pad scales the
    exploration headings' step size: the executed action is the policy's default step
    times the slider multiplier. The slider is built once and its value carries across
    steps; jumps are unaffected.
    """

    _KEYMAP: ClassVar[dict[str, str]] = {
        "w": "up",
        "up": "up",
        "s": "down",
        "down": "down",
        "a": "left",
        "left": "left",
        "d": "right",
        "right": "right",
        " ": JUMP,
        "delete": END_EPISODE,
    }

    def __init__(self, model: Monty) -> None:
        """Build the interactive policy adapter from the model's motor policy.

        Args:
            model: The Monty model whose motor system exposes the policy.
        """
        self.model = model
        self.fig: Figure | None = None
        self._patch_ax: Axes | None = None
        self._selected: str | None = None
        self._buttons: dict[str, Button] = {}
        self._step_slider: Slider | None = None
        self._policy: InteractivePolicy = interactive_policy_for(model)

    def build(self, fig: Figure, patch_ax: Axes) -> None:
        """Bind the figure and the RGB-patch axis the heading D-pad wraps.

        Buttons are created on demand in `override_action`, so nothing is drawn yet.

        Args:
            fig: The figure to place the buttons on.
            patch_ax: The RGB-patch axis the up/down/left/right buttons are arranged
                around.
        """
        self.fig = fig
        self._patch_ax = patch_ax

        # The step-size multiplier slider is persistent (built once here, not in
        # `_rebuild`), so its value carries across choice points. It is centered beneath
        # the D-pad; `_rebuild` re-centers it once the patch's real extent is known. The
        # caption sits below the track (centered) so it does not collide with the D-pad.
        ax_step = fig.add_axes(self._step_slider_rect())
        self._step_slider = Slider(
            ax_step,
            "Step Multiplier",
            STEP_SCALE_MIN,
            STEP_SCALE_MAX,
            valinit=STEP_SCALE_INIT,
        )
        self._step_slider.label.set_position((0.5, -0.8))
        self._step_slider.label.set_horizontalalignment("center")
        self._step_slider.label.set_verticalalignment("top")

        # Drop matplotlib's default key bindings.
        manager = fig.canvas.manager
        if getattr(manager, "key_press_handler_id", None) is not None:
            fig.canvas.mpl_disconnect(manager.key_press_handler_id)
        fig.canvas.mpl_connect("key_press_event", self._on_key)

    @staticmethod
    def _is_jump(proposed: list[Action]) -> bool:
        """Whether the model proposes a hypothesis-testing jump this step.

        Args:
            proposed: The actions the model computed for this step.

        Returns:
            True when the proposed actions begin with a `SetAgentPose` teleport.
        """
        return bool(proposed) and isinstance(proposed[0], SetAgentPose)

    def awaits_choice(self, proposed: list[Action]) -> bool:
        """Whether the user should choose this step's action.

        A jump is always a choice point so the user can accept it or move instead;
        otherwise the wrapped policy decides (e.g. the surface policy's tangential
        step).

        Args:
            proposed: The actions the model computed for this step.

        Returns:
            True when this step is a user choice point.
        """
        return self._is_jump(proposed) or self._policy.awaits_choice(proposed)

    def _rebuild(self, headings: list[str], specials: list[str]) -> None:
        """Replace the choice buttons: a D-pad around the patch, specials centered.

        Args:
            headings: The exploration headings (up / down / left / right) to wrap
                around the RGB patch as a D-pad.
            specials: The centered buttons below the figure ("jump" when offered, then
                "End episode"), in left-to-right order.
        """
        for button in self._buttons.values():
            button.ax.remove()
        self._buttons = {}
        dpad = self._dpad_rects()
        for heading in headings:
            self._add_button(heading, dpad[heading])
        for label, rect in self._special_rects(specials).items():
            self._add_button(label, rect)

        # Re-center the slider now that the patch's real extent is known (the build-time
        # placement used the axis-cell fallback before any patch was drawn).
        if self._step_slider is not None:
            self._step_slider.ax.set_position(self._step_slider_rect())

        # Force a full redraw so the new buttons replace the old ones immediately.
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _add_button(self, label: str, rect: list[float]) -> None:
        """Add a single choice button at a figure-fraction rectangle.

        Args:
            label: The button label, also the selection it records when clicked.
            rect: The `[left, bottom, width, height]` figure-fraction placement.
        """
        ax_btn = self.fig.add_axes(rect)
        btn = Button(ax_btn, label)
        btn.on_clicked(lambda _event, lbl=label: self._on_click(lbl))
        self._buttons[label] = btn

    def _patch_rect(self):
        """The rendered RGB patch's figure-fraction box, not the axis cell's.

        `imshow` centers the square patch within the wider axis cell (equal aspect),
        so the axis position is far wider than the image; the buttons must hug the
        image. The drawn image artist's window extent reflects that centering, so it
        is transformed to figure fractions. Falls back to the axis cell when the patch
        has not been drawn (e.g. a placeholder step).

        Returns:
            The `Bbox` of the rendered patch in figure-fraction coordinates.
        """
        images = self._patch_ax.images
        if not images:
            return self._patch_ax.get_position(self.fig)
        extent = images[-1].get_window_extent()
        return extent.transformed(self.fig.transFigure.inverted())

    def _dpad_rects(self) -> dict[str, list[float]]:
        """Lay the four heading buttons around the RGB patch as a D-pad.

        Each button is a thin bar hugging one edge of the patch and spanning most of
        that edge.

        Returns:
            The `[left, bottom, width, height]` rectangle per heading.
        """
        p = self._patch_rect()
        width, height = p.x1 - p.x0, p.y1 - p.y0
        h_thick, v_thick, gap = 0.03, 0.04, 0.006
        bar_w, bar_h = width * 0.8, height * 0.8
        cx = p.x0 + (width - bar_w) / 2
        cy = p.y0 + (height - bar_h) / 2
        return {
            "up": [cx, p.y1 + gap, bar_w, h_thick],
            "down": [cx, p.y0 - gap - h_thick, bar_w, h_thick],
            "left": [max(0.0, p.x0 - gap - v_thick), cy, v_thick, bar_h],
            "right": [p.x1 + gap, cy, v_thick, bar_h],
        }

    def _step_slider_rect(self) -> list[float]:
        """Place the step-size slider centered beneath the D-pad's `down` bar.

        Matches the `down` bar's left and width so the slider aligns under the D-pad,
        sitting a small gap below it.

        Returns:
            The `[left, bottom, width, height]` figure-fraction placement.
        """
        left, down_bottom, bar_w, _ = self._dpad_rects()["down"]
        slider_h, gap = 0.03, 0.05
        return [left, down_bottom - gap - slider_h, bar_w, slider_h]

    @staticmethod
    def _special_rects(specials: list[str]) -> dict[str, list[float]]:
        """Center the jump / end-episode buttons in a row at the figure's bottom.

        Args:
            specials: The button labels, in left-to-right order.

        Returns:
            The `[left, bottom, width, height]` rectangle per label.
        """
        bw, bh, gap, bottom = 0.12, 0.05, 0.02, 0.04
        total = len(specials) * bw + (len(specials) - 1) * gap
        left = 0.5 - total / 2
        return {
            label: [left + i * (bw + gap), bottom, bw, bh]
            for i, label in enumerate(specials)
        }

    def override_action(
        self, ctx: RuntimeContext, proposed: list[Action]
    ) -> list[Action]:
        """Draw this step's buttons, block until one is clicked, return its action.

        The buttons are the policy's headings wrapped around the patch, plus a "jump"
        button when `proposed` is a hypothesis-testing jump, plus "End episode". The
        wait is guarded on `self._selected`, so a pre-set selection skips the event
        loop entirely. Selector-button clicks repaint the figure without setting
        `self._selected`, so they do not end the wait. Clicking "jump" executes the
        proposed jump unchanged; any other heading is computed by the interactive
        policy from the current motor-system state.

        Args:
            ctx: The runtime context supplying the random state.
            proposed: The actions the model computed for this step.

        Returns:
            The actions to execute next, built from the user's button choice.

        Raises:
            StopIteration: When the user clicks "End episode", after setting any LMs
                that have not reached a terminal state to time_out so the episode logs
                cleanly.
        """
        specials = [JUMP, END_EPISODE] if self._is_jump(proposed) else [END_EPISODE]
        self._rebuild(list(HEADINGS), specials)

        while self._selected is None:
            self.fig.canvas.start_event_loop(0.1)
        selected, self._selected = self._selected, None
        if selected == END_EPISODE:
            self.model.deal_with_time_out()
            raise StopIteration
        if selected == JUMP:
            return proposed

        state = self.model.motor_system.action_sequence[-1][1]
        scale = (
            self._step_slider.val
            if self._step_slider is not None
            else STEP_SCALE_INIT
        )
        return self._policy.compute(ctx, selected, state, scale)

    def close(self) -> None:
        """Drop the button and slider references."""
        self._buttons = {}
        self._step_slider = None

    def _on_click(self, label: str) -> None:
        """Record the clicked action and stop the blocking event loop.

        Args:
            label: The clicked button's label (a heading, jump, or "End episode").
        """
        self._selected = label
        with contextlib.suppress(Exception):
            self.fig.canvas.stop_event_loop()

    def _on_key(self, event: KeyEvent) -> None:
        """Select the action bound to a key, if its button is shown this step.

        Mirrors `_on_click`: keys map to the same labels as the buttons (WASD or arrow
        keys for the headings, space for jump, delete for end episode), and a key is
        ignored unless its button is currently drawn (e.g. space does nothing when no
        jump is offered). The matplotlib quit keys (q) still close the figure window.

        Args:
            event: The matplotlib key-press event.
        """
        if event.key in plt.rcParams["keymap.quit"]:
            plt.close(self.fig)
            return
        label = self._KEYMAP.get(event.key)
        if label is not None and label in self._buttons:
            self._on_click(label)


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
    don't interfere with the interactive action wait.
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
