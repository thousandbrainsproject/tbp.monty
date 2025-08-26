# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame, Series
from pubsub.core import Publisher
from vedo import Button, Circle, Image, Mesh, Plotter, Slider2D, Text2D

from tools.plot.interactive.data import (
    DataLocator,
    DataLocatorStep,
    DataParser,
    YCBMeshLoader,
)
from tools.plot.interactive.topics import TopicMessage, TopicSpec
from tools.plot.interactive.utils import (
    Bounds,
    CoordinateMapper,
    Location2D,
    Location3D,
)
from tools.plot.interactive.widget_updaters import WidgetUpdater
from tools.plot.interactive.widgets import (
    VtkDebounceScheduler,
    Widget,
    extract_slider_state,
    set_slider_state,
)

if TYPE_CHECKING:
    import argparse


logger = logging.getLogger(__name__)


HUE_PALETTE = {
    "Added": "#66c2a5",
    "Removed": "#fc8d62",
    "Maintained": "#8da0cb",
}


class EpisodeSliderWidgetOps:
    """WidgetOps implementation for an Episode slider.

    This class sets the slider's range based on the number of
    available episodes and publishes changes as `TopicMessage` items
    under the "episode_number" topic.

    Attributes:
        plotter: A `vedo.Plotter` object to add or remove the slider and render.
        data_parser: A parser that extracts or queries information from the
            json log file.
        _add_kwargs: Default keyword arguments passed to `plotter.add_slider`.
        _locators: Data accessors keyed by name. These instruct the DataParser
            on how to retrieve the required information.
    """

    def __init__(self, plotter: Plotter, data_parser: DataParser) -> None:
        self.plotter = plotter
        self.data_parser = data_parser

        self._add_kwargs = dict(
            xmin=0, xmax=10, value=0, pos=[(0.1, 0.2), (0.7, 0.2)], title="Episode"
        )

        self._locators = self.create_locators()

    def create_locators(self) -> dict[str, DataLocator]:
        """Create and return data locators used by this widget.

        Returns:
            A dictionary containing the created locators.
        """
        locators = {}

        locators["episode"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
            ],
        )
        return locators

    def add(self, callback: Callable[[Slider2D, str], None]) -> Slider2D:
        """Create the slider widget and set its range from the data.

        The slider's `xmax` is set to the number of episodes.

        Args:
            callback: Function called with the arguments `(widget, event)` when
                the slider changes in the UI.

        Returns:
            The created widget as returned by the plotter.
        """
        kwargs = deepcopy(self._add_kwargs)
        locator = self._locators["episode"]
        kwargs.update({"xmax": len(self.data_parser.query(locator)) - 1})
        widget = self.plotter.add_slider(callback, **kwargs)
        self.plotter.render()
        return widget

    def remove(self, widget: Slider2D) -> None:
        """Remove the slider widget and re-render.

        Args:
            widget: The widget object.
        """
        self.plotter.remove(widget)
        self.plotter.render()

    def extract_state(self, widget: Slider2D) -> int:
        """Read the current slider value from its VTK representation.

        Args:
            widget: The widget object.

        Returns:
            The current slider value rounded to the nearest integer.
        """
        return extract_slider_state(widget)

    def set_state(self, widget: Slider2D, value: int) -> None:
        """Set the slider's value.

        Args:
            widget: Slider widget object.
            value: Desired episode index.
        """
        set_slider_state(widget, value)

    def state_to_messages(self, state: int) -> Iterable[TopicMessage]:
        """Convert the slider state to pubsub messages.

        Args:
            state: Selected episode index.

        Returns:
            A list with a single `TopicMessage` named `"episode_number"`.
        """
        messages = [TopicMessage(name="episode_number", value=state)]
        return messages


class StepSliderWidgetOps:
    """WidgetOps implementation for a Step slider.

    This class listens for the current episode selection and adjusts the step
    slider range to match the number of steps in that episode. It publishes
    changes as `TopicMessage` items under the "step_number" topic.

    Attributes:
        plotter: A `vedo.Plotter` object to add or remove the slider and render.
        data_parser: A parser that extracts or queries information from the
            JSON log file.
        updaters: A list with a single `WidgetUpdater` that reacts to the
            `"episode_number"` topic and calls `update_slider_range`.
        _add_kwargs: Default keyword arguments passed to `plotter.add_slider`.
        _locators: Data accessors keyed by name that instruct the `DataParser`
            how to retrieve the required information.
    """

    def __init__(self, plotter: Plotter, data_parser: DataParser) -> None:
        self.plotter = plotter
        self.data_parser = data_parser
        self.updaters = [
            WidgetUpdater(
                topics=[TopicSpec("episode_number", required=True)],
                callback=self.update_slider_range,
            )
        ]

        self._add_kwargs = dict(
            xmin=0,
            xmax=10,
            value=0,
            pos=[(0.1, 0.1), (0.7, 0.1)],
            title="Step",
        )
        self._locators = self.create_locators()

    def create_locators(self) -> dict[str, DataLocator]:
        """Create and return data locators used by this widget.

        Returns:
            A dictionary containing the created locators.
        """
        locators = {}
        locators["step"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="lm", value="LM_0"),
                DataLocatorStep.key(
                    name="telemetry", value="hypotheses_updater_telemetry"
                ),
                DataLocatorStep.index(name="step"),
            ]
        )
        return locators

    def add(self, callback: Callable) -> Slider2D:
        """Create the slider widget.

        Args:
            callback: Function called with `(widget, event)` when the UI changes.

        Returns:
            The created `Slider2D` widget.
        """
        widget = self.plotter.add_slider(callback, **self._add_kwargs)
        self.plotter.render()
        return widget

    def remove(self, widget: Slider2D) -> None:
        """Remove the slider widget and re-render.

        Args:
            widget: The slider widget object.
        """
        self.plotter.remove(widget)
        self.plotter.render()

    def extract_state(self, widget: Slider2D) -> int:
        """Read the current slider value.

        Args:
            widget: The slider widget.

        Returns:
            The current slider value rounded to the nearest integer.
        """
        return extract_slider_state(widget)

    def set_state(self, widget: Slider2D, value: int) -> None:
        """Set the slider's value.

        Args:
            widget: Slider widget object.
            value: Desired step index.
        """
        set_slider_state(widget, value)

    def state_to_messages(self, state: int) -> Iterable[TopicMessage]:
        """Convert the slider state to pubsub messages.

        Args:
            state: Selected step index.

        Returns:
            A list with a single `TopicMessage` for the topic `"step_number"`.
        """
        messages = [TopicMessage(name="step_number", value=state)]
        return messages

    def update_slider_range(
        self, widget: Any, msgs: list[TopicMessage]
    ) -> tuple[Any, bool]:
        """Adjust slider range based on the selected episode and reset to 0.

        Looks up the `"episode_number"` message, queries the number of steps for
        that episode, sets the slider range to `[0, num_steps - 1]`, resets the
        value to 0, and re-renders.

        Args:
            widget: The slider widget to update.
            msgs: Messages from the `WidgetUpdater`.

        Returns:
            A tuple `(widget, True)` indicating the updated widget and whether
            a publish should occur.
        """
        msgs_dict = {msg.name: msg.value for msg in msgs}

        # set widget range to the correct step number
        widget.range = [
            0,
            len(
                self.data_parser.query(
                    self._locators["step"], episode=str(msgs_dict["episode_number"])
                )
            )
            - 1,
        ]

        # set slider value back to zero
        self.set_state(widget, 0)
        self.plotter.render()

        return widget, True


class GtMeshWidgetOps:
    """WidgetOps implementation for rendering the ground-truth target mesh.

    This widget is display-only. It listens for `"episode_number"` updates,
    loads the target object's YCB mesh, applies the episode-specific rotations,
    scales and positions it, and adds it to the plotter. It does not publish
    any messages.

    Attributes:
        plotter: A `vedo.Plotter` used to add and remove actors.
        data_parser: A parser that extracts entries from the JSON log.
        ycb_loader: Loader that returns a textured `vedo.Mesh` for a YCB object.
        updaters: A single `WidgetUpdater` that reacts to `"episode_number"`.
        _locators: Data accessors keyed by name for the parser.
    """

    def __init__(
        self, plotter: Plotter, data_parser: DataParser, ycb_loader: YCBMeshLoader
    ):
        self.plotter = plotter
        self.data_parser = data_parser
        self.ycb_loader = ycb_loader
        self.updaters = [
            WidgetUpdater(
                topics=[TopicSpec("episode_number", required=True)],
                callback=self.update_mesh,
            )
        ]
        self._locators = self.create_locators()

    def create_locators(self) -> dict[str, DataLocator]:
        """Create and return data locators used by this widget.

        Returns:
            A dictionary containing the created locators.
        """
        locators = {}
        locators["target"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="lm", value="target"),
            ]
        )
        return locators

    def remove(self, widget: Mesh) -> None:
        """Remove the mesh widget and re-render.

        Args:
            widget: The mesh widget to remove. If `None`, no action is taken.
        """
        if widget is not None:
            self.plotter.remove(widget)
            self.plotter.render()

    def update_mesh(self, widget: Mesh, msgs: list[TopicMessage]) -> tuple[Mesh, bool]:
        """Update the target mesh when the episode changes.

        Removes any existing mesh, loads the episode's primary target object,
        applies its Euler rotations, scales and positions it, then adds it to
        the plotter.

        Args:
            widget: The currently displayed mesh, if any.
            msgs: Messages received from the `WidgetUpdater`.

        Returns:
            A tuple `(mesh, False)`. The second value is `False` to indicate
            that no publish should occur.
        """
        self.remove(widget)
        msgs_dict = {msg.name: msg.value for msg in msgs}

        locator = self._locators["target"]
        target = self.data_parser.extract(
            locator, episode=str(msgs_dict["episode_number"])
        )
        target_id = target["primary_target_object"]
        target_rot = target["primary_target_rotation_euler"]
        widget = self.ycb_loader.create_mesh(target_id).clone(deep=True)
        widget.rotate_x(target_rot[0])
        widget.rotate_y(target_rot[1])
        widget.rotate_z(target_rot[2])
        widget.scale(1500)
        widget.pos(-300, 100, -500)

        self.plotter.add(widget)
        self.plotter.render()

        return widget, False


class PrimaryButtonWidgetOps:
    """WidgetOps implementation for a primary-target button.

    The button publishes a `"primary_button"` boolean message whenever it is
    pressed.

    Attributes:
        plotter: A `vedo.Plotter` object used to add/remove actors and render.
        _add_kwargs: Default keyword arguments passed to `plotter.add_button`.
    """

    def __init__(self, plotter: Plotter):
        self.plotter = plotter

        self._add_kwargs = dict(
            pos=(0.85, 0.2),
            states=["Primary Target"],
            c="w",
            bc="dg",
            size=30,
            font="Calco",
            bold=True,
        )

    def add(self, callback: Callable) -> Button:
        """Create the button widget and re-render.

        Args:
            callback: Function called with `(widget, event)` on UI interaction.

        Returns:
            The created `vedo.Button`.
        """
        widget = self.plotter.add_button(callback, **self._add_kwargs)
        self.plotter.render()
        return widget

    def state_to_messages(self, state: str) -> Iterable[TopicMessage]:
        """Convert the button state to pubsub messages.

        Args:
            state: Current button state.

        Returns:
            A list with a single `TopicMessage` with the topic "primary_button" .
        """
        messages = [
            TopicMessage(name="primary_button", value=True),
        ]
        return messages


class PrevButtonWidgetOps:
    """WidgetOps implementation for a previous object button.

    The button publishes a `"prev_button"` boolean message whenever it is
    pressed.

    Attributes:
        plotter: A `vedo.Plotter` object used to add/remove actors and render.
        _add_kwargs: Default keyword arguments passed to `plotter.add_button`.
    """

    def __init__(self, plotter: Plotter):
        self.plotter = plotter

        self._add_kwargs = dict(
            pos=(0.83, 0.13),
            states=["<"],
            c=["w"],
            bc=["dg"],
            size=30,
            font="Calco",
            bold=True,
        )

    def add(self, callback: Callable) -> Button:
        """Create the button widget and re-render.

        Args:
            callback: Function called with `(widget, event)` on UI interaction.

        Returns:
            The created `vedo.Button`.
        """
        widget = self.plotter.add_button(callback, **self._add_kwargs)
        self.plotter.render()
        return widget

    def state_to_messages(self, state: str) -> Iterable[TopicMessage]:
        """Convert the button state to pubsub messages.

        Args:
            state: Current button state.

        Returns:
            A list with a single `TopicMessage` with the topic `"prev_button"`.
        """
        messages = [
            TopicMessage(name="prev_button", value=True),
        ]
        return messages


class NextButtonWidgetOps:
    """WidgetOps implementation for a next object button.

    The button publishes a `"next_button"` boolean message whenever it is
    pressed.

    Attributes:
        plotter: A `vedo.Plotter` object used to add/remove actors and render.
        _add_kwargs: Default keyword arguments passed to `plotter.add_button`.
    """

    def __init__(self, plotter: Plotter):
        self.plotter = plotter

        self._add_kwargs = dict(
            pos=(0.88, 0.13),
            states=[">"],
            c=["w"],
            bc=["dg"],
            size=30,
            font="Calco",
            bold=True,
        )

    def add(self, callback: Callable) -> Button:
        """Create the button widget and re-render.

        Args:
            callback: Function called with `(widget, event)` on UI interaction.

        Returns:
            The created `vedo.Button`.
        """
        widget = self.plotter.add_button(callback, **self._add_kwargs)
        self.plotter.render()
        return widget

    def state_to_messages(self, state: str) -> Iterable[TopicMessage]:
        """Convert the button state to pubsub messages.

        Args:
            state: Current button state.

        Returns:
            A list with a single `TopicMessage` with the topic `"next_button"`.
        """
        return [TopicMessage(name="next_button", value=True)]


class AgeThresholdWidgetOps:
    """WidgetOps implementation for an age-threshold slider.

    Publishes `"age_threshold"` with the current integer value whenever the
    slider changes.

    Attributes:
        plotter: A `vedo.Plotter` used to add/remove the slider and render.
        _add_kwargs: Default keyword arguments passed to `plotter.add_slider`.
    """

    def __init__(self, plotter: Plotter) -> None:
        self.plotter = plotter

        self._add_kwargs = dict(
            xmin=0,
            xmax=10,
            value=0,
            pos=[(0.05, 0.01), (0.05, 0.3)],
            title="Age",
        )

    def add(self, callback: Callable) -> Slider2D:
        """Create the slider widget and re-render.

        Args:
            callback: Function called with `(widget, event)` when the UI changes.

        Returns:
            The created `Slider2D` widget.
        """
        widget = self.plotter.add_slider(callback, **self._add_kwargs)
        self.plotter.render()
        return widget

    def set_state(self, widget: Slider2D, value: int) -> None:
        """Set the slider value.

        Args:
            widget: The slider widget.
            value: Desired threshold (integer).
        """
        set_slider_state(widget, value)

    def extract_state(self, widget: Slider2D) -> int:
        """Read the current slider value.

        Args:
            widget: The slider widget.

        Returns:
            The current value as an integer.
        """
        return extract_slider_state(widget)

    def state_to_messages(self, state: int) -> Iterable[TopicMessage]:
        """Convert the slider state to pubsub messages.

        Args:
            state: Current threshold value.

        Returns:
            A list with a single `TopicMessage` named `"age_threshold"`.
        """
        return [TopicMessage(name="age_threshold", value=state)]


class CurrentObjectWidgetOps:
    """Tracks and publishes the currently selected object label.

    This class has no visual widget of its own. It listens to:
      - `"episode_number"` (and optionally `"primary_button"`) to jump selection
        to the episode's primary target object.
      - `"prev_button"` and `"next_button"` to step backward/forward within the
        episode's object list.

    It publishes the `"current_object"` topic with the selected `graph_id` label.

    Attributes:
        data_parser: Parser used to query objects and target info from logs.
        updaters: Three `WidgetUpdater`s receiving the topics described above.
        _locators: Data locators for the objects list and target info.
        objects_list: Cached list of available object labels for the episode/step.
        current_object_ix: Current index into `objects_list`, or None if unset.
    """

    def __init__(self, data_parser: DataParser) -> None:
        self.data_parser = data_parser
        self.updaters = [
            WidgetUpdater(
                topics=[
                    TopicSpec("episode_number", required=True),
                    TopicSpec("primary_button", required=False),
                ],
                callback=self.update_to_primary,
            ),
            WidgetUpdater(
                topics=[
                    TopicSpec("prev_button", required=True),
                ],
                callback=self.update_current_object,
            ),
            WidgetUpdater(
                topics=[
                    TopicSpec("next_button", required=True),
                ],
                callback=self.update_current_object,
            ),
        ]

        self._locators = self.create_locators()
        self.objects_list = self.add_object_list()
        self.current_object_ix = None

    def create_locators(self) -> dict[str, DataLocator]:
        """Create and return data locators used by this class.

        Two locators are defined:
            - "objects_list": can be used to query the list of
                objects available to the episode
            - "target": can be used to extract the MLH

        Returns:
            Dictionary of data locators
        """
        locators = {}
        locators["objects_list"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="lm", value="LM_0"),
                DataLocatorStep.key(
                    name="telemetry", value="hypotheses_updater_telemetry"
                ),
                DataLocatorStep.index(name="step"),
                DataLocatorStep.key(name="objects"),
            ]
        )
        locators["target"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="lm", value="target"),
            ]
        )

        return locators

    def add_object_list(self) -> list[str] | list[int]:
        """Initialize internal state; no visual widget is created.

        Preloads the objects list for episode 0, step 0 as a default.

        Returns:
            List of graph ids as the object list
        """
        obj_list_locator = self._locators["objects_list"]
        objects_list = self.data_parser.query(
            obj_list_locator,
            episode="0",
            step=0,
        )
        return objects_list

    def state_to_messages(self, state: str) -> Iterable[TopicMessage]:
        """Convert the current object to a pubsub message with topic `current_object`.

        Returns:
            List of topic messages to be published.

        Raises:
            RuntimeError: If there is no current selection or the objects list is empty.
        """
        if self.current_object_ix is None or not self.objects_list:
            raise RuntimeError("No current object is selected or list is empty.")

        obj = self.objects_list[self.current_object_ix]
        return [TopicMessage(name="current_object", value=obj)]

    def update_to_primary(
        self, widget: None, msgs: list[TopicMessage]
    ) -> tuple[None, bool]:
        """Jump selection to the episode's primary target object.

        Also refreshes `objects_list` for that episode at step 0.

        Args:
            widget: Unused (no visual widget).
            msgs: Inbox containing `"episode_number"` and optionally `"primary_button"`.

        Returns:
            tuple of `(widget, True)` to publish the `current_object` state.
        """
        msgs_dict = {msg.name: msg.value for msg in msgs}
        episode = msgs_dict["episode_number"]

        # Refresh objects_list for the new episode at step 0
        obj_list_locator = self._locators["objects_list"]
        self.objects_list = list(
            self.data_parser.query(obj_list_locator, episode=str(episode), step=0)
        )

        if not self.objects_list:
            self.current_object_ix = None
            return widget, False

        target_locator = self._locators["target"]
        current_object = self.data_parser.extract(
            target_locator,
            episode=str(episode),
        )["primary_target_object"]

        try:
            self.current_object_ix = self.objects_list.index(current_object)
        except ValueError:
            # If the primary target is not in the list, fall back to the first item.
            self.current_object_ix = 0

        return widget, True

    def update_current_object(
        self, widget: None, msgs: list[TopicMessage]
    ) -> tuple[None, bool]:
        """Step backward or forward through `objects_list`.

        Args:
            widget: Returned as is. Value is `None` for this class with no visual
                widget.
            msgs: Single message with the topic "prev_button" or "next_button".

        Returns:
            `(widget, True)` if the selection changed, else `(widget, False)`.
        """
        if not self.objects_list:
            return widget, False

        # If object index not initialized, set to object 0
        if self.current_object_ix is None:
            self.current_object_ix = 0

        # This callback listens to a single topic
        if len(msgs) != 1:
            return widget, False

        # check topic name
        topic_name = msgs[0].name
        if topic_name == "prev_button":
            self.current_object_ix -= 1
        elif topic_name == "next_button":
            self.current_object_ix += 1
        else:
            return widget, False

        self.current_object_ix %= len(self.objects_list)
        return widget, True


class ClickWidgetOps:
    """Captures 3D click positions and publish them on the bus.

    This class registers plotter-level mouse callbacks. A left-click picks a 3D
    point (if available) and triggers the widget callback; a right-click
    resets the camera pose. There is no visual widget created by this class.

    Attributes:
        plotter: The `vedo.Plotter` where callbacks are installed.
        cam_dict: Dictionary for camera default specs.
        click_location: Last picked 3D location, if any.
        _on_change_cb: The widget callback to invoke on left-click.
    """

    def __init__(self, plotter: Plotter, cam_dict: dict[str, Any]) -> None:
        self.plotter = plotter
        self.cam_dict = cam_dict
        self.click_location: Location3D
        self._on_change_cb: Callable

    def add(self, callback: Callable) -> None:
        """Register mouse callbacks on the plotter.

        Note that this callback makes use of the `VtkDebounceScheduler`
        to publish messages. Storing the callback and triggering it, will
        simulate a UI change on e.g., a button or a slider, which schedules
        a publish. We use this callback because this event is not triggered
        by receiving topics from a `WidgetUpdater`.


        Args:
            callback: Function invoked like `(widget, event)` when a left-click
                captures a 3D location.
        """
        self._on_change_cb = callback
        self.plotter.add_callback("LeftButtonPress", self.on_right_click)
        self.plotter.add_callback("RightButtonPress", self.on_left_click)

    def extract_state(self, widget: None) -> Location3D:
        """Return the last picked 3D location."""
        return self.click_location

    def state_to_messages(self, state: Location3D) -> Iterable[TopicMessage]:
        """Convert the current click location to pubsub messages.

        Publishes a single "click_location" message whose value is a Location3D with
        "x,y,z" attributes.

        Args:
            state: The last picked 3D point.

        Returns:
            A list containing one `TopicMessage` with name "click_location".
        """
        messages = [
            TopicMessage(name="click_location", value=state),
        ]
        return messages

    def on_right_click(self, event) -> None:
        """Handle left mouse press (picks a 3D point if available).

        Notes:
            Bound to the `LeftButtonPress` event in `self.add()`.
        """
        location = getattr(event, "picked3d", None)
        if location is None or self._on_change_cb is None:
            return

        self.click_location = Location3D(*location)
        self._on_change_cb(widget=None, _event=event)

    def on_left_click(self, event):
        """Handle right mouse press (reset camera pose and render).

        Notes:
            Bound to the "RightButtonPress" event in `self.add()`.
        """
        renderer = self.plotter.renderer
        if renderer is not None:
            cam = renderer.GetActiveCamera()
            cam.SetPosition(self.cam_dict["pos"])
            cam.SetFocalPoint(self.cam_dict["focal_point"])
            cam.SetViewUp((0, 1, 0))
            cam.SetClippingRange((0.01, 1000.01))
            self.plotter.render()


class CorrelationPlotWidgetOps:
    """WidgetOps for a correlation scatter plot with selection highlighting.

    Listens for episode, step, current object, and age threshold updates to
    rebuild a seaborn joint plot. Also listens for a 3D click location to
    select the nearest hypothesis in data space and highlight it on the plot.

    Attributes:
        plotter: The `vedo.Plotter` used to add and remove actors.
        data_parser: Parser that extracts entries from the JSON log.
        updaters: Two `WidgetUpdater`s, one for plot updates and one for selection.
        df: The current pandas DataFrame for the correlation plot.
        highlight_circle: The small circle placed over the selected point.
        selected_hypothesis: The most recently selected row as a pandas Series.
        info_widget: A `Text2D` widget with a brief summary of the hyp. space.
        _locators: Data accessors used to query channels and updater stats.
        _coordinate_mapper: Maps GUI pixel coordinates to data coordinates, and back.
    """

    def __init__(self, plotter: Plotter, data_parser: DataParser) -> None:
        self.plotter = plotter
        self.data_parser = data_parser
        self.updaters = [
            WidgetUpdater(
                topics=[
                    TopicSpec("episode_number", required=True),
                    TopicSpec("step_number", required=True),
                    TopicSpec("current_object", required=True),
                    TopicSpec("age_threshold", required=True),
                ],
                callback=self.update_plot,
            ),
            WidgetUpdater(
                topics=[
                    TopicSpec("click_location", required=True),
                ],
                callback=self.update_selection,
            ),
        ]
        self._locators = self.create_locators()
        self._coordinate_mapper = CoordinateMapper(
            gui=Bounds(74, 496, 64, 496),
            data=Bounds(-2.0, 2.0, 0.0, 3.25),
        )

        self.df: DataFrame
        self.selected_hypothesis: Series | None = None
        self.highlight_circle: Circle | None = None
        self.info_widget: Text2D | None = None

    def create_locators(self) -> dict[str, DataLocator]:
        """Create and return data locators used by this widget.

        Returns:
            A dictionary with entries for `"channel"` and `"updater"`.
        """
        locators = {}
        locators["channel"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="lm", value="LM_0"),
                DataLocatorStep.key(
                    name="telemetry", value="hypotheses_updater_telemetry"
                ),
                DataLocatorStep.index(name="step"),
                DataLocatorStep.key(name="obj"),
                DataLocatorStep.key(name="channel"),
            ],
        )
        locators["updater"] = locators["channel"].extend(
            steps=[
                DataLocatorStep.key(name="stat", value="hypotheses_updater"),
            ],
        )
        return locators

    def state_to_messages(self, state: None) -> Iterable[TopicMessage]:
        """Publish either the selected hypothesis row or a clear signal.

        Notes:
            - We do not use the widget state here, we instead store the
                state in this class in `self.selected_hypothesis` as a pandas `Series`.
                An alternative would be to define a `set_state` and `extract_state` to
                store the state and retrieve it from the `Widget` class.
            - The `Widget` state will always be `None` in this case. So make sure
                `dedupe` is set to False, otherwise, messages will be not be published
                because they would be considered duplicates.

        Returns:
            A single message:
              - "selected_hypothesis" if a selection exists.
              - "clear_selected_hypothesis" otherwise.
        """
        if self.selected_hypothesis is None:
            return [TopicMessage("clear_selected_hypothesis", value=True)]
        return [TopicMessage("selected_hypothesis", value=self.selected_hypothesis)]

    def increment_step(self, episode: int, step: int) -> tuple[int, int]:
        """Compute the next `(episode, step)` pair.

        If this is the final step of the final episode, returns the same pair.

        Args:
            episode: Current episode index.
            step: Current step index.

        Returns:
            A tuple `(next_episode, next_step)`.
        """
        last_episode = len(self.data_parser.query(self._locators["channel"])) - 1
        last_step = (
            len(self.data_parser.query(self._locators["channel"], episode=str(episode)))
            - 1
        )
        if episode == last_episode and step == last_step:
            return episode, step
        if step < last_step:
            return episode, step + 1
        return episode + 1, 0

    def generate_df(self, episode: int, step: int, graph_id: str) -> DataFrame:
        """Build a DataFrame of hypotheses and their stats.

        Rows are labeled as Added, Removed, or Maintained based on the hypotheses
        updater stats.

        Note that we retrieve the removed indices from the next step. At the current
        step, we mark the existing hypotheses that will be removed in the next step,
        as "Removed"

        Args:
            episode: Episode index.
            step: Step index.
            graph_id: Object identifier to select within the episode.

        Returns:
            A concatenated `DataFrame` (across input channels) with columns:
            `["graph_id", "Evidence", "Evidence Slope", "Rot_x", "Rot_y", "Rot_z",
              "Pose Error", "age", "kind", "input_channel"]`.
        """
        input_channels = self.data_parser.query(
            self._locators["channel"],
            episode=str(episode),
            step=step,
            obj=graph_id,
        )

        all_dfs: list[DataFrame] = []
        for input_channel in input_channels:
            channel_data = self.data_parser.extract(
                self._locators["channel"],
                episode=str(episode),
                step=step,
                obj=graph_id,
                channel=input_channel,
            )
            updater_data = self.data_parser.extract(
                self._locators["updater"],
                episode=str(episode),
                step=step,
                obj=graph_id,
                channel=input_channel,
            )
            inc_episode, inc_step = self.increment_step(episode, step)
            inc_updater_data = self.data_parser.extract(
                self._locators["updater"],
                episode=str(inc_episode),
                step=inc_step,
                obj=graph_id,
                channel=input_channel,
            )

            # Removed hypotheses
            removed_ids = inc_updater_data.get("removed_ids", [])
            if len(removed_ids) > 0:
                df_removed = DataFrame(
                    {
                        "graph_id": graph_id,
                        "Evidence": np.array(channel_data["evidence"])[removed_ids],
                        "Evidence Slope": np.array(updater_data["evidence_slopes"])[
                            removed_ids
                        ],
                        "Rot_x": np.array(channel_data["rotations"])[removed_ids][:, 0],
                        "Rot_y": np.array(channel_data["rotations"])[removed_ids][:, 1],
                        "Rot_z": np.array(channel_data["rotations"])[removed_ids][:, 2],
                        "Pose Error": np.array(channel_data["pose_errors"])[
                            removed_ids
                        ],
                        "age": np.array(updater_data["ages"])[removed_ids],
                        "kind": "Removed",
                        "input_channel": input_channel,
                    }
                )
                all_dfs.append(df_removed)

            # Added hypotheses
            added_ids = updater_data.get("added_ids", [])
            added_ids = sorted(set(added_ids) - set(removed_ids))
            if added_ids:
                df_added = DataFrame(
                    {
                        "graph_id": graph_id,
                        "Evidence": np.array(channel_data["evidence"])[added_ids],
                        "Evidence Slope": np.array(updater_data["evidence_slopes"])[
                            added_ids
                        ],
                        "Rot_x": np.array(channel_data["rotations"])[added_ids][:, 0],
                        "Rot_y": np.array(channel_data["rotations"])[added_ids][:, 1],
                        "Rot_z": np.array(channel_data["rotations"])[added_ids][:, 2],
                        "Pose Error": np.array(channel_data["pose_errors"])[added_ids],
                        "age": np.array(updater_data["ages"])[added_ids],
                        "kind": "Added",
                        "input_channel": input_channel,
                    }
                )
                all_dfs.append(df_added)

            # Maintained hypotheses
            total_ids = list(range(len(updater_data["evidence_slopes"])))
            maintained_ids = sorted(set(total_ids) - set(added_ids) - set(removed_ids))
            if maintained_ids:
                df_maintained = DataFrame(
                    {
                        "graph_id": graph_id,
                        "Evidence": np.array(channel_data["evidence"])[maintained_ids],
                        "Evidence Slope": np.array(updater_data["evidence_slopes"])[
                            maintained_ids
                        ],
                        "Rot_x": np.array(channel_data["rotations"])[maintained_ids][
                            :, 0
                        ],
                        "Rot_y": np.array(channel_data["rotations"])[maintained_ids][
                            :, 1
                        ],
                        "Rot_z": np.array(channel_data["rotations"])[maintained_ids][
                            :, 2
                        ],
                        "Pose Error": np.array(channel_data["pose_errors"])[
                            maintained_ids
                        ],
                        "age": np.array(updater_data["ages"])[maintained_ids],
                        "kind": "Maintained",
                        "input_channel": input_channel,
                    }
                )
                all_dfs.append(df_maintained)

        return pd.concat(all_dfs, ignore_index=True)

    def create_figure(
        self, df: DataFrame, x="Evidence Slope", y="Pose Error"
    ) -> plt.Figure:
        """Create a seaborn joint scatter with marginal KDEs.

        Args:
            df: Data frame to plot.
            x: X column name. Defaults to "Evidence Slope".
            y: Y column name. Defaults to "Pose Error".

        Returns:
            A Matplotlib `Figure`.
        """
        g = sns.JointGrid(data=df, x=x, y=y, height=6)

        sns.scatterplot(
            data=df,
            x=x,
            y=y,
            hue="kind",
            ax=g.ax_joint,
            s=8,
            alpha=0.8,
            palette=HUE_PALETTE,
        )

        sns.kdeplot(
            data=df,
            x=x,
            hue="kind",
            ax=g.ax_marg_x,
            fill=True,
            alpha=0.2,
            common_norm=False,
            palette=HUE_PALETTE,
            legend=False,
        )

        sns.kdeplot(
            data=df,
            y=y,
            hue="kind",
            ax=g.ax_marg_y,
            fill=True,
            alpha=0.2,
            common_norm=False,
            palette=HUE_PALETTE,
            legend=False,
        )

        legend = g.ax_joint.get_legend()
        if legend:
            legend.set_title(None)

        g.ax_joint.set_xlim(-2.0, 2.0)
        g.ax_joint.set_ylim(0, 3.25)
        g.ax_joint.set_xlabel(x, labelpad=10)
        g.ax_joint.set_ylabel(y, labelpad=10)
        g.figure.tight_layout()
        return g.figure

    def get_closest_row(self, df: DataFrame, slope: float, error: float) -> Series:
        """Return the row whose (Evidence Slope, Pose Error) is closest to a point.

        Args:
            df: Data to search.
            slope: Target x value in data space.
            error: Target y value in data space.

        Returns:
            The closest row as a pandas Series.

        Raises:
            ValueError: If `df` is empty.
        """
        if df.empty:
            raise ValueError("DataFrame is empty.")

        # Compute Euclidean distance
        distances = np.sqrt(
            (df["Evidence Slope"] - slope) ** 2 + (df["Pose Error"] - error) ** 2
        )
        return df.loc[distances.idxmin()]

    def create_info_text(self, df: DataFrame) -> str:
        """Summarize hypotheses statistics from a dataframe.

        Args:
            df: DataFrame with at least 'graph_id' and 'kind' columns.
                 'kind' should contain values 'Added', 'Removed', or 'Maintained'.

        Returns:
            str: Formatted summary string.
        """
        if df.empty:
            return "No hypotheses found."

        # Assume all rows share the same object name
        graph_id = df["graph_id"].iloc[0]

        # Count per kind
        kind_counts = df["kind"].value_counts()

        total = len(df)
        added = kind_counts.get("Added", 0)
        removed = kind_counts.get("Removed", 0)

        text = (
            f"Object: {graph_id}\n"
            f"Total Existing Hypotheses: {total}\n"
            f"Added Hypotheses: {added}\n"
            f"To be removed Hypotheses: {removed}"
        )
        return text

    def update_plot(self, widget: Image, msgs: list[TopicMessage]) -> tuple[Any, bool]:
        """Rebuild the plot for the selected episode, step, object, and age threshold.

        Removes previous plot, generates the new DataFrame, creates a
        joint scatter plot, places it in the scene, and adds an info text panel.

        Args:
            widget: The previous figure, if any.
            msgs: Messages received, containing `"episode_number"`, `"step_number"`,
                `"current_object"`, and `"age_threshold"`.

        Returns:
            `(new_widget, True)` where `new_widget` is the new image actor.
        """
        # Clear previous plot and selection
        if widget is not None:
            self.plotter.remove(widget)

        if self.highlight_circle is not None:
            self.plotter.remove(self.highlight_circle)

        if self.info_widget is not None:
            self.plotter.remove(self.info_widget)

        self.selected_hypothesis = None

        # Build DataFrame and filter by age
        msgs_dict = {msg.name: msg.value for msg in msgs}
        df = self.generate_df(
            episode=msgs_dict["episode_number"],
            step=msgs_dict["step_number"],
            graph_id=msgs_dict["current_object"],
        )
        age_threshold: int = int(msgs_dict["age_threshold"])
        mask: Series = df["age"] >= age_threshold
        self.df: DataFrame = df.loc[mask].copy()

        # Create figure and add to scene
        fig = self.create_figure(self.df)
        widget = Image(fig)
        plt.close(fig)
        self.plotter.add(widget)

        # Add info text to scene
        info_text = self.create_info_text(self.df)
        self.info_widget = Text2D(txt=info_text, pos="top-left")
        self.plotter.add(self.info_widget)

        self.plotter.render()
        return widget, True

    def update_selection(
        self, widget: Image | None, msgs: list[TopicMessage]
    ) -> tuple[Any, bool]:
        """Highlight the data point nearest to a GUI click location.

        Maps the 2D GUI click to data coordinates, finds the closest row,
        places a small circle over the corresponding location in GUI space,
        and stores the selected row in `selected_hypothesis`.

        Args:
            widget: The current image plot. If `None`, selection is ignored.
            msgs: Inbox with a single "click_location" message whose value
                is a `Location3D`.

        Returns:
            `(widget, True)` if a selection was made, otherwise `(widget, False)`.
        """
        if widget is None or self.df is None or self.df.empty:
            return widget, False

        msgs_dict = {msg.name: msg.value for msg in msgs}
        location = msgs_dict["click_location"].to_2d()

        if not self._coordinate_mapper.gui.contains(location):
            return widget, False

        # Get the location in data (slope, error) space
        data_location = self._coordinate_mapper.map_click_to_data_coords(location)

        # Find the closest data point in the data frame
        df_row = self.get_closest_row(
            self.df, slope=data_location.x, error=data_location.y
        )
        df_location = Location2D(
            float(df_row["Evidence Slope"]), float(df_row["Pose Error"])
        )
        self.selected_hypothesis = df_row

        # Map location back to a Location3D in GUI Space
        gui_location = self._coordinate_mapper.map_data_coords_to_world(
            df_location
        ).to_3d(z=0.1)

        # Add the highlight circle
        if self.highlight_circle is not None:
            self.plotter.remove(self.highlight_circle)

        self.highlight_circle = Circle(pos=gui_location.to_numpy(), r=3.0, res=16)
        self.highlight_circle.c("red")
        self.plotter.add(self.highlight_circle)

        self.plotter.render()
        return widget, True


class HypothesisMeshWidgetOps:
    """WidgetOps for displaying the selected hypothesis as a 3D mesh with info.

    Listens to:
      - `"clear_selected_hypothesis"` to remove any displayed mesh and info.
      - `"selected_hypothesis"` to load and show the object mesh with its rotation.

    This class only display the widget and does not publish any messages.

    Attributes:
        plotter: A `vedo.Plotter` used to add and remove actors.
        ycb_loader: Loader that returns a textured `vedo.Mesh` for a YCB object.
        updaters: Two `WidgetUpdater`s for clear and update actions.
        info_widget: The text panel shown alongside the mesh.
    """

    def __init__(self, plotter: Plotter, ycb_loader: YCBMeshLoader) -> None:
        self.plotter = plotter
        self.ycb_loader = ycb_loader
        self.updaters = [
            WidgetUpdater(
                topics=[TopicSpec("clear_selected_hypothesis", required=True)],
                callback=self.clear_mesh,
            ),
            WidgetUpdater(
                topics=[TopicSpec("selected_hypothesis", required=True)],
                callback=self.update_mesh,
            ),
        ]
        self.info_widget: Text2D | None = None

    def clear_mesh(
        self, widget: Mesh | None, msgs: list[TopicMessage]
    ) -> tuple[Any, bool]:
        """Clear the mesh and info panel if present.

        Args:
            widget: Current mesh object, if any.
            msgs: Unused. Present for the updater interface.

        Returns:
            `(widget, False)` to indicate no publish should occur.
        """
        if widget is not None:
            self.plotter.remove(widget)

        if self.info_widget is not None:
            self.plotter.remove(self.info_widget)
            self.info_widget = None

        self.plotter.render()
        return widget, False

    def update_mesh(
        self, widget: Mesh | None, msgs: list[TopicMessage]
    ) -> tuple[Any, bool]:
        """Render the mesh and info panel for the selected hypothesis.

        The hypothesis is expected to be a pandas Series with keys:
        `graph_id`, `Rot_x`, `Rot_y`, `Rot_z`, `age`, `Evidence`, `Evidence Slope`.

        Args:
            widget: Current mesh, if any.
            msgs: Messages received from the `WidgetUpdater` with a single
                `"selected_hypothesis"` message.

        Returns:
            `(new_widget, False)` to indicate no publish should occur.
        """
        # Clear existing mesh and text
        widget, _ = self.clear_mesh(widget, msgs)

        msgs_dict = {msg.name: msg.value for msg in msgs}
        hypothesis = msgs_dict["selected_hypothesis"]

        # Add object mesh
        widget = self.ycb_loader.create_mesh(hypothesis["graph_id"]).clone(deep=True)
        widget.rotate_x(hypothesis["Rot_x"])
        widget.rotate_y(hypothesis["Rot_y"])
        widget.rotate_z(hypothesis["Rot_z"])
        widget.scale(1500)
        widget.pos(1000, 100, -500)
        self.plotter.add(widget)

        # Add info text
        info = (
            f"Object: {hypothesis['graph_id']}\n"
            + f"Age: {hypothesis['age']}\n"
            + f"Evidence: {hypothesis['Evidence']:.2f}\n"
            + f"Evidence Slope: {hypothesis['Evidence Slope']:.2f}"
        )
        info_widget = Text2D(txt=info, pos="top-right")
        self.plotter.add(info_widget)
        self.info_widget = info_widget

        self.plotter.render()

        return widget, False


class InteractivePlot:
    """An interactive plot for correlation of evidence slopes and pose errors.

    This visualization provides means for inspecting the resampling of hypotheses
    at every step. The main view is a scatter correlation plot where pose error is
    expected to decrease as evidence slope increases. You can click points to inspect
    the selected hypothesis and view its 3D mesh with basic stats. Additional controls
    let you switch objects and threshold by hypothesis age.

    Args:
        exp_path: Path to the experiment log consumed by `DataParser`.
        data_path: Root directory containing YCB meshes for `YCBMeshLoader`.

    Attributes:
        data_parser: Parser that reads the JSON log file and serves queries.
        ycb_loader: Loader that provides textured YCB meshes.
        event_bus: Publisher used to route `TopicMessage` events among widgets.
        plotter: Vedo `Plotter` hosting all widgets.
        scheduler: Debounce scheduler bound to the plotter interactor.
        _widgets: Mapping of widget names to their `Widget` instances. It
            includes episode and step sliders, primary/prev/next buttons, an
            age-threshold slider, the correlation plot, and mesh viewers.

    """

    def __init__(
        self,
        exp_path: str,
        data_path: str,
    ):
        self.data_parser = DataParser(exp_path)
        self.ycb_loader = YCBMeshLoader(data_path)
        self.event_bus = Publisher()
        self.plotter = Plotter().render()
        self.scheduler = VtkDebounceScheduler(self.plotter.interactor, period_ms=33)

        # create and add the widgets to the plotter
        self._widgets = self.create_widgets()
        for w in self._widgets.values():
            w.add()
        self._widgets["episode_slider"].set_state(0)
        self._widgets["age_threshold"].set_state(0)

        self.plotter.show(interactive=True, resetcam=False, camera=self.cam_dict())

    def cam_dict(self) -> dict[str, tuple[float, float, float]]:
        """Returns camera parameters for an overhead view of the plot.

        Returns:
            Dictionary with camera position and focal point.
        """
        x_val = 300
        y_val = 200
        z_val = 1500
        return {"pos": (x_val, y_val, z_val), "focal_point": (x_val, y_val, 0)}

    def create_widgets(self):
        widgets = {}

        widgets["episode_slider"] = Widget[Slider2D, int](
            widget_ops=EpisodeSliderWidgetOps(
                plotter=self.plotter,
                data_parser=self.data_parser,
            ),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["step_slider"] = Widget[Slider2D, int](
            widget_ops=StepSliderWidgetOps(
                plotter=self.plotter,
                data_parser=self.data_parser,
            ),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["primary_mesh"] = Widget[Mesh, None](
            widget_ops=GtMeshWidgetOps(
                plotter=self.plotter,
                data_parser=self.data_parser,
                ycb_loader=self.ycb_loader,
            ),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["primary_button"] = Widget[Button, str](
            widget_ops=PrimaryButtonWidgetOps(plotter=self.plotter),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.2,
            dedupe=False,
        )

        widgets["prev_button"] = Widget[Button, str](
            widget_ops=PrevButtonWidgetOps(plotter=self.plotter),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.2,
            dedupe=False,
        )

        widgets["next_button"] = Widget[Button, str](
            widget_ops=NextButtonWidgetOps(plotter=self.plotter),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.2,
            dedupe=False,
        )

        widgets["age_threshold"] = Widget[Slider2D, int](
            widget_ops=AgeThresholdWidgetOps(plotter=self.plotter),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["current_object"] = Widget[None, str](
            widget_ops=CurrentObjectWidgetOps(data_parser=self.data_parser),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.2,
            dedupe=False,
        )

        widgets["click_widget"] = Widget[None, Location3D](
            widget_ops=ClickWidgetOps(plotter=self.plotter, cam_dict=self.cam_dict()),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.1,
            dedupe=True,
        )

        widgets["correlation_plot"] = Widget[None, Series](
            widget_ops=CorrelationPlotWidgetOps(
                plotter=self.plotter, data_parser=self.data_parser
            ),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.3,
            dedupe=False,
        )

        widgets["hypothesis_mesh"] = Widget[Mesh, None](
            widget_ops=HypothesisMeshWidgetOps(
                plotter=self.plotter,
                ycb_loader=self.ycb_loader,
            ),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.0,
            dedupe=False,
        )

        return widgets


def plot_interactive_hypothesis_space_correlation(
    exp_path: str,
    data_path: str,
) -> int:
    """Interactive visualization for inspecting the hypothesis space.

    Args:
        exp_path: Path to the experiment directory containing the detailed stats file.
        data_path: Path to the root directory of YCB object meshes.

    Returns:
        Exit code.
    """
    if not Path(exp_path).exists():
        logger.error(f"Experiment path not found: {exp_path}")
        return 1

    data_path = str(Path(data_path).expanduser())

    InteractivePlot(exp_path, data_path)

    return 0


def add_subparser(
    subparsers: argparse._SubParsersAction,
    parent_parser: argparse.ArgumentParser | None = None,
) -> None:
    """Add the interactive slope and pose error subparser to the main parser.

    Args:
        subparsers: The subparsers object from the main parser.
        parent_parser: Optional parent parser for shared arguments.
    """
    parser = subparsers.add_parser(
        "interactive_hypothesis_space_correlation",
        help="Creates a plot of evidence slope and pose error correlation.",
        parents=[parent_parser] if parent_parser else [],
    )
    parser.add_argument(
        "experiment_log_dir",
        help=(
            "The directory containing the experiment log with the detailed stats file."
        ),
    )
    parser.add_argument(
        "--objects_mesh_dir",
        default="~/tbp/data/habitat/objects/ycb/meshes",
        help=("The directory containing the mesh objects."),
    )
    parser.set_defaults(
        func=lambda args: sys.exit(
            plot_interactive_hypothesis_space_correlation(
                args.experiment_log_dir,
                args.objects_mesh_dir,
            )
        )
    )
