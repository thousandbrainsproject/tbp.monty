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
from typing import TYPE_CHECKING, Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from pubsub.core import Publisher
from vedo import Image, Plotter

from tools.plot.interactive.data import (
    DataLocator,
    DataLocatorStep,
    DataParser,
    YCBMeshLoader,
)
from tools.plot.interactive.widgets import (
    TopicMessage,
    TopicSpec,
    VtkDebounceScheduler,
    Widget,
    WidgetUpdater,
    set_slider_state,
)

if TYPE_CHECKING:
    import argparse


logger = logging.getLogger(__name__)


# Set splitting ratio for renderers, font, and disable immediate_rendering
# settings.immediate_rendering = True
# settings.default_font = "Theemim"
# settings.window_splitting_position = 0.2

HUE_PALETTE = {
    "Added": "#66c2a5",
    "Removed": "#fc8d62",
    "Maintained": "#8da0cb",
}


class EpisodeSliderWidgetOps:
    def __init__(self, plotter, data_parser):
        self.plotter = plotter
        self.data_parser = data_parser

        self.updaters = []
        self._add_kwargs = dict(
            xmin=0, xmax=10, value=0, pos=[(0.1, 0.2), (0.7, 0.2)], title="Episode"
        )

        self._locators = self.create_locators()

    def create_locators(self):
        locators = {}

        locators["episode"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
            ],
        )
        return locators

    def add(self, callback: Callable) -> int:
        kwargs = deepcopy(self._add_kwargs)
        locator = self._locators["episode"]
        kwargs.update({"xmax": len(self.data_parser.query(locator)) - 1})
        return self.plotter.add_slider(callback, **kwargs)

    def remove(self, widget: Any):
        self.plotter.remove(widget)

    def extract_state(self, widget: Any) -> int:
        return round(widget.GetRepresentation().GetValue())

    def set_state(self, widget: Any, value: Any) -> None:
        set_slider_state(widget, value)

    def state_to_messages(self, state: int) -> list[TopicMessage]:
        messages = [TopicMessage(name="episode_number", value=state)]
        return messages


class StepSliderWidgetOps:
    def __init__(self, plotter, data_parser):
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

    def create_locators(self):
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

    def add(self, callback: Callable) -> int:
        return self.plotter.add_slider(callback, **self._add_kwargs)

    def remove(self, widget: Any):
        self.plotter.remove(widget)

    def extract_state(self, widget: Any) -> int:
        return round(widget.GetRepresentation().GetValue())

    def set_state(self, widget: Any, value: Any) -> None:
        return set_slider_state(widget, value)

    def state_to_messages(self, state: int) -> list[TopicMessage]:
        messages = [TopicMessage(name="step_number", value=state)]
        return messages

    def update_slider_range(
        self, widget: Any, msgs: list[TopicMessage]
    ) -> tuple[Any, bool]:
        for msg in msgs:
            if msg.name == "episode_number":
                episode_number = msg.value
                break

        # set widget range to the correct step number
        widget.range = [
            0,
            len(
                self.data_parser.query(
                    self._locators["step"], episode=str(episode_number)
                )
            )
            - 1,
        ]

        # set slider value back to zero
        self.set_state(widget, 0)

        # render the plotter to show the changes
        self.plotter.render()

        return widget, True


class GtMeshWidgetOps:
    def __init__(self, plotter, data_parser, ycb_loader=None):
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

    def create_locators(self):
        locators = {}
        locators["target"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="lm", value="target"),
            ]
        )
        return locators

    def add(self, callback: Callable) -> int:
        return None

    def remove(self, widget: Any):
        self.plotter.remove(widget)

    def extract_state(self, widget: Any) -> int:
        return None

    def state_to_messages(self, state: int) -> list[TopicMessage]:
        return []

    def update_mesh(self, widget: Any, msgs: list[TopicMessage]) -> tuple[Any, bool]:
        if widget is not None:
            self.remove(widget)

        for msg in msgs:
            if msg.name == "episode_number":
                episode_number = msg.value
                break

        locator = self._locators["target"]
        target = self.data_parser.extract(locator, episode=str(episode_number))
        target_id = target["primary_target_object"]
        target_rot = target["primary_target_rotation_euler"]
        widget = self.ycb_loader.create_mesh(target["primary_target_object"]).clone(
            deep=True
        )
        widget.rotate_x(target_rot[0])
        widget.rotate_y(target_rot[1])
        widget.rotate_z(target_rot[2])
        widget.scale(1500)

        widget.shift(-np.mean(widget.bounds().reshape(3, 2), axis=1))
        widget.pos(-200, 250)
        self.plotter.add(widget)

        self.plotter.render()

        return widget, False


class PrimaryButtonWidgetOps:
    def __init__(self, plotter):
        self.plotter = plotter
        self.updaters = []

        self._add_kwargs = dict(
            pos=(0.85, 0.2),
            states=["Primary Target"],
            c=["w"],
            bc=["dg"],
            size=30,
            font="Calco",
            bold=True,
        )
        self._locators = self.create_locators()

        self.objects_list = []
        self.current_object = None

    def create_locators(self):
        return {}

    def add(self, callback: Callable) -> int:
        return self.plotter.add_button(callback, **self._add_kwargs)

    def remove(self, widget: Any):
        self.plotter.remove(widget)

    def extract_state(self, widget: Any) -> bool:
        return True

    def state_to_messages(self, state: str) -> list[TopicMessage]:
        messages = [
            TopicMessage(name="primary_button", value=True),
        ]
        return messages


class PrevButtonWidgetOps:
    def __init__(self, plotter):
        self.plotter = plotter
        self.updaters = []

        self._add_kwargs = dict(
            pos=(0.83, 0.13),
            states=["<"],
            c=["w"],
            bc=["dg"],
            size=30,
            font="Calco",
            bold=True,
        )
        self._locators = self.create_locators()

        self.objects_list = []
        self.current_object = None

    def create_locators(self):
        return {}

    def add(self, callback: Callable) -> int:
        return self.plotter.add_button(callback, **self._add_kwargs)

    def remove(self, widget: Any):
        self.plotter.remove(widget)

    def extract_state(self, widget: Any) -> bool:
        return True

    def state_to_messages(self, state: str) -> list[TopicMessage]:
        messages = [
            TopicMessage(name="prev_button", value=True),
        ]
        return messages


class NextButtonWidgetOps:
    def __init__(self, plotter):
        self.plotter = plotter
        self.updaters = []

        self._add_kwargs = dict(
            pos=(0.88, 0.13),
            states=[">"],
            c=["w"],
            bc=["dg"],
            size=30,
            font="Calco",
            bold=True,
        )
        self._locators = self.create_locators()

        self.objects_list = []
        self.current_object = None

    def create_locators(self):
        return {}

    def add(self, callback: Callable) -> int:
        return self.plotter.add_button(callback, **self._add_kwargs)

    def remove(self, widget: Any):
        self.plotter.remove(widget)

    def extract_state(self, widget: Any) -> bool:
        return True

    def state_to_messages(self, state: str) -> list[TopicMessage]:
        messages = [
            TopicMessage(name="next_button", value=True),
        ]
        return messages


class CurrentObjectWidgetOps:
    def __init__(self, data_parser):
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

        self._add_kwargs = dict(
            pos=(0.83, 0.13),
            states=["<"],
            c=["w"],
            bc=["dg"],
            size=30,
            font="Calco",
            bold=True,
        )
        self._locators = self.create_locators()

        self.objects_list = []
        self.current_object_ix = None

    def create_locators(self):
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

    def add(self, callback: Callable) -> int:
        obj_list_locator = self._locators["objects_list"]
        self.objects_list = self.data_parser.query(
            obj_list_locator,
            episode="0",
            step=0,
        )
        return None

    def remove(self, widget: Any):
        pass

    def extract_state(self, widget: Any) -> str:
        state = self.objects_list[self.current_object_ix]
        return state

    def state_to_messages(self, state: str) -> list[TopicMessage]:
        messages = [
            TopicMessage(name="current_object", value=state),
        ]
        return messages

    def update_to_primary(self, widget: Any, msgs: list[TopicMessage]) -> tuple(
        Any, bool
    ):
        msgs_dict = {msg.name: msg.value for msg in msgs}

        target_locator = self._locators["target"]
        current_object = self.data_parser.extract(
            target_locator,
            episode=str(msgs_dict["episode_number"]),
        )["primary_target_object"]
        self.current_object_ix = self.objects_list.index(current_object)

        return widget, True

    def update_current_object(self, widget: Any, msgs: list[TopicMessage]) -> tuple(
        Any, bool
    ):
        # This callback listens to a single topic
        assert len(msgs) == 1

        if msgs[0].name == "prev_button":
            self.current_object_ix -= 1
        elif msgs[0].name == "next_button":
            self.current_object_ix += 1

        self.current_object_ix %= len(self.objects_list)
        return widget, True


class ClickWidgetOps:
    def __init__(self, plotter: Plotter):
        self.plotter = plotter
        self.updaters = []

    def add(self, callback: Callable) -> int:
        self._on_change_cb = callback
        self.plotter.add_callback("LeftButtonPress", self.on_click)

    def remove(self, widget: Any):
        self.plotter.remove_callback("LeftButtonPress")

    def extract_state(self, widget: Any) -> str:
        return self.click_location

    def state_to_messages(self, state: str) -> list[TopicMessage]:
        messages = [
            TopicMessage(name="LeftButtonPress", value=state),
        ]
        return messages

    def on_click(self, event):
        location = event.picked2d

        if location is None:
            return

        self.click_location = location
        self._on_change_cb(widget=None, _event="")


class CorrelationPlotWidgetOps:
    def __init__(self, plotter: Plotter, data_parser: DataParser):
        self.plotter = plotter
        self.data_parser = data_parser
        self.updaters = [
            WidgetUpdater(
                topics=[
                    TopicSpec("episode_number", required=True),
                    TopicSpec("step_number", required=True),
                    TopicSpec("current_object", required=True),
                ],
                callback=self.update_plot,
            ),
        ]

        self._locators = self.create_locators()

    def create_locators(self):
        locators = {}
        locators["input_channel"] = DataLocator(
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
        locators["pose_error"] = locators["input_channel"].extend(
            steps=[
                DataLocatorStep.key(name="stat", value="pose_errors"),
            ],
        )
        locators["evidence"] = locators["input_channel"].extend(
            steps=[
                DataLocatorStep.key(name="stat", value="evidence"),
            ],
        )
        locators["rotations"] = locators["input_channel"].extend(
            steps=[
                DataLocatorStep.key(name="stat", value="rotations"),
            ],
        )
        locators["added_ids"] = locators["input_channel"].extend(
            steps=[
                DataLocatorStep.key(name="stat", value="hypotheses_updater"),
                DataLocatorStep.key(name="updater_stats", value="added_ids"),
            ],
        )
        locators["ages"] = locators["input_channel"].extend(
            steps=[
                DataLocatorStep.key(name="stat", value="hypotheses_updater"),
                DataLocatorStep.key(name="updater_stats", value="ages"),
            ],
        )
        locators["removed_ids"] = locators["input_channel"].extend(
            steps=[
                DataLocatorStep.key(name="stat", value="hypotheses_updater"),
                DataLocatorStep.key(name="updater_stats", value="removed_ids"),
            ],
        )
        locators["evidence_slopes"] = locators["input_channel"].extend(
            steps=[
                DataLocatorStep.key(name="stat", value="hypotheses_updater"),
                DataLocatorStep.key(name="updater_stats", value="evidence_slopes"),
            ],
        )
        return locators

    def add(self, callback: Callable) -> None:
        pass

    def remove(self, widget: Any):
        if widget is not None:
            self.plotter.remove(widget)

    def extract_state(self, widget: Any) -> str:
        return ""

    def state_to_messages(self, state: str) -> list[TopicMessage]:
        messages = []
        return messages

    def generate_df(self, episode, step, graph_id):
        input_channels = self.data_parser.query(
            self._locators["input_channel"],
            episode=str(episode),
            step=step,
            obj=graph_id,
        )

        all_dfs = []
        for input_channel in input_channels:
            channel_data = self.data_parser.extract(
                self._locators["input_channel"],
                episode=str(episode),
                step=step,
                obj=graph_id,
                channel=input_channel,
            )
            updater_data = channel_data["hypotheses_updater"]

            # -- Added --
            added_ids = updater_data.get("added_ids", [])
            if added_ids:
                df_added = DataFrame(
                    {
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

            # -- Removed --
            removed_ids = updater_data.get("removed_ids", [])
            if len(removed_ids) > 0:
                # If step is 0, go to last step of previous episode
                if episode == 0 and step == 0:
                    break
                prev_episode = episode - 1 if step == 0 else episode
                prev_step = -1 if step == 0 else step - 1

                prev_channel_data = self.data_parser.extract(
                    self._locators["input_channel"],
                    episode=str(prev_episode),
                    step=prev_step,
                    obj=graph_id,
                    channel=input_channel,
                )
                prev_updater_data = channel_data["hypotheses_updater"]

                df_removed = DataFrame(
                    {
                        "Evidence": np.array(prev_channel_data["evidence"])[
                            removed_ids
                        ],
                        "Evidence Slope": np.array(
                            prev_updater_data["evidence_slopes"]
                        )[removed_ids],
                        "Rot_x": np.array(prev_channel_data["rotations"])[removed_ids][
                            :, 0
                        ],
                        "Rot_y": np.array(prev_channel_data["rotations"])[removed_ids][
                            :, 1
                        ],
                        "Rot_z": np.array(prev_channel_data["rotations"])[removed_ids][
                            :, 2
                        ],
                        "Pose Error": np.array(prev_channel_data["pose_errors"])[
                            removed_ids
                        ],
                        "age": np.array(prev_updater_data["ages"])[removed_ids],
                        "kind": "Removed",
                        "input_channel": input_channel,
                    }
                )
                all_dfs.append(df_removed)

            # -- Maintained --
            total_ids = list(range(len(updater_data["evidence_slopes"])))
            maintained_ids = sorted(set(total_ids) - set(added_ids))
            if maintained_ids:
                df_maintained = DataFrame(
                    {
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

    def create_figure(self, df: DataFrame):
        g = sns.JointGrid(data=df, x="Evidence Slope", y="Pose Error", height=6)

        sns.scatterplot(
            data=df,
            x="Evidence Slope",
            y="Pose Error",
            hue="kind",
            ax=g.ax_joint,
            s=8,
            alpha=0.8,
            palette=HUE_PALETTE,
        )

        sns.kdeplot(
            data=df,
            x="Evidence Slope",
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
            y="Pose Error",
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
        g.ax_joint.set_xlabel("Evidence Slope", labelpad=10)
        g.ax_joint.set_ylabel("Pose Error", labelpad=10)
        g.fig.tight_layout()
        return g.fig

    def update_plot(self, widget: Any, msgs: list[TopicMessage]) -> tuple[Any, bool]:
        self.remove(widget)

        msgs_dict = {msg.name: msg.value for msg in msgs}
        df = self.generate_df(
            episode=msgs_dict["episode_number"],
            step=msgs_dict["step_number"],
            graph_id=msgs_dict["current_object"],
        )
        fig = self.create_figure(df)
        widget = Image(fig)
        plt.close(fig)

        self.plotter.add(widget)
        self.plotter.render()
        return widget, False


class InteractivePlot:
    """An interactive plot for correlation of evidence slopes and pose errors.

    Args:
        exp_path: Path to the JSON directory containing detailed run statistics.
        data_path: Path to the root directory of YCB object meshes.
        learning_module: Which learning module to use for data extraction.
        throttle_time: Minimum delay between slider callbacks (seconds).
            Defaults to 0.2 seconds.

    Attributes:
        throttle_time: Minimum delay between slider callbacks (seconds).
        data_extractor: Instance of DataExtractor for parsing json data.
        gt_sim: GroundTruthSimulator for rendering sensor and target objects.
        mlh_sim: MlhSimulator for visualizing most likely hypotheses.
        correlation_plotter: EvidencePlot for plotting evidence scores.
        plotter: The main vedo.Plotter instance managing multiple renderers.
        slider: The step slider widget.
        curr_slider_val: The last processed slider value.
        last_call_time: Timestamp of last callback execution (for throttling).
    """

    def __init__(
        self,
        exp_path: str,
        data_path: str,
        learning_module: str,
    ):
        self.data_parser = DataParser(exp_path)
        self.ycb_loader = YCBMeshLoader(data_path)
        self.event_bus = Publisher()
        self.plotter = Plotter(size=(1000, 1000), sharecam=False).render()
        self.scheduler = VtkDebounceScheduler(self.plotter.interactor, period_ms=33)

        # create and add the widgets to the plotter
        self._widgets = self.create_widgets()
        for w in self._widgets.values():
            w.add()

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

        widgets["episode_slider"] = Widget(
            widget_ops=EpisodeSliderWidgetOps(
                plotter=self.plotter,
                data_parser=self.data_parser,
            ),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["step_slider"] = Widget(
            widget_ops=StepSliderWidgetOps(
                plotter=self.plotter,
                data_parser=self.data_parser,
            ),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["primary_mesh"] = Widget(
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

        widgets["primary_button"] = Widget(
            widget_ops=PrimaryButtonWidgetOps(plotter=self.plotter),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.2,
            dedupe=False,
        )

        widgets["prev_button"] = Widget(
            widget_ops=PrevButtonWidgetOps(plotter=self.plotter),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.2,
            dedupe=False,
        )

        widgets["next_button"] = Widget(
            widget_ops=NextButtonWidgetOps(plotter=self.plotter),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.2,
            dedupe=False,
        )

        widgets["current_object"] = Widget(
            widget_ops=CurrentObjectWidgetOps(data_parser=self.data_parser),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.2,
            dedupe=True,
        )

        widgets["click_widget"] = Widget(
            widget_ops=ClickWidgetOps(plotter=self.plotter),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.1,
            dedupe=True,
        )

        widgets["correlation_plot"] = Widget(
            widget_ops=CorrelationPlotWidgetOps(
                plotter=self.plotter, data_parser=self.data_parser
            ),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.3,
            dedupe=True,
        )

        return widgets


def plot_interactive_hypothesis_space_correlation(
    exp_path: str,
    data_path: str,
    learning_module: str,
) -> int:
    """Interactive visualization for unsupervised inference experiments.

    This visualization provides a 3-pane renderers to allow for inspecting the objects,
    MLH, and sensor locations while stepping through the maximum evidence scores for
    each object.

    Args:
        exp_path: Path to the experiment directory containing the detailed stats file.
        data_path: Path to the root directory of YCB object meshes.
        learning_module: The learning module to use for extracting evidence data.

    Returns:
        Exit code.
    """
    if not Path(exp_path).exists():
        logger.error(f"Experiment path not found: {exp_path}")
        return 1

    data_path = str(Path(data_path).expanduser())

    plot = InteractivePlot(exp_path, data_path, learning_module)

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
    parser.add_argument(
        "-lm",
        "--learning_module",
        default="LM_0",
        help='The name of the learning module (default: "LM_0").',
    )
    parser.set_defaults(
        func=lambda args: sys.exit(
            plot_interactive_hypothesis_space_correlation(
                args.experiment_log_dir, args.objects_mesh_dir, args.learning_module
            )
        )
    )
