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
import time
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import trimesh
from pandas import DataFrame
from scipy.spatial import cKDTree
from vedo import (
    Button,
    Circle,
    Image,
    Mesh,
    Plotter,
    Text2D,
    settings,
)

from tbp.monty.frameworks.utils.logging_utils import load_stats

if TYPE_CHECKING:
    import argparse

    from vedo import Slider2D

logger = logging.getLogger(__name__)


# Set splitting ratio for renderers, font, and disable immediate_rendering
settings.immediate_rendering = False
settings.default_font = "Theemim"
settings.window_splitting_position = 0.5

HUE_PALETTE = {
    "Added": "#66c2a5",
    "Removed": "#fc8d62",
    "Maintained": "#8da0cb",
}


def get_closest_row(df: pd.DataFrame, slope: float, error: float) -> pd.Series:
    if df.empty:
        raise ValueError("DataFrame is empty.")
    tree = cKDTree(df[["Evidence Slope", "Pose Error"]].values)
    dist, idx = tree.query([slope, error])
    return df.iloc[idx]


class DataExtractor:
    """Extracts and processes data from JSON logs of unsupervised inference experiments.

    Args:
        exp_path: Path to the experiment directory.
        data_path: Path to the root directory containing object meshes.
            default set to `~/tbp/data/habitat/objects/ycb/meshes`
        learning_module: Which learning module to use for data extraction.

    Attributes:
        exp_path: Path to the experiment directory where a `detailed_run_stats.json`
            exists.
        lm: Learning module defined by the user. Defaults to "LM_0".
        data: Dictionary with keys being episodes, and values being slopes, pose errors
            and more.
    """

    def __init__(self, exp_path: str, data_path: str, learning_module: str):
        self.exp_path = exp_path
        self.data_path = data_path
        self.lm = learning_module

        self.read_data()
        self.import_objects()

    def read_data(self) -> None:
        _, _, self.data, _ = load_stats(self.exp_path, False, False, True, False)
        self.object_names = set(self.__getitem__(0)[0].keys())

    def resolve_primary(self, episode: int):
        return self.data[str(episode)]["target"]["primary_target_object"]

    def cycle_graph(self, current, shift_by):
        graphs = list(self.__getitem__(0)[0].keys())
        total_graphs = len(graphs)
        curr_ix = graphs.index(current)
        return graphs[(curr_ix + shift_by) % total_graphs]

    def __getitem__(self, ix: str | int) -> dict:
        if type(ix) is int:
            ix = str(ix)

        return self.data[ix][self.lm]["hypotheses_updater_telemetry"]

    def _find_glb_file(self, obj_name: str) -> str:
        """Search for the .glb.orig file of a given YCB object in a directory.

        Args:
            obj_name: The object name to search for (e.g., "potted_meat_can").

        Returns:
            Full path to the .glb.orig file.

        Raises:
            FileNotFoundError: If the .glb.orig file for the object is not found.

        """
        for path in Path(self.data_path).rglob("*"):
            if path.is_dir() and path.name.endswith(obj_name):
                glb_orig_path = path / "google_16k" / "textured.glb.orig"
                if glb_orig_path.exists():
                    return str(glb_orig_path)

        raise FileNotFoundError(
            f"Could not find .glb.orig file for '{obj_name}' in '{self.data_path}'"
        )

    def create_mesh(self, obj_name: str) -> Mesh:
        """Reads a 3D object file in glb format and returns a Vedo Mesh object.

        Args:
            obj_name: Name of the object to load.

        Returns:
            vedo.Mesh object with UV texture and transformed orientation.
        """
        file_path = self._find_glb_file(obj_name)
        with open(file_path, "rb") as f:
            mesh = trimesh.load_mesh(f, file_type="glb")

        # create mesh from vertices and faces
        obj = Mesh([mesh.vertices, mesh.faces])

        # add texture
        obj.texture(
            tname=np.array(mesh.visual.material.baseColorTexture),
            tcoords=mesh.visual.uv,
        )

        # Shift to geometry mean and rotate to the up/front of the glb
        obj.shift(-np.mean(obj.bounds().reshape(3, 2), axis=1))
        obj.rotate_x(-90)

        return obj

    def import_objects(self) -> None:
        """Load all unique object meshes into memory."""
        self.imported_objects = {
            obj_name: self.create_mesh(obj_name) for obj_name in self.object_names
        }

    def __len__(self) -> int:
        return len(self.data)


class CorrelationPlot:
    """Renders a correlation plot of evidence slopes and pose errors.

    Args:
        data_extractor: A DataExtractor instance with evidence scores and transitions.
        renderer_ix: Index of the Vedo renderer to draw into. Defaults to 0.

    Attributes:
        lines: List of vedo.Line objects, one per object class.
        bg_rects: Colored background rectangles indicating the current target object.
        bg_labels: Text labels above rectangles indicating target object names.
        guide_line: Vertical guide line indicating the current step.
        added_plot_flag: Whether the static elements (lines and background) were added.
    """

    def __init__(self, data_extractor: DataExtractor, renderer_ix: int = 0):
        self.data_extractor = data_extractor
        self.renderer_ix = renderer_ix
        self.fig = None

    def _data_at_ix(self, episode, step, graph_id):
        step_data = self.data_extractor[episode][step][graph_id]
        all_dfs = []

        for input_channel in step_data:
            channel_data = step_data[input_channel]
            updater_data = channel_data["hypotheses_updater"]

            # -- Added --
            add_ids = updater_data.get("add_ids", [])
            if add_ids:
                df_added = DataFrame(
                    {
                        "Evidence Slope": np.array(updater_data["evidence_slopes"])[
                            add_ids
                        ],
                        "Rot_x": np.array(channel_data["rotations"])[add_ids][:, 0],
                        "Rot_y": np.array(channel_data["rotations"])[add_ids][:, 1],
                        "Rot_z": np.array(channel_data["rotations"])[add_ids][:, 2],
                        "Pose Error": np.array(channel_data["pose_errors"])[add_ids],
                        "kind": "Added",
                        "input_channel": input_channel,
                    }
                )
                all_dfs.append(df_added)

            # -- Removed --
            removed_ids = updater_data.get("removed_ids", [])
            if len(removed_ids) > 0:
                # If step is 0, go to last step of previous episode
                if step == 0 and episode == 0:
                    break
                prev_episode = episode - 1 if step == 0 else episode
                prev_step = -1 if step == 0 else step - 1

                prev_data = self.data_extractor[prev_episode][prev_step][graph_id]
                prev_channel = prev_data[input_channel]
                prev_updater = prev_channel["hypotheses_updater"]
                df_removed = DataFrame(
                    {
                        "Evidence Slope": np.array(prev_updater["evidence_slopes"])[
                            removed_ids
                        ],
                        "Rot_x": np.array(prev_channel["rotations"])[removed_ids][:, 0],
                        "Rot_y": np.array(prev_channel["rotations"])[removed_ids][:, 1],
                        "Rot_z": np.array(prev_channel["rotations"])[removed_ids][:, 2],
                        "Pose Error": np.array(prev_channel["pose_errors"])[
                            removed_ids
                        ],
                        "kind": "Removed",
                        "input_channel": input_channel,
                    }
                )
                all_dfs.append(df_removed)

            # -- Maintained --
            total_ids = list(range(len(updater_data["evidence_slopes"])))
            maintained_ids = sorted(set(total_ids) - set(add_ids))
            if maintained_ids:
                df_maintained = DataFrame(
                    {
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

    def __call__(
        self, plotter: Plotter, episode: int, step: int, graph_id: str
    ) -> None:
        """Render evidence lines, background labels, and guide line.

        Args:
            plotter: The vedo.Plotter instance to draw into.
            episode: Current episode index
            step: Current step index
            graph_id: Current object to visualize
        """
        if self.fig is not None:
            plotter.remove(self.fig)

        df = self._data_at_ix(episode=episode, step=step, graph_id=graph_id)
        img = self.create_figure(df)
        self.fig = Image(img)
        plt.close(img)
        plotter.add(self.fig)

    def cam_dict(self) -> dict[str, tuple[float, float, float]]:
        """Returns camera parameters for an overhead view of the plot.

        Returns:
            Dictionary with camera position and focal point.
        """
        x_val = 300
        y_val = 200
        z_val = 1500
        return {"pos": (x_val, y_val, z_val), "focal_point": (x_val, y_val, 0)}


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
        throttle_time: float = 0.3,
    ):
        self.throttle_time = throttle_time
        self.data_extractor = DataExtractor(exp_path, data_path, learning_module)

        self.correlation_plotter = CorrelationPlot(
            data_extractor=self.data_extractor, renderer_ix=0
        )

        self.plotter = Plotter(size=(1000, 1000))

        # Create a slider on the plot
        self.episode_slider = self.plotter.at(0).add_slider(
            self.episode_slider_callback,
            xmin=0,
            xmax=len(self.data_extractor) - 1,
            value=0,
            pos=[(0.1, 0.2), (0.7, 0.2)],
            title="Episode",
        )

        self.episode_curr_slider_val = None
        self.episode_last_call_time = time.time()

        self.step_slider = self.plotter.at(0).add_slider(
            self.step_slider_callback,
            xmin=0,
            xmax=len(self.data_extractor[0]) - 1,
            value=0,
            pos=[(0.1, 0.1), (0.7, 0.1)],
            title="Step",
        )

        self.step_curr_slider_val = None
        self.step_last_call_time = time.time()

        self.primary_button = self.plotter.at(0).add_button(
            self.primary_button_callback,
            pos=(0.85, 0.2),
            states=["Primary Target"],
            size=30,
            font="Calco",
            bold=True,
        )

        self.previous_button = self.plotter.at(0).add_button(
            self.previous_button_callback,
            pos=(0.83, 0.13),
            states=["<"],
            size=30,
            font="Calco",
            bold=True,
        )
        self.next_button = self.plotter.at(0).add_button(
            self.next_button_callback,
            pos=(0.88, 0.13),
            states=[">"],
            size=30,
            font="Calco",
            bold=True,
        )

        self.current_graph = self.data_extractor.resolve_primary(0)
        self.current_graph_label = Text2D(
            txt=self.current_graph,
            pos=(0.85, 0.26),
            s=1.2,
            font="Calco",
            justify="center",
            bold=True,
            c="black",
        )
        self.plotter.at(0).add(self.current_graph_label)
        self.current_graph_object = self.data_extractor.imported_objects[
            self.current_graph
        ].clone(deep=True)
        self.plotter.at(0).add(self.current_graph_object.scale(2000).pos(-200, 250))
        self.predicted_graph_object = None

        self.highlight_circle = None
        self.plotter.add_callback("LeftButtonPress", self.on_scatter_click)

    def highlight_point(self, slope: float, error: float):
        if self.highlight_circle:
            self.plotter.remove(self.highlight_circle)

        pos = self.map_data_coords_to_world(slope, error)
        self.highlight_circle = Circle(pos=pos, r=3.0, res=16).c("red")
        self.plotter.add(self.highlight_circle)
        self.plotter.render()

    def map_click_to_data_coords(self, click_point: np.ndarray) -> tuple[float, float]:
        xmin, xmax, ymin, ymax = 74, 496, 64, 496
        xlim = [-2.0, 2.0]
        ylim = [0, 3.25]

        # Normalize click to image coordinate
        x_rel = (click_point[0] - xmin) / (xmax - xmin)
        y_rel = (click_point[1] - ymin) / (ymax - ymin)

        # Map to data coordinate space
        slope = xlim[0] + x_rel * (xlim[1] - xlim[0])
        error = ylim[0] + y_rel * (ylim[1] - ylim[0])

        return slope, error

    def map_data_coords_to_world(
        self, slope: float, error: float
    ) -> tuple[float, float, float]:
        xmin, xmax, ymin, ymax = 74, 496, 64, 496
        xlim = [-2.0, 2.0]
        ylim = [0, 3.25]

        x_rel = (slope - xlim[0]) / (xlim[1] - xlim[0])
        y_rel = (error - ylim[0]) / (ylim[1] - ylim[0])

        x_world = xmin + x_rel * (xmax - xmin)
        y_world = ymin + y_rel * (ymax - ymin)
        return x_world, y_world, 0.1

    def on_scatter_click(self, event):
        if event.picked3d is None:
            return

        # Ignore clicks outside of scatter plot limits
        (slope, error) = self.map_click_to_data_coords(event.picked3d)
        if not -2.0 <= slope <= 2.0 or not 0 <= error <= 3.25:
            return

        episode = round(self.episode_slider.GetRepresentation().GetValue())
        step = round(self.step_slider.GetRepresentation().GetValue())
        graph_id = self.current_graph

        # Get current dataframe
        df = self.correlation_plotter._data_at_ix(episode, step, graph_id)

        # Get closest point
        closest = get_closest_row(df, slope, error)
        self.highlight_point(closest["Evidence Slope"], closest["Pose Error"])

        if self.predicted_graph_object:
            self.plotter.remove(self.predicted_graph_object)
        self.predicted_graph_object = self.data_extractor.imported_objects[
            graph_id
        ].clone(deep=True)
        self.predicted_graph_object.shift(
            -np.mean(self.predicted_graph_object.bounds().reshape(3, 2), axis=1)
        )
        self.predicted_graph_object.rotate_x(closest["Rot_x"]).rotate_y(
            closest["Rot_y"]
        ).rotate_z(closest["Rot_z"])
        self.predicted_graph_object.scale(2000)
        self.predicted_graph_object.shift(840, 250)

        self.plotter.at(0).add(self.predicted_graph_object)

    def primary_button_callback(self, widget: Button, _event: str) -> None:
        episode_val = round(self.episode_slider.GetRepresentation().GetValue())
        resolved_graph = self.data_extractor.resolve_primary(episode_val)
        if resolved_graph != self.current_graph:
            self.current_graph = resolved_graph
            self.current_graph_label.text(self.current_graph)

            # Replace primary object
            if self.current_graph_object:
                self.plotter.remove(self.current_graph_object)
            self.current_graph_object = self.data_extractor.imported_objects[
                self.current_graph
            ].clone(deep=True)
            self.plotter.at(0).add(self.current_graph_object.scale(2000).pos(-200, 250))

            self.step_slider.representation.SetValue(0)
            self.step_slider_callback(self.step_slider, "force_run")

    def previous_button_callback(self, widget: Button, _event: str) -> None:
        resolved_graph = self.data_extractor.cycle_graph(self.current_graph, -1)
        if resolved_graph != self.current_graph:
            self.current_graph = resolved_graph
            self.current_graph_label.text(self.current_graph)
            self.step_slider.representation.SetValue(0)
            self.step_slider_callback(self.step_slider, "force_run")

    def next_button_callback(self, widget: Button, _event: str) -> None:
        resolved_graph = self.data_extractor.cycle_graph(self.current_graph, 1)
        if resolved_graph != self.current_graph:
            self.current_graph = resolved_graph
            self.current_graph_label.text(self.current_graph)
            self.step_slider.representation.SetValue(0)
            self.step_slider_callback(self.step_slider, "force_run")

    def episode_slider_callback(self, widget: Slider2D, event: str) -> None:
        """Respond to episode change by updating the visualization.

        Note: This function is throttled to prevent recursion depth errors
            while continue to be responsive.
        """
        episode_val = round(widget.GetRepresentation().GetValue())
        if event == "force_run" or (
            episode_val != self.episode_curr_slider_val
            and time.time() - self.episode_last_call_time > self.throttle_time
        ):
            self.episode_curr_slider_val = episode_val
            self.episode_last_call_time = time.time()

            self.primary_button_callback(self.primary_button, "")

    def step_slider_callback(self, widget: Slider2D, event: str) -> None:
        """Respond to episode change by updating the visualization.

        Note: This function is throttled to prevent recursion depth errors
            while continue to be responsive.
        """
        step_val = round(widget.GetRepresentation().GetValue())
        if event == "force_run" or (
            step_val != self.step_curr_slider_val
            and time.time() - self.step_last_call_time > self.throttle_time
        ):
            episode_val = round(self.episode_slider.GetRepresentation().GetValue())
            self.correlation_plotter(
                plotter=self.plotter,
                episode=episode_val,
                step=step_val,
                graph_id=self.current_graph,
            )

            if self.highlight_circle:
                self.plotter.remove(self.highlight_circle)
                self.highlight_circle = None

            if self.predicted_graph_object:
                self.plotter.remove(self.predicted_graph_object)
                self.predicted_graph_object = None

            self.step_curr_slider_val = step_val
            self.step_last_call_time = time.time()
            self.render()

    def render(self, resetcam: bool = False) -> None:
        """Render the visualization layout.

        Args:
            resetcam: If True, resets camera for all renderers.
        """
        self.plotter.render()
        self.plotter.at(self.correlation_plotter.renderer_ix).show(
            camera=self.correlation_plotter.cam_dict(),
            resetcam=resetcam,
            interactive=True,
        )


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
    plot.render(resetcam=False)

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
