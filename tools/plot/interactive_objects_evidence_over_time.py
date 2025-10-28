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
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from vedo import (
    Line,
    Mesh,
    Plotter,
    Rectangle,
    Sphere,
    Text2D,
    Text3D,
    settings,
)

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.ycb import DISTINCT_OBJECTS
from tbp.monty.frameworks.utils.logging_utils import load_stats

if TYPE_CHECKING:
    import argparse

    from vedo import Button, Slider2D

logger = logging.getLogger(__name__)


# Set splitting ratio for renderers, font, and disable immediate_rendering
settings.immediate_rendering = False
settings.default_font = "Theemim"
settings.window_splitting_position = 0.5


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
        data_path: Path to the root directory containing object meshes.
        target_names: List of target object names per timestep.
        target_positions: List of target object positions per timestep.
        target_rotations: List of target object rotations per timestep.
        mlh_names: List of MLH (most likely hypothesis) object names per timestep.
        mlh_positions: List of MLH sensor positions per timestep.
        mlh_rotations: List of MLH object rotations per timestep.
        sensor_positions: List of sensor (agent) positions per timestep.
        patch_locations: List of observed patch locations per timestep.
        target_transitions: List of episode lengths (number of LM steps).
        primary_target_objects: List of primary target object names per episode.
        classes: Dictionary mapping object name to evidence score lists over time.
        imported_objects: Dictionary of vedo.Mesh objects keyed by object name.
    """

    def __init__(self, exp_path: str, data_path: str, learning_module: str):
        self.exp_path = exp_path
        self.data_path = data_path
        self.lm = learning_module

        self.extract_data()
        self.import_objects()

    def extract_data(self) -> None:
        _, _, detailed_stats, _ = load_stats(self.exp_path, False, False, True, False)

        # Initialize the lists for tracking experiment states across time steps
        self.target_names: list[str] = []
        self.target_positions: list[list[float]] = []
        self.target_rotations: list[list[float]] = []

        self.mlh_names: list[str] = []
        self.mlh_positions: list[list[float]] = []
        self.mlh_rotations: list[list[float]] = []

        self.sensor_positions: list[list[float]] = []
        self.patch_locations: list[list[float]] = []

        self.target_transitions: list[int] = []
        self.primary_target_objects: list[str] = []

        # Evidence scores per object class
        self.classes: dict[str, list[float]] = {
            k: [] for k in detailed_stats["0"][self.lm]["max_evidence"][0].keys()
        }

        for episode_data in detailed_stats.values():
            self.target_transitions.append(len(episode_data[self.lm]["max_evidence"]))
            self.primary_target_objects.append(
                episode_data["target"]["primary_target_object"]
            )
            self.patch_locations.extend(episode_data[self.lm]["locations"]["patch"])

            for ts, evidence_data in enumerate(episode_data[self.lm]["max_evidence"]):
                # Max evidence scores
                for k, v in evidence_data.items():
                    self.classes[k].append(v)

                # Target data (ground truth)
                self.target_names.append(
                    episode_data["target"]["primary_target_object"]
                )
                self.target_positions.append(
                    episode_data["target"]["primary_target_position"]
                )
                self.target_rotations.append(
                    episode_data["target"]["primary_target_rotation_euler"]
                )

                # MLH data
                self.mlh_names.append(
                    episode_data[self.lm]["current_mlh"][ts]["graph_id"]
                )
                self.mlh_positions.append(
                    episode_data[self.lm]["current_mlh"][ts]["location"]
                )
                self.mlh_rotations.append(
                    episode_data[self.lm]["current_mlh"][ts]["rotation"]
                )

            # Not all agent positions and rotations are processed by the LM, we want
            # to extract the processed steps only.
            processed_steps = episode_data[self.lm]["lm_processed_steps"]
            for processed_step, seq_step in zip(
                processed_steps, episode_data["motor_system"]["action_sequence"]
            ):
                if processed_step:
                    self.sensor_positions.append(
                        seq_step[1][AgentID("agent_id_0")]["position"]
                    )

        # Make sure that the agent positions (sampled by processed steps) is the same
        # length as the other logged data
        assert (
            len(self.target_positions)
            == len(self.sensor_positions)
            == len(self.patch_locations)
        )

    def __len__(self) -> int:
        return len(self.target_names)

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
        objects_to_import = {*self.mlh_names, *self.target_names}
        self.imported_objects = {
            obj_name: self.create_mesh(obj_name) for obj_name in objects_to_import
        }


class GroundTruthSimulator:
    """Simulates the ground truth target object and sensor pose over time.

    Args:
        data_extractor: The DataExtractor instance containing episode data and meshes.
        renderer_ix: Index of the Vedo renderer to draw into. Defaults to 0.

    Attributes:
        target_object: Current vedo.Mesh representing the target object.
        sensor_object: Current vedo.Sphere representing the sensor position.
        line_object: Current vedo.Line from sensor to patch on object.
        title_object: Text label title for simulator.
        target_name: The current object name being visualized.
    """

    def __init__(self, data_extractor: DataExtractor, renderer_ix: int = 0):
        self.data_extractor = data_extractor
        self.renderer_ix = renderer_ix

        self.target_object = None
        self.sensor_object = None
        self.line_object = None
        self.title_object = None
        self.target_name = None

    def data_at_step(
        self, step: int
    ) -> tuple[str, list[float], list[float], list[float], list[float]]:
        """Extract all relevant data for a given timestep.

        Args:
            step: The step index.

        Returns:
            Tuple containing target name, position, rotation, sensor position,
            and patch location.
        """
        return (
            self.data_extractor.target_names[step],
            self.data_extractor.target_positions[step],
            self.data_extractor.target_rotations[step],
            self.data_extractor.sensor_positions[step],
            self.data_extractor.patch_locations[step],
        )

    def __call__(self, plotter: Plotter, step: int = 0) -> None:
        """Render the target object, sensor, and viewing ray at a specific timestep.

        Args:
            plotter: The vedo.Plotter instance with multiple renderers.
            step: The timestep to visualize.
        """
        # Extract relevant time-step data
        obj_name, target_pos, target_rot, sensor_pos, patch_loc = self.data_at_step(
            step
        )

        # Check if target object does not exist or needs updating
        if (self.target_name is None) or (self.target_name != obj_name):
            target_obj = self.data_extractor.imported_objects[obj_name].clone(deep=True)
            target_obj.rotate_x(target_rot[0])
            target_obj.rotate_y(target_rot[1])
            target_obj.rotate_z(target_rot[2])
            target_obj.shift(*target_pos)

            if self.target_name is not None:
                plotter.at(self.renderer_ix).remove(self.target_object)
            plotter.at(self.renderer_ix).add(target_obj)

            self.target_object = target_obj
            self.target_name = obj_name

        # Add title label if it hasn't been created
        if self.title_object is None:
            title_obj = Text2D("Simulator", s=1.0).pos((0.44, 1))
            self.title_object = title_obj
            plotter.at(self.renderer_ix).add(title_obj)

        # Add or update sensor sphere
        if self.sensor_object is None:
            sensor_obj = Sphere(pos=sensor_pos, r=0.002)
            self.sensor_object = sensor_obj
            plotter.at(self.renderer_ix).add(sensor_obj)
        self.sensor_object.pos(sensor_pos)

        # Add or update the viewing ray line
        if self.line_object is None:
            line_obj = Line(sensor_pos, patch_loc, c="black", lw=2)
            self.line_object = line_obj
            plotter.at(self.renderer_ix).add(line_obj)
        self.line_object.points = [sensor_pos, patch_loc]

    def axes_dict(self) -> dict[str, tuple[float, float]]:
        """Returns axis ranges.

        Note:
            Monty translates the object by 1.5 in the y-direction.
        """
        return {
            "xrange": (-0.05, 0.05),
            "yrange": (1.45, 1.55),
            "zrange": (-0.05, 0.05),
        }


class MlhSimulator:
    """Plots the most likely hypothesis (MLH) object over time.

    Args:
        data_extractor: The DataExtractor instance containing episode data and meshes.
        renderer_ix: Index of the Vedo renderer to draw into. Defaults to 0.

    Attributes:
        mlh_object: The current vedo.Mesh representing the MLH object.
        loc_object: A small sphere indicating the MLH estimated location of the sensor.
        title_object: Title of the current plot.
        mlh_name: The currently displayed MLH object name.
    """

    def __init__(self, data_extractor: DataExtractor, renderer_ix: int = 0):
        self.data_extractor = data_extractor
        self.renderer_ix = renderer_ix

        self.mlh_object = None
        self.loc_object = None
        self.title_object = None
        self.mlh_name = None

    def data_at_step(
        self, step: int
    ) -> tuple[str, list[float], list[float], list[float]]:
        """Get MLH data for a given timestep.

        Args:
            step: The step index.

        Returns:
            Tuple containing MLH object name, target position (for shifting),
            MLH estimated position, and MLH rotation.
        """
        return (
            self.data_extractor.mlh_names[step],
            self.data_extractor.target_positions[step],
            self.data_extractor.mlh_positions[step],
            self.data_extractor.mlh_rotations[step],
        )

    def __call__(self, plotter: Plotter, step: int = 0) -> None:
        """Render the MLH object and sensor location hypothesis.

        Args:
            plotter: The vedo.Plotter instance with multiple renderers.
            step: The timestep to visualize.
        """
        obj_name, target_pos, mlh_pos, mlh_rot = self.data_at_step(step)

        # Create a transformed clone of the MLH mesh
        mlh_obj = self.data_extractor.imported_objects[obj_name].clone(deep=True)
        mlh_obj.rotate_x(mlh_rot[0])
        mlh_obj.rotate_y(mlh_rot[1])
        mlh_obj.rotate_z(mlh_rot[2])
        mlh_obj.shift(*target_pos)
        if self.mlh_name is not None:
            plotter.at(self.renderer_ix).remove(self.mlh_object)
        plotter.at(self.renderer_ix).add(mlh_obj)

        self.mlh_object = mlh_obj
        self.mlh_name = obj_name

        # Create or update the MLH location indicator
        if self.loc_object is None:
            loc_obj = Sphere(pos=mlh_pos, r=0.002, c="green")
            self.loc_object = loc_obj
            plotter.at(self.renderer_ix).add(loc_obj)
        self.loc_object.pos(mlh_pos)

        # Add title once if it doesn't exist
        if self.title_object is None:
            title_obj = Text2D("Most Likely Hypothesis", s=1.0).pos((0.38, 1))
            self.title_object = title_obj
            plotter.at(self.renderer_ix).add(title_obj)

    def axes_dict(self) -> dict[str, tuple[float, float]]:
        """Returns axis ranges.

        Note:
            Monty translates the object by 1.5 in the y-direction.
        """
        return {
            "xrange": (-0.05, 0.05),
            "yrange": (1.45, 1.55),
            "zrange": (-0.05, 0.05),
        }


class EvidencePlot:
    """Renders a line plot of evidence scores for object classes over time.

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

        self.classes: dict[str, list[float]] = self.data_extractor.classes
        self.target_transitions: list[int] = self.data_extractor.target_transitions
        self.primary_target_objects: list[str] = (
            self.data_extractor.primary_target_objects
        )

        object_names = list(self.classes.keys())

        # Create a color mapping for distinct YCB object names
        # NOTE: This only works for a maximum of 10 objects, as constrained by the
        # color map (`tab10`) and the number of objects in `DISTINCT_OBJECTS`.
        cmap = plt.cm.tab10
        num_colors = len(DISTINCT_OBJECTS)
        colors = {
            obj: cmap(i / num_colors)[:3] for i, obj in enumerate(DISTINCT_OBJECTS)
        }

        self.lines: list[Line] = []
        self.bg_rects: list[Rectangle] = []
        self.bg_labels: list[Text3D] = []

        # Create background rectangles and labels
        prev_x = 0
        for obj, x in zip(self.primary_target_objects, self.target_transitions):
            color = colors.get(obj, (0.7, 0.7, 0.7))
            x_start = prev_x
            x_end = prev_x + x
            y_top = 80.0
            y_bottom = 80.0 - 3.0

            # Create background rectangle
            rect = Rectangle(
                p1=(x_start, y_bottom, 0),
                p2=(x_end, y_top, 0),
                res=(1, 1),
                c=color,
                alpha=1.0,
            )
            self.bg_rects.append(rect)

            # Add label centered above the rectangle
            label = Text3D(
                obj,
                pos=((x_start + x_end) / 2, y_top + 3, 0),
                s=3.0,
                justify="center",
                c=color,
                literal=True,
            )
            self.bg_labels.append(label)

            prev_x = x_end

        # Create lines for evidence scores
        for obj in object_names:
            evidence_scores = self.classes[obj]
            steps = np.arange(len(evidence_scores))
            pts = np.c_[steps, evidence_scores, np.zeros_like(steps)]
            line = Line(pts, c=colors.get(obj, (0.5, 0.5, 0.5)), lw=3)
            self.lines.append(line)

        # Add dynamic guide line. This will be modified with the plot slider.
        self.guide_line = Line(p0=(0, 0, 0), p1=(0, 80, 0), lw=2, c="black")
        self.added_plot_flag = False

    def axes_dict(self) -> dict[str, Any]:
        """Returns axes settings for configuring the vedo Plotter.

        Returns:
            Dictionary of axis parameters.
        """
        total_steps = sum(self.target_transitions)

        return dict(
            xtitle="Timesteps",
            xtitle_position=0.6,
            xtitle_offset=0.1,
            ytitle="Evidence Score",
            ytitle_position=0.85,
            ytitle_offset=0.03,
            yminor_ticks=1,
            xrange=(0, total_steps),
            yrange=(0, 80),
            zrange=(0, 0),
            xygrid=True,
            yzgrid=False,
            zxgrid=False,
            htitle="Evidence Scores Over Time with Resampling",
            htitle_offset=(0.1, 0.1, 0),
            number_of_divisions=10,
        )

    def __call__(self, plotter: Plotter, step: int = 0) -> None:
        """Render evidence lines, background labels, and guide line.

        Args:
            plotter: The vedo.Plotter instance to draw into.
            step: Current step index for the vertical guide line.
        """
        if not self.added_plot_flag:
            plotter.at(self.renderer_ix).add(
                self.bg_labels, self.bg_rects, self.lines, self.guide_line
            )
            self.added_plot_flag = True

        self.guide_line.points = [(step, 0, 0), (step, 80, 0)]

    def cam_dict(self) -> dict[str, tuple[float, float, float]]:
        """Returns camera parameters for an overhead view of the plot.

        Returns:
            Dictionary with camera position and focal point.
        """
        return {"pos": (125, 25, 300), "focal_point": (125, 25, 0)}


class InteractivePlot:
    """An interactive 3-pane plot for sensor data, MLH hypotheses, and evidence scores.

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
        evidence_plotter: EvidencePlot for plotting evidence scores.
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
        throttle_time: float = 0.2,
    ):
        self.throttle_time = throttle_time

        self.data_extractor = DataExtractor(exp_path, data_path, learning_module)

        self.gt_sim = GroundTruthSimulator(
            data_extractor=self.data_extractor, renderer_ix=1
        )

        self.mlh_sim = MlhSimulator(data_extractor=self.data_extractor, renderer_ix=2)

        self.evidence_plotter = EvidencePlot(
            data_extractor=self.data_extractor, renderer_ix=0
        )

        # Create a plotter with 3 renderers (2 on top, 1 on bottom)
        self.plotter = Plotter(shape="2/1", size=(1000, 1000), sharecam=False)

        # Create a slider on the plot
        self.slider = self.plotter.at(0).add_slider(
            self.slider_callback,
            xmin=0,
            xmax=len(self.data_extractor) - 1,
            value=0,
            pos=[(0.2, 0.05), (0.8, 0.05)],
            title="Step",
        )
        self.curr_slider_val = None
        self.last_call_time = time.time()
        self.slider_callback(self.slider, "")

        # Create buttons for aligning cameras
        self.plotter.at(1).add_button(
            self.simulator_callback,
            pos=(0.1, 0.9),
            states=["Align"],
            size=30,
            font="Calco",
            bold=True,
        )
        self.plotter.at(2).add_button(
            self.mlh_callback,
            pos=(0.9, 0.9),
            states=["Align"],
            size=30,
            font="Calco",
            bold=True,
        )

    def align_camera(self, cam_a: Any, cam_b: Any) -> None:
        """Align the camera objects."""
        cam_a.SetPosition(cam_b.GetPosition())
        cam_a.SetFocalPoint(cam_b.GetFocalPoint())
        cam_a.SetViewUp(cam_b.GetViewUp())
        cam_a.SetClippingRange(cam_b.GetClippingRange())
        cam_a.SetParallelScale(cam_b.GetParallelScale())

    def simulator_callback(self, _widget: Button, _event: str) -> None:
        """Align the ground truth renderer's camera with the MLH renderer."""
        cam_a = self.plotter.renderers[1].GetActiveCamera()
        cam_b = self.plotter.renderers[2].GetActiveCamera()
        self.align_camera(cam_a, cam_b)
        self.plotter.render()

    def mlh_callback(self, _widget: Button, _event: str) -> None:
        """Align the MLH renderer's camera with the groundtruth renderer."""
        cam_a = self.plotter.renderers[1].GetActiveCamera()
        cam_b = self.plotter.renderers[2].GetActiveCamera()
        self.align_camera(cam_b, cam_a)
        self.plotter.render()

    def slider_callback(self, widget: Slider2D, _event: str) -> None:
        """Respond to slider step change by updating all subplots.

        Note: This function is throttled to prevent recursion depth errors
            while continue to be responsive.
        """
        val = round(widget.GetRepresentation().GetValue())
        if (
            val != self.curr_slider_val
            and time.time() - self.last_call_time > self.throttle_time
        ):
            self.gt_sim(plotter=self.plotter, step=val)
            self.mlh_sim(plotter=self.plotter, step=val)
            self.evidence_plotter(plotter=self.plotter, step=val)

            self.curr_slider_val = val
            self.last_call_time = time.time()
            self.render()

    def render(self, resetcam: bool = False) -> None:
        """Render the visualization layout.

        Args:
            resetcam: If True, resets camera for all renderers.
        """
        self.plotter.render()
        self.plotter.at(self.gt_sim.renderer_ix).show(
            axes=self.gt_sim.axes_dict(),
            resetcam=resetcam,
            interactive=False,
        )
        self.plotter.at(self.mlh_sim.renderer_ix).show(
            axes=self.mlh_sim.axes_dict(), resetcam=resetcam, interactive=False
        )
        self.plotter.at(self.evidence_plotter.renderer_ix).show(
            axes=self.evidence_plotter.axes_dict(),
            camera=self.evidence_plotter.cam_dict(),
            resetcam=False,
            interactive=True,
        )


def plot_interactive_objects_evidence_over_time(
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
    plot.render(resetcam=True)

    return 0


def add_subparser(
    subparsers: argparse._SubParsersAction,
    parent_parser: argparse.ArgumentParser | None = None,
) -> None:
    """Add the interactive_objects_evidence_over_time subparser to the main parser.

    Args:
        subparsers: The subparsers object from the main parser.
        parent_parser: Optional parent parser for shared arguments.
    """
    parser = subparsers.add_parser(
        "interactive_objects_evidence_over_time",
        help="Creates an interactive plot of object evidence and sensor visualization.",
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
            plot_interactive_objects_evidence_over_time(
                args.experiment_log_dir, args.objects_mesh_dir, args.learning_module
            )
        )
    )
