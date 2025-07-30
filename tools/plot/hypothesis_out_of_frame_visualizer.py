# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from vedo import (
    Points,
    Plotter,
    Sphere,
    Text2D,
    settings,
)

from tbp.monty.frameworks.run_env import setup_env

if TYPE_CHECKING:
    import argparse

logger = logging.getLogger(__name__)

# Vedo settings
settings.immediate_rendering = False
settings.default_font = "Theemim"
settings.default_backend = "vtk"  # Use vtk backend for standalone scripts

setup_env()


class ObjectModel:
    """Mutable wrapper for object models.

    Args:
        pos (ArrayLike): The points of the object model as a sequence of points
          (i.e., has shape (n_points, 3)).
        features (Optional[Mapping]): The features of the object model. For
          convenience, the features become attributes of the ObjectModel instance.
    """

    def __init__(
        self,
        pos: np.ndarray,
        features: Optional[dict[str, np.ndarray]] = None,
    ):
        self.pos = np.asarray(pos, dtype=float)
        if features:
            for key, value in features.items():
                setattr(self, key, np.asarray(value))

    @property
    def x(self) -> np.ndarray:
        return self.pos[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.pos[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self.pos[:, 2]

    def copy(self, deep: bool = True) -> ObjectModel:
        from copy import deepcopy

        return deepcopy(self) if deep else self

    def rotated(
        self,
        rotation: R | np.ndarray,
        degrees: bool = False,
    ) -> ObjectModel:
        """Rotate the object model.

        Args:
            rotation: The rotation to apply to the object model.
            degrees: Whether the rotation is in degrees or radians.

        Returns:
            The rotated object model.

        Raises:
            ValueError: If the rotation argument is invalid.
        """
        if isinstance(rotation, R):
            rot = rotation
        else:
            arr = np.asarray(rotation)
            if arr.shape == (3,):
                rot = R.from_euler("xyz", arr, degrees=degrees)
            elif arr.shape == (3, 3):
                rot = R.from_matrix(arr)
            else:
                raise ValueError(f"Invalid rotation argument: {rotation}")

        pos = rot.apply(self.pos)
        out = self.copy()
        out.pos = pos
        return out

    def __add__(self, translation: np.ndarray) -> "ObjectModel":
        translation = np.asarray(translation)
        out = self.copy(deep=True)
        out.pos += translation
        return out

    def __sub__(self, translation: np.ndarray) -> "ObjectModel":
        translation = np.asarray(translation)
        return self + (-translation)


def load_object_model(
    exp_name: str,
    object_name: str,
    features: Optional[list[str]] = None,
    lm_id: int = 0,
) -> ObjectModel:
    """Load an object model from a pretraining experiment.

    Args:
        exp_name: The name of the experiment.
        object_name: The name of the object.
        features: The features to load from the object model.
        lm_id: The ID of the LM to load the object model from.

    Returns:
        The object model.

    Raises:
        ValueError: If the experiment name does not contain 'dist' or 'surf'.
    """
    if features is None:
        features = ["rgba"]

    monty_models_dir = Path(os.getenv("MONTY_MODELS"))
    pretrain_dir = monty_models_dir / "pretrained_ycb_v10"

    if "dist" in exp_name:
        pretrain_dir = pretrain_dir / "supervised_pre_training_base" / "pretrained"
    elif "surf" in exp_name:
        pretrain_dir = pretrain_dir / "surf_agent_1lm_77obj" / "pretrained"
    else:
        raise ValueError(
            f"The experiment name must contain 'dist' or 'surf' in order to load correct pretrained model for objects."
        )

    model_path = pretrain_dir / "model.pt"

    data = torch.load(model_path, map_location=torch.device("cpu"))
    data = data["lm_dict"][lm_id]["graph_memory"][object_name]["patch"]
    points = np.array(data.pos, dtype=float)

    feature_dict = {}
    if features:
        for feature in features:
            if feature not in data.feature_mapping:
                print(f"WARNING: Feature {feature} not found in data.feature_mapping")
                continue
            idx = data.feature_mapping[feature]
            feature_data = np.array(data.x[:, idx[0] : idx[1]])
            if feature == "rgba":
                feature_data = feature_data / 255.0
            feature_dict[feature] = feature_data

    return ObjectModel(points, features=feature_dict)


class TargetHypothesisVisualizer:
    """Interactive visualization of target object hypotheses across timesteps."""

    def __init__(self, json_path: str, model_name: str = "dist_agent_1lm"):
        self.json_path = json_path
        self.model_name = model_name
        self.current_timestep = 0

        # Will store data for all timesteps
        self.all_target_locations = []
        self.all_target_evidences = []
        self.all_mlh_locations = []
        self.all_mlh_rotations = []
        self.all_mlh_graph_ids = []

        # Visualization objects
        self.hypothesis_points = None
        self.object_points = None
        self.object_center = None
        self.mlh_sphere = None
        self.stats_text = None
        self.slider = None
        self.plotter = None

        # Load data
        self.load_episode_data()
        self.load_target_model()

    def load_episode_data(self) -> None:
        """Load all timesteps data from JSON file."""
        print(f"Loading episode data from: {self.json_path}")

        with open(self.json_path, "r") as f:
            first_line = f.readline().strip()
            data = json.loads(first_line)

        # Navigate to LM_0 data
        if "0" in data:
            self.lm_data = data["0"]["LM_0"]
            self.target_data = data["0"]["target"]
        else:
            raise ValueError("Could not find episode 0 data")

        # Get the actual target object name
        self.target_object_name = self.target_data.get(
            "primary_target_object", self.target_data.get("object", "")
        )
        if not self.target_object_name:
            raise ValueError("Could not determine target object name from episode data")

        print(f"Target object for episode 0: {self.target_object_name}")

        # Get number of timesteps
        self.num_timesteps = len(self.lm_data["possible_locations"])
        print(f"Episode has {self.num_timesteps} timesteps")

        # Extract data for all timesteps
        for timestep in range(self.num_timesteps):
            timestep_data = self.lm_data["possible_locations"][timestep]
            if self.target_object_name in timestep_data:
                self.all_target_locations.append(
                    np.array(timestep_data[self.target_object_name])
                )
                self.all_target_evidences.append(
                    np.array(
                        self.lm_data["evidences"][timestep][self.target_object_name]
                    )
                )
            else:
                # Empty data for this timestep
                self.all_target_locations.append(np.array([]))
                self.all_target_evidences.append(np.array([]))

            # Extract MLH data if available
            if "current_mlh" in self.lm_data and timestep < len(
                self.lm_data["current_mlh"]
            ):
                mlh_data = self.lm_data["current_mlh"][timestep]
                if "location" in mlh_data:
                    self.all_mlh_locations.append(np.array(mlh_data["location"]))
                else:
                    self.all_mlh_locations.append(None)
                if "rotation" in mlh_data:
                    self.all_mlh_rotations.append(np.array(mlh_data["rotation"]))
                else:
                    self.all_mlh_rotations.append(None)
                if "graph_id" in mlh_data:
                    self.all_mlh_graph_ids.append(mlh_data["graph_id"])
                else:
                    self.all_mlh_graph_ids.append(None)
            else:
                self.all_mlh_locations.append(None)
                self.all_mlh_rotations.append(None)
                self.all_mlh_graph_ids.append(None)

        # Get target object pose
        self.target_position = np.array(
            self.target_data.get(
                "primary_target_position", self.target_data.get("position", [0, 0, 0])
            )
        )
        self.target_rotation = np.array(
            self.target_data.get(
                "primary_target_rotation_euler",
                self.target_data.get("euler_rotation", [0, 0, 0]),
            )
        )
        print(f"{self.target_object_name} is at position: {self.target_position}")

    def load_target_model(self) -> None:
        """Load the target object model."""
        try:
            self.target_model = load_object_model(
                self.model_name, self.target_object_name
            )
            print(
                f"Loaded {self.target_object_name} model with {len(self.target_model.pos)} points"
            )
        except Exception as e:
            print(f"Warning: Could not load {self.target_object_name} model: {e}")
            self.target_model = None

    def create_interactive_visualization(self) -> None:
        """Create interactive visualization with slider for timestep navigation."""
        self.plotter = Plotter(
            title=f"Sensor Location Hypotheses for {self.target_object_name.title()} - Episode 0"
        )

        # Add slider
        self.slider = self.plotter.add_slider(
            self.slider_callback,
            xmin=0,
            xmax=self.num_timesteps - 1,
            value=0,
            pos=[(0.2, 0.05), (0.8, 0.05)],
            title="Timestep",
        )

        # Initial visualization
        self.update_visualization(0)

        # Show
        self.plotter.show(
            axes={
                "xtitle": "X",
                "ytitle": "Y",
                "ztitle": "Z",
                "xrange": (-0.2, 0.2),
                "yrange": (1.3, 1.7),
                "zrange": (-0.2, 0.2),
            },
            viewup="z",
            interactive=True,
        )

    def slider_callback(self, widget, event) -> None:
        """Handle slider value changes."""
        timestep = int(round(widget.GetRepresentation().GetValue()))
        if timestep != self.current_timestep:
            self.update_visualization(timestep)
            self.plotter.render()

    def update_visualization(self, timestep: int) -> None:
        """Update visualization for given timestep."""
        self.current_timestep = timestep

        # Get data for current timestep
        if (
            timestep >= len(self.all_target_locations)
            or len(self.all_target_locations[timestep]) == 0
        ):
            print(f"No data available for timestep {timestep}")
            return

        target_locations = self.all_target_locations[timestep]
        target_evidences = self.all_target_evidences[timestep]
        mlh_location = self.all_mlh_locations[timestep]
        mlh_graph_id = self.all_mlh_graph_ids[timestep]

        # Remove previous objects if they exist
        if self.hypothesis_points is not None:
            self.plotter.remove(self.hypothesis_points)
        if self.mlh_sphere is not None:
            self.plotter.remove(self.mlh_sphere)
        if self.stats_text is not None:
            self.plotter.remove(self.stats_text)

        # Add target object model on first call
        if self.object_points is None and self.target_model is not None:
            # Transform model to target pose
            model = self.target_model.copy()
            # model = model.rotated(self.target_rotation, degrees=True)
            # model = model + self.target_position

            # Create point cloud
            self.object_points = Points(model.pos, c="gray")
            self.object_points.point_size(10)
            self.plotter.add(self.object_points)

            # Add label
            self.plotter.add(
                Text2D(
                    f"{self.target_object_name.title()} Object (Ground Truth)",
                    pos="top-left",
                    s=1,
                )
            )

        # Create hypothesis points colored by evidence
        if len(target_evidences) > 0:
            # Normalize evidence to [0, 1] for coloring
            # Use log scale for better visualization
            evidence_shifted = (
                target_evidences - target_evidences.min() + 1.0
            )  # Shift to positive
            evidence_log = np.log(evidence_shifted)
            evidence_norm = (evidence_log - evidence_log.min()) / (
                evidence_log.max() - evidence_log.min() + 1e-8
            )

            # Create points with color based on evidence
            self.hypothesis_points = Points(target_locations)
            self.hypothesis_points.point_size(8)  # Increased from 4 to 8

            # Apply colormap - using 'jet' for better contrast
            self.hypothesis_points.pointdata["Evidence"] = evidence_norm
            self.hypothesis_points = self.hypothesis_points.cmap("jet", "Evidence")
            self.plotter.add(self.hypothesis_points)

        # Add current MLH location if available
        if mlh_location is not None:
            self.mlh_sphere = Sphere(mlh_location, r=0.005, c="red")
            self.plotter.add(self.mlh_sphere)

        # Update statistics text
        if len(target_evidences) > 0:
            max_evidence_idx = np.argmax(target_evidences)
            max_evidence_location = target_locations[max_evidence_idx]
            stats_text = (
                f"Target: {self.target_object_name}\n"
                f"Object position: [{self.target_position[0]:.3f}, {self.target_position[1]:.3f}, {self.target_position[2]:.3f}]\n"
                f"Timestep: {timestep}\n"
                f"Total sensor location hypotheses: {len(target_locations)}\n"
                f"Evidence range: [{target_evidences.min():.4f}, "
                f"{target_evidences.max():.4f}]\n"
                f"Best sensor hypothesis: [{max_evidence_location[0]:.3f}, {max_evidence_location[1]:.3f}, {max_evidence_location[2]:.3f}]"
            )

            # Add MLH info if available
            if mlh_location is not None:
                stats_text += f"\nCurrent MLH location: [{mlh_location[0]:.3f}, {mlh_location[1]:.3f}, {mlh_location[2]:.3f}]"
                if mlh_graph_id:
                    stats_text += f"\nCurrent MLH object: {mlh_graph_id}"

            self.stats_text = Text2D(stats_text, pos="top-right", s=0.7)
            self.plotter.add(self.stats_text)


def plot_target_hypotheses(
    exp_path: str,
    model_name: str = "dist_agent_1lm",
) -> int:
    """Plot target object hypotheses with interactive timestep slider.

    Args:
        exp_path: Path to experiment directory containing detailed_run_stats.json
        model_name: Name of pretrained model to load object from

    Returns:
        Exit code
    """
    json_path = Path(exp_path) / "detailed_run_stats.json"

    if not json_path.exists():
        logger.error(f"Could not find detailed_run_stats.json at {json_path}")
        return 1

    try:
        visualizer = TargetHypothesisVisualizer(str(json_path), model_name)
        visualizer.create_interactive_visualization()
        return 0
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        return 1


def add_subparser(
    subparsers: argparse._SubParsersAction,
    parent_parser: argparse.ArgumentParser | None = None,
) -> None:
    """Add the hypothesis_out_of_frame subparser to the main parser."""
    parser = subparsers.add_parser(
        "hypothesis_out_of_frame",
        help="Interactive visualization of target object hypothesis locations.",
        parents=[parent_parser] if parent_parser else [],
    )
    parser.add_argument(
        "experiment_log_dir",
        help="The directory containing the detailed_run_stats.json file.",
    )
    parser.add_argument(
        "--model_name",
        default="dist_agent_1lm",
        help="Name of pretrained model to load target object from.",
    )
    parser.set_defaults(
        func=lambda args: sys.exit(
            plot_target_hypotheses(
                args.experiment_log_dir,
                args.model_name,
            )
        )
    )
