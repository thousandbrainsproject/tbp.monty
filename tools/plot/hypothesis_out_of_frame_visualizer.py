# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import os
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from vedo import (
    Points,
    Plotter,
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
    features: Optional[list[str]] = ["rgba"],
    lm_id: int = 0,
) -> ObjectModel:
    monty_models_dir = Path(os.getenv("MONTY_MODELS"))
    pretrain_dir = monty_models_dir / "pretrained_ycb_v10"

    if "dist" in exp_name:
        pretrain_dir = (
            pretrain_dir / "supervised_pre_training_all_objects" / "pretrained"
        )
    elif "surf" in exp_name:
        pretrain_dir = (
            pretrain_dir / "supervised_pre_training_surface_view" / "pretrained"
        )
    else:
        raise ValueError(
            f"The experiment name must contain 'dist' or 'surf' in order to load correct pretrained model for objects."
        )

    model_path = pretrain_dir / "checkpoints" / "model.pt"

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
    """Visualize target object hypotheses from a single timestep."""

    def __init__(
        self, json_path: str, model_name: str = "dist_agent_1lm", timestep: int = 25
    ):
        self.json_path = json_path
        self.model_name = model_name
        self.timestep = timestep  # Default to last timestep (25) in episode 0

        # Load data
        self.load_episode_data()
        self.load_target_model()

    def load_episode_data(self) -> None:
        """Load first episode data from JSON file."""
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

        # Extract target object hypothesis data for the specified timestep
        if self.timestep >= len(self.lm_data["possible_locations"]):
            print(
                f"Warning: timestep {self.timestep} not available, using last timestep"
            )
            self.timestep = len(self.lm_data["possible_locations"]) - 1

        timestep_data = self.lm_data["possible_locations"][self.timestep]
        if self.target_object_name not in timestep_data:
            raise ValueError(
                f"No {self.target_object_name} data found at timestep {self.timestep}"
            )

        self.target_locations = np.array(timestep_data[self.target_object_name])
        self.target_evidences = np.array(
            self.lm_data["evidences"][self.timestep][self.target_object_name]
        )

        print(
            f"Loaded {len(self.target_locations)} {self.target_object_name} hypotheses from timestep {self.timestep}"
        )
        print(
            f"Evidence range: [{self.target_evidences.min():.4f}, {self.target_evidences.max():.4f}]"
        )

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

        # Check for current MLH (Most Likely Hypothesis) data
        self.mlh_location = None
        self.mlh_rotation = None
        self.mlh_graph_id = None
        if "current_mlh" in self.lm_data and self.timestep < len(
            self.lm_data["current_mlh"]
        ):
            mlh_data = self.lm_data["current_mlh"][self.timestep]
            if "location" in mlh_data:
                self.mlh_location = np.array(mlh_data["location"])
                print(f"Current MLH location: {self.mlh_location}")
            if "rotation" in mlh_data:
                self.mlh_rotation = np.array(mlh_data["rotation"])
            if "graph_id" in mlh_data:
                self.mlh_graph_id = mlh_data["graph_id"]
                print(f"Current MLH object: {self.mlh_graph_id}")

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

    def create_visualization(self) -> None:
        """Create the visualization with vedo."""
        plotter = Plotter(
            title=f"Sensor Location Hypotheses for {self.target_object_name.title()} - Episode 0, Timestep {self.timestep}"
        )

        # Add target object model if available
        if self.target_model is not None:
            # Transform model to target pose
            model = self.target_model.copy()
            # model = model.rotated(self.target_rotation, degrees=True)
            # model = model + self.target_position

            # Create point cloud - use a single color for simplicity
            object_points = Points(model.pos, c="cyan")
            object_points.point_size(10)  # Larger size for better visibility
            plotter.add(object_points)

            # Add label
            plotter.add(
                Text2D(
                    f"{self.target_object_name.title()} Object (Ground Truth)",
                    pos="top-left",
                    s=1,
                )
            )

        # Create hypothesis points colored by evidence
        # Normalize evidence to [0, 1] for coloring
        # Use log scale for better visualization due to wide range
        evidence_shifted = (
            self.target_evidences - self.target_evidences.min() + 1.0
        )  # Shift to positive
        evidence_log = np.log(evidence_shifted)
        evidence_norm = (evidence_log - evidence_log.min()) / (
            evidence_log.max() - evidence_log.min() + 1e-8
        )

        print(
            f"Evidence normalization - Min: {evidence_norm.min():.3f}, Max: {evidence_norm.max():.3f}"
        )

        # Create points with color based on evidence
        hypothesis_points = Points(self.target_locations)
        hypothesis_points.point_size(4)  # Smaller size for hypotheses

        # Apply colormap using the standard vedo approach
        hypothesis_points.pointdata["Evidence"] = evidence_norm
        hypothesis_points = hypothesis_points.cmap("coolwarm", "Evidence")
        # hypothesis_points.alpha(0.6)  # Make points semi-transparent

        plotter.add(hypothesis_points)

        # Add a sphere at the object center to make it more visible
        from vedo import Sphere, Line

        object_center = Sphere(self.target_position, r=0.03, c="red")
        plotter.add(object_center)
        plotter.add(Text2D("Object Center", pos=(0.5, 0.95), s=0.8, c="red"))

        # Add current MLH location if available
        if self.mlh_location is not None:
            mlh_sphere = Sphere(self.mlh_location, r=0.01, c="magenta")
            plotter.add(mlh_sphere)
            mlh_label = f"Current MLH"
            if self.mlh_graph_id:
                mlh_label += f" ({self.mlh_graph_id})"
            plotter.add(Text2D(mlh_label, pos=(0.05, 0.05), s=0.8, c="magenta"))

        # Add colorbar
        # Note: Colorbar causing issues with vedo, commenting out for now
        # hypothesis_points.add_scalarbar(title="Evidence\n(log scale)",
        #                                horizontal=False,
        #                                pos=(0.85, 0.5))

        # Add statistics text
        max_evidence_idx = np.argmax(self.target_evidences)
        max_evidence_location = self.target_locations[max_evidence_idx]
        stats_text = (
            f"Target: {self.target_object_name}\n"
            f"Object position: [{self.target_position[0]:.3f}, {self.target_position[1]:.3f}, {self.target_position[2]:.3f}]\n"
            f"Timestep: {self.timestep}\n"
            f"Total sensor location hypotheses: {len(self.target_locations)}\n"
            f"Evidence range: [{self.target_evidences.min():.4f}, "
            f"{self.target_evidences.max():.4f}]\n"
            f"Best sensor hypothesis: [{max_evidence_location[0]:.3f}, {max_evidence_location[1]:.3f}, {max_evidence_location[2]:.3f}]"
        )

        # Add MLH info if available
        if self.mlh_location is not None:
            stats_text += f"\nCurrent MLH location: [{self.mlh_location[0]:.3f}, {self.mlh_location[1]:.3f}, {self.mlh_location[2]:.3f}]"
            if self.mlh_graph_id:
                stats_text += f"\nCurrent MLH object: {self.mlh_graph_id}"
        plotter.add(Text2D(stats_text, pos="bottom-left", s=0.8))

        # Add color legend
        plotter.add(Text2D("Blue = Low Evidence", pos=(0.85, 0.6), s=0.8, c="blue"))
        plotter.add(Text2D("Red = High Evidence", pos=(0.85, 0.55), s=0.8, c="red"))

        # Set camera and axes
        plotter.show(
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


def plot_target_hypotheses(
    exp_path: str,
    model_name: str = "dist_agent_1lm",
    timestep: int = 25,
) -> int:
    """Plot target object hypotheses for a single timestep.

    Args:
        exp_path: Path to experiment directory containing detailed_run_stats.json
        model_name: Name of pretrained model to load object from
        timestep: Which timestep to visualize (0-25 for episode 0)

    Returns:
        Exit code
    """
    json_path = Path(exp_path) / "detailed_run_stats.json"

    if not json_path.exists():
        logger.error(f"Could not find detailed_run_stats.json at {json_path}")
        return 1

    try:
        visualizer = TargetHypothesisVisualizer(str(json_path), model_name, timestep)
        visualizer.create_visualization()
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
        help="Visualize target object hypothesis locations for a single timestep.",
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
    parser.add_argument(
        "--timestep",
        type=int,
        default=25,
        help="Which timestep to visualize (0-25 for episode 0).",
    )
    parser.set_defaults(
        func=lambda args: sys.exit(
            plot_target_hypotheses(
                args.experiment_log_dir,
                args.model_name,
                args.timestep,
            )
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize target object hypothesis locations for a single timestep."
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
    parser.add_argument(
        "--timestep",
        type=int,
        default=25,
        help="Which timestep to visualize (0-25 for episode 0).",
    )

    args = parser.parse_args()
    sys.exit(
        plot_target_hypotheses(
            args.experiment_log_dir,
            args.model_name,
            args.timestep,
        )
    )
