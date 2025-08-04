# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Interactive visualization tool for analyzing hypotheses using radius-based evidence.

This visualizer shows ellipsoidal acceptance regions that represent the anisotropic
distance metric used by get_custom_distances, which takes into account surface
normals and curvature.

This visualizer requires that experiments have been run with detailed logging
enabled to generate detailed_run_stats.json files. To enable detailed logging,
use DetailedEvidenceLMLoggingConfig in your experiment configuration.

Usage:
    python tools/plot/cli.py hypothesis_out_of_frame_radius <experiment_log_dir>
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
from vedo import (
    Plotter,
    Points,
    Sphere,
    Ellipsoid,
    Text2D,
    settings,
)

from tbp.monty.frameworks.run_env import setup_env
from tbp.monty.frameworks.utils.graph_matching_utils import (
    get_custom_distances,
    get_relevant_curvature,
)

if TYPE_CHECKING:
    import argparse

    from vedo import Button, Slider2D

logger = logging.getLogger(__name__)

# Vedo settings
settings.immediate_rendering = False
settings.default_font = "Theemim"
settings.window_splitting_position = 0.5

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
        self.kd_tree = KDTree(self.pos, leafsize=40)

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

    def __add__(self, translation: np.ndarray) -> ObjectModel:
        translation = np.asarray(translation)
        out = self.copy(deep=True)
        out.pos += translation
        return out

    def __sub__(self, translation: np.ndarray) -> ObjectModel:
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
    monty_models_dir = Path(os.getenv("MONTY_MODELS"))
    pretrain_dir = monty_models_dir / "pretrained_ycb_v10"

    if "dist" in exp_name:
        pretrain_dir = pretrain_dir / "supervised_pre_training_base" / "pretrained"
    elif "surf" in exp_name:
        pretrain_dir = pretrain_dir / "surf_agent_1lm_77obj" / "pretrained"
    else:
        raise ValueError(
            "The experiment name must contain 'dist' or 'surf' in order to load "
            "correct pretrained model for objects."
        )

    model_path = pretrain_dir / "model.pt"
    data = torch.load(model_path, map_location=torch.device("cpu"))
    data = data["lm_dict"][lm_id]["graph_memory"][object_name][
        "patch"
    ]  # GraphObjectModel
    points = np.array(data.pos, dtype=float)

    feature_dict = {}
    # data.feature_mapping = {'node_ids': [0, 1], 'pose_vectors': [1, 10], 'pose_fully_defined': [10, 11],
    # 'on_object': [11, 12], 'object_coverage': [12, 13], 'rgba': [13, 17], 'hsv': [17, 20],
    # 'principal_curvatures': [20, 22], 'principal_curvatures_log': [22, 24], 'gaussian_curvature': [24, 25],
    # 'mean_curvature': [25, 26], 'gaussian_curvature_sc': [26, 27], 'mean_curvature_sc': [27, 28]}
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


def compute_distances_to_object(
    hypothesis_locations: np.ndarray,
    target_model: ObjectModel,
    hypothesis_rotations: np.ndarray,
    max_nneighbors: int = 3,
) -> np.ndarray:
    """Compute distance from each hypothesis to target object.

    Args:
        hypothesis_locations: Array of hypothesis locations (n_hypotheses, 3)
        target_model: ObjectModel containing points and features
        hypothesis_rotations: Rotation matrices for each hypothesis (n_hypotheses, 3, 3)
        max_nneighbors: Maximum number of nearest neighbors to consider (default: 3)

    Returns:
        Array of minimum distances for each hypothesis (n_hypotheses,)
    """
    _, nearest_node_ids = target_model.kd_tree.query(
        hypothesis_locations,
        k=max_nneighbors,
        p=2,
        workers=1,
    )

    if max_nneighbors == 1:
        nearest_node_ids = np.expand_dims(nearest_node_ids, axis=1)

    nearest_node_locs = target_model.pos[nearest_node_ids]
    surface_normals = hypothesis_rotations[:, :, 2]
    object_features = target_model.__dict__
    max_abs_curvature = get_relevant_curvature(object_features)

    custom_nearest_node_dists = get_custom_distances(
        nearest_node_locs,
        hypothesis_locations,
        surface_normals,
        max_abs_curvature,
    )

    # Take minimum distance across neighbors for each hypothesis
    min_distances = np.min(custom_nearest_node_dists, axis=1)

    return min_distances  # shape: (n_hypotheses,)


class HypothesesRadiusVisualizer:
    """Interactive visualizer for analyzing sensor location hypotheses using radius-based evidence.

    Args:
        json_path: Path to the detailed_run_stats.json file containing episode data.
        model_name: Name of pretrained model to load object from. Defaults to "dist_agent_1lm".
        max_match_distance: Maximum distance for matching (default: 0.05).

    Attributes:
        hypothesis_points: Current vedo.Points representing sensor location hypotheses.
        object_points: Current vedo.Points representing the target object point cloud.
        hypothesis_ellipsoids: List of ellipsoids showing anisotropic reach of hypotheses.
        mlh_sphere: Current vedo.Sphere representing the most likely hypothesis location.
        stats_text: Text label showing statistics for current timestep.
        target_object_name: The current object name being visualized.
    """

    def __init__(
        self,
        json_path: str,
        model_name: str = "dist_agent_1lm",
        max_match_distance: float = 0.05,
        max_nneighbors: int = 3,
    ):
        self.json_path = json_path
        self.model_name = model_name
        self.current_timestep = 0
        self.max_match_distance = max_match_distance
        self.max_nneighbors = max_nneighbors

        # Data for all timesteps
        self.all_hypotheses_locations = []
        self.all_hypotheses_evidences = []
        self.all_hypotheses_rotations = []
        self.all_mlh_locations = []
        self.all_mlh_rotations = []
        self.all_mlh_graph_ids = []

        self.hypotheses = None
        self.object_points = None
        self.object_center = None
        self.hypothesis_ellipsoids = []
        self.ellipsoids_visible = True
        self.mlh_sphere = None
        self.stats_text = None
        self.slider = None
        self.plotter = None
        self.ellipsoid_button = None
        self.hypotheses_filter_slider = None

        self.current_target_locations = None
        self.current_target_evidences = None
        self.current_hypotheses_rotations = None

        # Filtering settings
        self.max_hypotheses_to_show = None  # None means show all
        self.total_hypotheses_count = 0

        self.load_episode_data()
        self.load_target_model()

    def load_episode_data(self) -> None:
        """Load all timesteps data from JSON file.

        Raises:
            ValueError: If the episode 0 data is not found.
        """
        logger.info(f"Loading episode data from: {self.json_path}")

        with open(self.json_path, "r") as f:
            first_line = f.readline().strip()
            data = json.loads(first_line)

        if "0" in data:
            self.lm_data = data["0"]["LM_0"]
            self.target_data = data["0"]["target"]
        else:
            raise ValueError("Could not find episode 0 data")

        self.target_object_name = self.target_data.get(
            "primary_target_object", self.target_data.get("object", "")
        )

        logger.info(f"Target object for episode 0: {self.target_object_name}")

        self.num_timesteps = len(self.lm_data["possible_locations"])
        logger.info(f"Episode has {self.num_timesteps} timesteps")

        for timestep in range(self.num_timesteps):
            timestep_data = self.lm_data["possible_locations"][timestep]
            self.all_hypotheses_locations.append(
                np.array(timestep_data[self.target_object_name])
            )
            self.all_hypotheses_evidences.append(
                np.array(self.lm_data["evidences"][timestep][self.target_object_name])
            )
            # Rotation data is the same for all timesteps
            rotation_data = self.lm_data["possible_rotations"][
                0
            ]  # shape: (n_rotations, 3, 3)
            self.all_hypotheses_rotations.append(
                np.array(rotation_data[self.target_object_name])
            )

            mlh_data = self.lm_data["current_mlh"][timestep]
            self.all_mlh_locations.append(np.array(mlh_data["location"]))
            self.all_mlh_rotations.append(np.array(mlh_data["rotation"]))
            self.all_mlh_graph_ids.append(mlh_data["graph_id"])

        self.target_position = np.array(self.target_data.get("primary_target_position"))
        self.target_rotation = np.array(
            self.target_data.get("primary_target_rotation_euler")
        )
        logger.info(
            f"{self.target_object_name} is at position: {self.target_position} "
            f"and rotation: {self.target_rotation}"
        )

    def load_target_model(self) -> None:
        """Load the target object model."""
        features_to_load = [
            "rgba",
            "pose_vectors",
            "pose_fully_defined",
            "principal_curvatures",
            "principal_curvatures_log",
        ]
        self.target_model = load_object_model(
            self.model_name, self.target_object_name, features=features_to_load
        )
        logger.info(
            f"Loaded {self.target_object_name} model with "
            f"{len(self.target_model.pos)} points and features: "
            f"{list(self.target_model.__dict__.keys())}"
        )

    def create_interactive_visualization(self) -> None:
        """Create interactive visualization with slider for timestep navigation."""
        self.plotter = Plotter(
            title=(
                f"Sensor Location Hypotheses (Radius-based) for {self.target_object_name.title()} "
                "- Episode 0"
            )
        )
        self.update_visualization(timestep=0)

        self.slider = self.plotter.add_slider(
            self.slider_callback,
            xmin=0,
            xmax=self.num_timesteps - 1,
            value=0,
            pos=[(0.2, 0.05), (0.8, 0.05)],
            title="Timestep",
        )
        self.ellipsoid_button = self.plotter.add_button(
            self.toggle_ellipsoid_callback,
            pos=(0.9, 0.1),
            states=["Hide Ellipsoids", "Show Ellipsoids"],
            font="Calco",
            bold=True,
        )

        self.hypotheses_filter_slider = self.plotter.add_slider(
            self.hypotheses_filter_slider_callback,
            xmin=0,
            xmax=100,
            value=100,
            pos=[(0.2, 0.12), (0.8, 0.12)],
            title="Top N% Hypotheses",
            show_value=True,
        )

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

    def update_visualization(self, timestep: int) -> None:
        """Update visualization for given timestep."""
        self.current_timestep = timestep

        (
            hypotheses_locations,
            hypotheses_evidences,
            hypotheses_rotations,
            mlh_location,
            mlh_graph_id,
        ) = self._get_timestep_data(timestep)

        self._cleanup_previous_visualizations()

        if self.object_points is None and self.target_model is not None:
            self._initialize_object_visualization()

        self._add_hypotheses(
            hypotheses_locations, hypotheses_evidences, hypotheses_rotations
        )
        self._add_mlh_sphere(mlh_location)

        stats_text = self._create_statistics_text(
            timestep,
            hypotheses_locations,
            hypotheses_evidences,
            mlh_location,
            mlh_graph_id,
        )
        self.stats_text = Text2D(stats_text, pos="top-right", s=0.7)
        self.plotter.add(self.stats_text)

    def slider_callback(self, widget: Slider2D, _event: str) -> None:
        """Respond to slider step by updating the visualization."""
        timestep = round(widget.GetRepresentation().GetValue())
        if timestep != self.current_timestep:
            self.update_visualization(timestep)
            self.plotter.render()

    def toggle_ellipsoid_callback(self, _widget: Button, _event: str) -> None:
        """Toggle the visibility of the hypothesis ellipsoids."""
        self.ellipsoids_visible = not self.ellipsoids_visible

        for ellipsoid in self.hypothesis_ellipsoids:
            if self.ellipsoids_visible:
                ellipsoid.on()
            else:
                ellipsoid.off()

        if self.ellipsoid_button is not None:
            self.ellipsoid_button.switch()

        self.plotter.render()

    def hypotheses_filter_slider_callback(self, widget: Slider2D, _event: str) -> None:
        """Handle filter slider changes."""
        percentage = widget.GetRepresentation().GetValue()
        self.max_hypotheses_to_show = int(
            len(self.current_hypotheses_locations) * percentage / 100
        )

        self.update_visualization(self.current_timestep)
        self.plotter.render()

    def _get_timestep_data(self, timestep: int) -> tuple:
        """Get data for a specific timestep.

        Args:
            timestep: The timestep to retrieve data for

        Returns:
            Tuple of (hypotheses_locations, hypotheses_evidences, hypotheses_rotations,
            mlh_location, mlh_graph_id)
        """
        hypotheses_locations = self.all_hypotheses_locations[timestep]
        hypotheses_evidences = self.all_hypotheses_evidences[timestep]
        hypotheses_rotations = self.all_hypotheses_rotations[timestep]
        mlh_location = self.all_mlh_locations[timestep]
        mlh_graph_id = self.all_mlh_graph_ids[timestep]

        self.current_hypotheses_locations = hypotheses_locations
        self.current_hypotheses_evidences = hypotheses_evidences
        self.current_hypotheses_rotations = hypotheses_rotations

        return (
            hypotheses_locations,
            hypotheses_evidences,
            hypotheses_rotations,
            mlh_location,
            mlh_graph_id,
        )

    def _cleanup_previous_visualizations(self) -> None:
        """Remove previous visualization objects from the plotter."""
        if self.hypotheses is not None:
            if isinstance(self.hypotheses, list):
                for pts in self.hypotheses:
                    self.plotter.remove(pts)
            else:
                self.plotter.remove(self.hypotheses)
        if self.mlh_sphere is not None:
            self.plotter.remove(self.mlh_sphere)
        if self.stats_text is not None:
            self.plotter.remove(self.stats_text)
        # Clean up hypothesis ellipsoids
        for ellipsoid in self.hypothesis_ellipsoids:
            self.plotter.remove(ellipsoid)
        self.hypothesis_ellipsoids = []

    def _initialize_object_visualization(self) -> None:
        """Initialize object model and convex hull visualization."""
        model = self.target_model.copy()
        self.object_points = Points(model.pos, c="gray")
        self.object_points.point_size(10)
        self.plotter.add(self.object_points)

        self.plotter.add(
            Text2D(
                f"{self.target_object_name.title()} Object (Ground Truth)",
                pos="top-left",
                s=1,
            )
        )

    def _count_points_within_ellipsoids(
        self, points: np.ndarray, rotations: np.ndarray
    ) -> tuple[int, int]:
        """Count how many hypotheses have object points within their ellipsoids.

        Args:
            points: Array of hypothesis locations
            rotations: Array of hypothesis rotation matrices

        Returns:
            Tuple of (num_with_object_points, num_without_object_points)
        """
        has_object_points = self._check_object_points_in_ellipsoids(points, rotations)
        num_within = np.sum(has_object_points)
        num_outside = len(points) - num_within

        return num_within, num_outside

    def _add_hypotheses(
        self,
        hypotheses_locations: np.ndarray,
        hypotheses_evidences: np.ndarray,
        hypotheses_rotations: np.ndarray,
    ) -> None:
        """Create hypothesis points colored by whether they're within radius.

        Args:
            hypotheses_locations: Array of hypothesis locations
            hypotheses_evidences: Array of evidence values for each hypothesis
            hypotheses_rotations: Array of rotation matrices for each hypothesis
        """
        self.total_hypotheses_count = len(hypotheses_locations)

        if (
            self.max_hypotheses_to_show is not None
            and len(hypotheses_locations) > self.max_hypotheses_to_show
        ):
            # Sort by evidence values (descending) and take top N
            sorted_indices = np.argsort(hypotheses_evidences)[::-1]
            top_indices = sorted_indices[: self.max_hypotheses_to_show]

            filtered_locations = hypotheses_locations[top_indices]
            filtered_rotations = hypotheses_rotations[top_indices]
        else:
            filtered_locations = hypotheses_locations
            filtered_rotations = hypotheses_rotations

        # For each hypothesis, check if there are object points within its ellipsoid
        # This determines the color (green if object points found, red if not)
        hypothesis_has_object_points = self._check_object_points_in_ellipsoids(
            filtered_locations,
            filtered_rotations,
        )

        within_points = filtered_locations[hypothesis_has_object_points]
        outside_points = filtered_locations[~hypothesis_has_object_points]

        point_clouds = []

        if len(within_points) > 0:
            within_pts = Points(within_points, c="green")
            within_pts.point_size(8)
            point_clouds.append(within_pts)
            self.plotter.add(within_pts)

        if len(outside_points) > 0:
            outside_pts = Points(outside_points, c="red")
            outside_pts.point_size(4)
            point_clouds.append(outside_pts)
            self.plotter.add(outside_pts)

        self.hypotheses = point_clouds

        # Add ellipsoids for filtered hypotheses
        # Show ellipsoids for a subset to avoid clutter
        self._add_hypothesis_ellipsoids(filtered_locations, filtered_rotations)

    def _add_mlh_sphere(self, mlh_location: np.ndarray) -> None:
        """Add MLH sphere visualization.

        Args:
            mlh_location: 3D location of the most likely hypothesis
        """
        # Compute distance to object
        # For MLH, we use the MLH rotation if available
        mlh_rotation_data = self.all_mlh_rotations[self.current_timestep]
        if mlh_rotation_data is not None:
            # MLH rotation is stored as Euler angles, convert to rotation matrix
            mlh_rotation_euler = np.array(mlh_rotation_data)
            mlh_rotation_matrix = R.from_euler(
                "xyz", mlh_rotation_euler, degrees=True
            ).as_matrix()
            mlh_rotation = mlh_rotation_matrix.reshape(1, 3, 3)
        else:
            # Fallback: create identity rotation
            mlh_rotation = np.eye(3).reshape(1, 3, 3)

        distance = compute_distances_to_object(
            mlh_location.reshape(1, -1),
            self.target_model,
            mlh_rotation,
            max_nneighbors=self.max_nneighbors,
        )[0]
        mlh_color = "green" if distance <= self.max_match_distance else "red"

        self.mlh_sphere = Sphere(mlh_location, r=0.005, c=mlh_color)
        self.plotter.add(self.mlh_sphere)

    def _add_hypothesis_ellipsoids(
        self,
        locations: np.ndarray,
        rotations: np.ndarray,
    ) -> None:
        """Add ellipsoids around hypotheses showing their anisotropic reach.

        The ellipsoids are stretched along each hypothesis's surface normal to
        represent the anisotropic distance metric used by get_custom_distances.

        Args:
            locations: Array of hypothesis locations (n_hypotheses, 3)
            rotations: Array of rotation matrices for each hypothesis (n_hypotheses, 3, 3)
            max_to_show: Maximum number of ellipsoids to show
        """
        # Don't show ellipsoids if visibility is off
        if not self.ellipsoids_visible:
            return

        # Check which hypotheses have object points within their ellipsoids
        has_object_points = self._check_object_points_in_ellipsoids(
            locations, rotations
        )

        # Limit the number of ellipsoids shown
        max_to_show = 30
        if len(locations) > max_to_show:
            # Sample a subset, preferring those with object points
            with_indices = np.where(has_object_points)[0]
            without_indices = np.where(~has_object_points)[0]
            
            if len(with_indices) >= max_to_show:
                sampled_indices = np.random.choice(with_indices, max_to_show, replace=False)
            elif len(with_indices) > 0:
                # Include all with object points and sample from those without
                remaining = max_to_show - len(with_indices)
                sampled_without = np.random.choice(
                    without_indices,
                    min(remaining, len(without_indices)),
                    replace=False
                )
                sampled_indices = np.concatenate([with_indices, sampled_without])
            else:
                sampled_indices = np.random.choice(len(locations), max_to_show, replace=False)
            
            show_locations = locations[sampled_indices]
            show_rotations = rotations[sampled_indices]
            show_has_objects = has_object_points[sampled_indices]
        else:
            show_locations = locations
            show_rotations = rotations
            show_has_objects = has_object_points

        # Get the object's maximum curvature for stretch factor calculation
        object_features = self.target_model.__dict__
        max_abs_curvature = get_relevant_curvature(object_features)

        for location, rotation, has_objects in zip(
            show_locations, show_rotations, show_has_objects
        ):
            # Extract surface normal from rotation matrix (3rd column)
            surface_normal = rotation[:, 2]

            # Calculate ellipsoid axes based on the distance metric
            # The distance metric adds |dot_product| * (1 / (|curvature| + 0.5)) to euclidean distance
            stretch_factor = 1 / (np.abs(max_abs_curvature) + 0.5)

            # For a max_match_distance threshold:
            # - In tangent plane: can reach up to max_match_distance
            # - Along normal: solving d + d*stretch_factor = max_match_distance gives
            #   d = max_match_distance / (1 + stretch_factor)
            semi_axis_tangent = self.max_match_distance
            semi_axis_normal = self.max_match_distance / (1 + stretch_factor)

            # Color based on whether hypothesis has object points in its ellipsoid
            color = "lightgreen" if has_objects else "lightcoral"

            # Create ellipsoid with default orientation
            ellipsoid = Ellipsoid(
                pos=location,
                axis1=[semi_axis_tangent, 0, 0],
                axis2=[0, semi_axis_tangent, 0],
                axis3=[0, 0, semi_axis_normal],
                c=color,
            )

            # Rotate ellipsoid to align with surface normal
            # Create rotation that aligns [0,0,1] with the surface normal
            z_axis = np.array([0, 0, 1])
            if not np.allclose(surface_normal, z_axis):
                rotation_axis = np.cross(z_axis, surface_normal)
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                rotation_angle = np.arccos(
                    np.clip(np.dot(z_axis, surface_normal), -1, 1)
                )
                rot = R.from_rotvec(rotation_angle * rotation_axis)
                ellipsoid.apply_transform(rot.as_matrix())

            ellipsoid.alpha(0.15)
            ellipsoid.wireframe(True)
            self.hypothesis_ellipsoids.append(ellipsoid)
            self.plotter.add(ellipsoid)

    def _check_object_points_in_ellipsoids(
        self,
        hypothesis_locations: np.ndarray,
        hypothesis_rotations: np.ndarray,
    ) -> np.ndarray:
        """Check which hypotheses have object points within their ellipsoids.

        This follows the logic from hypotheses_displacer.py: a hypothesis gets
        positive evidence if at least one object point is within max_match_distance
        using the custom distance metric.

        Args:
            hypothesis_locations: Array of hypothesis locations (n_hypotheses, 3)
            hypothesis_rotations: Array of rotation matrices (n_hypotheses, 3, 3)

        Returns:
            Boolean array indicating which hypotheses have object points within
            their ellipsoids (n_hypotheses,)
        """
        # For each hypothesis, find nearest neighbors
        _, nearest_node_ids = self.target_model.kd_tree.query(
            hypothesis_locations,
            k=self.max_nneighbors,
            p=2,
            workers=1,
        )

        if self.max_nneighbors == 1:
            nearest_node_ids = np.expand_dims(nearest_node_ids, axis=1)

        # Get nearest node locations and compute custom distances
        nearest_node_locs = self.target_model.pos[nearest_node_ids]
        surface_normals = hypothesis_rotations[:, :, 2]
        object_features = self.target_model.__dict__
        max_abs_curvature = get_relevant_curvature(object_features)

        custom_nearest_node_dists = get_custom_distances(
            nearest_node_locs,
            hypothesis_locations,
            surface_normals,
            max_abs_curvature,
        )

        # Check if any nearest neighbor is within max_match_distance
        # If at least one is within distance, hypothesis has object points in ellipsoid
        min_distances = np.min(custom_nearest_node_dists, axis=1)
        has_object_points = min_distances <= self.max_match_distance

        return has_object_points

    def _create_statistics_text(
        self,
        timestep: int,
        hypotheses_locations: np.ndarray,
        hypotheses_evidences: np.ndarray,
        mlh_location: np.ndarray,
        mlh_graph_id: str,
    ) -> str:
        """Create statistics text for display.

        Args:
            timestep: Current timestep
            hypotheses_locations: Array of hypothesis locations
            hypotheses_evidences: Array of evidence values
            mlh_location: Location of most likely hypothesis
            mlh_graph_id: Graph ID of most likely hypothesis

        Returns:
            Formatted statistics text
        """
        if (
            self.max_hypotheses_to_show is not None
            and len(hypotheses_locations) > self.max_hypotheses_to_show
        ):
            sorted_indices = np.argsort(hypotheses_evidences)[::-1]
            top_indices = sorted_indices[: self.max_hypotheses_to_show]
            filtered_locations = hypotheses_locations[top_indices]
            # Re-check which filtered locations have object points
            filtered_has_objects = self._check_object_points_in_ellipsoids(
                filtered_locations, self.current_hypotheses_rotations[top_indices]
            )
            num_within = np.sum(filtered_has_objects)
            num_outside = len(filtered_locations) - num_within
        else:
            # Get all hypothesis rotations
            all_rotations = self.current_hypotheses_rotations
            num_within, num_outside = self._count_points_within_ellipsoids(
                hypotheses_locations, all_rotations
            )

        stats_text = (
            f"Target: {self.target_object_name}\n"
            f"Object position: [{self.target_position[0]:.3f}, "
            f"{self.target_position[1]:.3f}, {self.target_position[2]:.3f}]\n"
            f"Object rotation: [{self.target_rotation[0]:.3f}, "
            f"{self.target_rotation[1]:.3f}, {self.target_rotation[2]:.3f}]\n\n"
            f"Timestep: {timestep}\n"
            f"Total hypotheses: {len(hypotheses_locations)} (showing top {self.max_hypotheses_to_show})\n"
            f"With object points in ellipsoid: {num_within} (green)\n"
            f"Without object points: {num_outside} (red)\n"
            f"Evidence range: [{hypotheses_evidences.min():.4f}, "
            f"{hypotheses_evidences.max():.4f}]\n"
            f"Current MLH object: {mlh_graph_id}\n"
            f"MLH location: [{mlh_location[0]:.3f}, {mlh_location[1]:.3f}, "
            f"{mlh_location[2]:.3f}]"
        )

        return stats_text


def plot_target_hypotheses(
    exp_path: str,
    model_name: str = "dist_agent_1lm",
    max_match_distance: float = 0.001,
    max_nneighbors: int = 3,
) -> int:
    """Plot target object hypotheses with interactive timestep slider.

    Args:
        exp_path: Path to experiment directory containing detailed_run_stats.json
        model_name: Name of pretrained model to load object from
        max_match_distance: Maximum distance for matching (default: 0.05)
        max_nneighbors: Maximum number of nearest neighbors to consider (default: 3)

    Returns:
        Exit code
    """
    json_path = Path(exp_path) / "detailed_run_stats.json"

    if not json_path.exists():
        logger.error(f"Could not find detailed_run_stats.json at {json_path}")
        return 1

    visualizer = HypothesesRadiusVisualizer(
        str(json_path), model_name, max_match_distance, max_nneighbors
    )
    visualizer.create_interactive_visualization()

    return 0


def add_subparser(
    subparsers: argparse._SubParsersAction,
    parent_parser: argparse.ArgumentParser | None = None,
) -> None:
    """Add the hypothesis_out_of_frame subparser to the main parser.

    Args:
        subparsers: The subparsers object from the main parser.
        parent_parser: Optional parent parser for shared arguments.
    """
    parser = subparsers.add_parser(
        "hypothesis_out_of_frame_radius",
        help="Interactive visualization of target object hypothesis locations using radius-based evidence.",
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
        "--max_match_distance",
        type=float,
        default=0.05,
        help="Maximum distance for matching (default: 0.05).",
    )
    parser.add_argument(
        "--max_nneighbors",
        type=int,
        default=3,
        help="Maximum number of nearest neighbors to consider (default: 3).",
    )
    parser.set_defaults(
        func=lambda args: sys.exit(
            plot_target_hypotheses(
                args.experiment_log_dir,
                args.model_name,
                args.max_match_distance,
                args.max_nneighbors,
            )
        )
    )
