# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Interactive tool for visualizing hypotheses that are out of object's reference frame.

This visualizer requires that experiments have been run with detailed logging
enabled to generate detailed_run_stats.json files. To enable detailed logging,
use DetailedEvidenceLMLoggingConfig in your experiment configuration.

Usage:
    python tools/plot/cli.py hypothesis_oorf <experiment_log_dir>
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from vedo import (
    Arrow,
    Cube,
    Ellipsoid,
    Image,
    Plotter,
    Point,
    Points,
    Sphere,
    Text2D,
    settings,
)

from tbp.monty.frameworks.run_env import setup_env
from tbp.monty.frameworks.utils.graph_matching_utils import get_custom_distances
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    rotate_pose_dependent_features,
)

from .data_models import ObjectModelForVisualization
from .episode_loader import EpisodeDataLoader, get_model_path

if TYPE_CHECKING:
    import argparse

    from vedo import Button, Slider2D

logger = logging.getLogger(__name__)

settings.immediate_rendering = False
settings.default_font = "Calco"

setup_env()

TBP_COLORS = {
    "black": "#000000",
    "blue": "#00A0DF",
    "pink": "#F737BD",
    "purple": "#5D11BF",
    "green": "#008E42",
    "yellow": "#FFBE31",
}


def is_hypothesis_inside_object_reference_frame(
    target_model: ObjectModelForVisualization,
    hypothesis_locations: np.ndarray,
    hypothesis_rotations: np.ndarray,
    sensed_rotation: np.ndarray,
    max_abs_curvature: float,
    max_nneighbors: int = 3,
    max_match_distance: float = 0.01,
) -> np.ndarray:
    """Check if hypotheses are in the object reference frame.

    Args:
        target_model: ObjectModel containing points and features.
        hypothesis_locations: Array of hypothesis locations (n_hypotheses, 3).
        hypothesis_rotations: Rotation matrices for each hypothesis (n_hypotheses, 3, 3).
        sensed_rotation: Current sensed pose vectors (3, 3) to be transformed by hypotheses.
        max_abs_curvature: Sensed curvature of the object (max absolute value of principal_curvature_log)
        max_nneighbors: Maximum number of nearest neighbors to consider (default: 3).
        max_match_distance: Maximum distance for matching (default: 0.01).

    Returns:
        Array of booleans indicating if each hypothesis is in the object reference frame.
    """
    _, nearest_node_ids = target_model.kd_tree.query(
        hypothesis_locations,
        k=max_nneighbors,
        p=2,
        workers=1,
    )

    if max_nneighbors == 1:
        nearest_node_ids = np.expand_dims(nearest_node_ids, axis=1)

    nearest_node_locs = target_model.object_points_wrt_world[nearest_node_ids]

    # Transform sensed rotation by all hypothesis rotations at once using spatial_arithmetics
    # Create features dict for rotate_pose_dependent_features
    features = {"pose_vectors": sensed_rotation}

    # hyp_rotation: R^W_B (body frame → world frame, after world transform)
    # sensed_rotation: R_sensed^B (pose vectors in body/sensor frame)
    # Result: R^W_B * R_sensed^B = pose vectors transformed to world frame
    rotated_features = rotate_pose_dependent_features(features, hypothesis_rotations)
    transformed_pose_vectors = rotated_features[
        "pose_vectors"
    ]  # Shape: (n_hypotheses, 3, 3)

    # Extract surface normals (first row of each pose vector matrix)
    surface_normals = transformed_pose_vectors[:, 0, :]  # Shape: (n_hypotheses, 3)

    custom_nearest_node_dists = get_custom_distances(
        nearest_node_locs,
        hypothesis_locations,
        surface_normals,
        max_abs_curvature,
    )
    node_distance_weights = (
        max_match_distance - custom_nearest_node_dists
    ) / max_match_distance
    mask = node_distance_weights <= 0

    # A hypothesis is outside if ALL its nearest neighbors are outside.
    is_outside = np.all(mask, axis=1)
    return {
        "is_inside_reference_frame": ~is_outside,
        "nearest_node_ids": nearest_node_ids,
        "nearest_node_locs": nearest_node_locs,
        "custom_nearest_node_dists": custom_nearest_node_dists,
    }


class HypothesesOORFVisualizer:
    """Interactive visualizer for hypotheses that are out of object's reference frame.

    Args:
        json_path: Path to the detailed_run_stats.json file containing episode data.
        model_name: Name of pretrained model to load object from.
        max_match_distance: Maximum distance for matching.
        max_nneighbors: Maximum number of nearest neighbors to consider.
    """

    def __init__(
        self,
        json_path: Path,
        model_path: Path,
        episode_id: int = 0,
    ):
        # ============ CONFIGURATION PARAMETERS ============
        self.json_path = json_path
        self.model_path = model_path
        self.episode_id = episode_id
        self.max_match_distance = 0.01
        self.max_nneighbors = 3
        self.current_timestep = 0

        # ============ DATA ============
        self.data_loader = EpisodeDataLoader(
            self.json_path, self.model_path, self.episode_id
        )
        self.current_hypotheses_locations = None
        self.current_hypotheses_rotations = None
        self.current_highest_evidence_location = None
        self.current_highest_evidence_rotation = None
        self.current_mlh_graph_id = None
        self.current_sm0_location = None
        self.current_sm1_location = None
        self.current_sm0_rgba = None
        self.current_sm1_rgba = None
        self.current_sensed_curvature = None

        # ============ 3D VEDO OBJECTS ============
        self.target_pointcloud = None
        self.hypotheses_points = []
        self.hypothesis_ellipsoids = []
        self.ellipsoid_centers = []
        self.hypothesis_axes = []
        self.hypothesis_neighbor_points = []
        self.mlh_cube = None
        self.mlh_ellipsoid = None
        self.mlh_axes = []
        self.sm0_sphere = None
        self.sm1_sphere = None
        self.sm0_image = None
        self.sm1_image = None
        self.sm0_label = None
        self.sm1_label = None
        self.pose_vector_arrows = []
        self.pose_vectors_visible = False
        self.current_hypothesis_index = 0

        # ============ UI ============
        self.plotter = None
        self.slider = None
        self.stats_text = None

        # ============ RENDERER ============
        self.main_renderer_ix = 0
        self.sm0_renderer_ix = 1
        self.sm1_renderer_ix = 2

        self.data_loader.load_episode_data()

    def create_interactive_visualization(self) -> None:
        """Create interactive visualization with slider for timestep navigation."""
        # Create plotter main view and 2 small overlays for sensor images
        custom_shape = [
            dict(bottomleft=(0.0, 0.0), topright=(1.0, 1.0)),  # Main view (full window)
            dict(bottomleft=(0.73, 0.79), topright=(0.86, 0.99)),  # SM_0 (top-right)
            dict(bottomleft=(0.86, 0.79), topright=(0.99, 0.99)),  # SM_1 (top-right)
        ]

        self.plotter = Plotter(
            shape=custom_shape,
            size=(1400, 1000),
            sharecam=False,
            title=f"Hypotheses Out of Reference Frame",
        )

        self.update_visualization(timestep=0)

        self.slider = self.plotter.at(self.main_renderer_ix).add_slider(
            self.slider_callback,
            xmin=0,
            xmax=self.data_loader.num_lm_steps - 1,
            value=0,
            pos=[(0.2, 0.05), (0.8, 0.05)],
            title="LM step",
            show_value=False,
        )

        self.resample_button = self.plotter.at(self.main_renderer_ix).add_button(
            self.resample_ellipsoids_callback,
            pos=(0.40, 0.1),
            states=[" Resample Hypothesis "],
            size=20,
            font="Calco",
        )

        self.pose_vectors_button = self.plotter.at(self.main_renderer_ix).add_button(
            self.toggle_pose_vectors_callback,
            pos=(0.6, 0.1),
            states=[" Show Pose Vectors ", " Hide Pose Vectors "],
            size=20,
            font="Calco",
        )

        self._add_legend()

        self.plotter.at(self.sm0_renderer_ix).axes = 0
        self.plotter.at(self.sm0_renderer_ix).resetcam = True

        self.plotter.at(self.sm1_renderer_ix).axes = 0
        self.plotter.at(self.sm1_renderer_ix).resetcam = True

        self.plotter.at(self.main_renderer_ix).axes = {
            "xtitle": "X",
            "ytitle": "Y",
            "ztitle": "Z",
            "xrange": (-0.2, 0.2),
            "yrange": (1.3, 1.7),
            "zrange": (-0.2, 0.2),
        }

        self.plotter.at(self.main_renderer_ix).show(
            axes=True,
            viewup="y",
            camera=dict(pos=(0.5, 1.5, 2.0), focal_point=(0, 1.5, 0)),
        )
        self.plotter.show(interactive=True)

    def update_visualization(self, timestep: int) -> None:
        """Update visualization for given timestep."""
        self._cleanup_previous_visualizations()

        self.current_timestep = timestep
        self._initialize_timestep_data(self.current_timestep)

        if self.target_pointcloud is None:
            self._initialize_target_visualization()

        hypotheses_oorf_info = is_hypothesis_inside_object_reference_frame(
            self.object_model,
            self.current_hypotheses_locations,
            self.current_hypotheses_rotations,
            self.current_sensed_rotation,
            self.current_sensed_curvature,
            self.max_nneighbors,
            self.max_match_distance,
        )
        self.hypotheses_inside_reference_frame = hypotheses_oorf_info[
            "is_inside_reference_frame"
        ]
        self.hypotheses_nearest_node_locs = hypotheses_oorf_info["nearest_node_locs"]
        self.hypotheses_custom_nearest_node_dists = hypotheses_oorf_info[
            "custom_nearest_node_dists"
        ]

        self._add_hypotheses_points()

        # Select a random hypothesis and add ellipsoid, center point, nearest neighbors, and axes arrows
        idx = np.random.choice(len(self.current_hypotheses_locations), 1)[0]
        self.current_hypothesis_index = idx
        self._add_ellipsoid(
            self.current_hypotheses_locations[idx],
            self.current_hypotheses_rotations[idx],
            self.hypotheses_inside_reference_frame[idx],
            is_mlh=False,
        )
        self._add_hypothesis_center_point(self.current_hypotheses_locations[idx])
        self._add_nearest_neighbor_points(self.hypotheses_nearest_node_locs[idx])
        self._add_axes_arrows(
            self.current_hypotheses_locations[idx],
            self.current_hypotheses_rotations[idx],
        )

        mlh_oorf_info = is_hypothesis_inside_object_reference_frame(
            self.object_model,
            self.current_highest_evidence_location.reshape(1, 3),
            self.current_highest_evidence_rotation.reshape(1, 3, 3),
            self.current_sensed_rotation,
            self.current_sensed_curvature,
            self.max_nneighbors,
            self.max_match_distance,
        )
        self.is_mlh_inside_reference_frame = mlh_oorf_info["is_inside_reference_frame"]
        self.mlh_nearest_node_locs = mlh_oorf_info["nearest_node_locs"]
        self.mlh_custom_nearest_node_dists = mlh_oorf_info["custom_nearest_node_dists"]

        self._add_ellipsoid(
            self.current_highest_evidence_location,
            self.current_highest_evidence_rotation,
            self.is_mlh_inside_reference_frame,
            is_mlh=True,
        )
        self._add_mlh_cube(
            self.current_highest_evidence_location, self.is_mlh_inside_reference_frame
        )
        self._add_nearest_neighbor_points(self.mlh_nearest_node_locs)
        self._add_axes_arrows(
            self.current_highest_evidence_location,
            self.current_highest_evidence_rotation,
        )

        self._add_sensor_spheres(timestep)
        self._add_sensor_images(timestep)

        stats_text = self._create_summary_text(
            timestep,
        )
        # Left-justified stats text in top-left, lowered to avoid cutoff
        self.stats_text = Text2D(
            stats_text,
            pos="top-left",
            s=0.8,
            font="Calco",
        )
        self.plotter.at(self.main_renderer_ix).add(self.stats_text)

        # Add compact legend in bottom-left corner
        self._add_legend()

    def slider_callback(self, widget: Slider2D, _event: str) -> None:
        """Respond to slider step by updating the visualization."""
        timestep = round(widget.GetRepresentation().GetValue())
        if timestep != self.current_timestep:
            self.update_visualization(timestep)
            self.plotter.render()

    def resample_ellipsoids_callback(self, _widget: Button, _event: str) -> None:
        """Resample the ellipsoids to show a different random selection."""
        for ellipsoid in self.hypothesis_ellipsoids:
            self.plotter.remove(ellipsoid)
        self.hypothesis_ellipsoids = []

        for point in self.ellipsoid_centers:
            self.plotter.remove(point)
        self.ellipsoid_centers = []

        for arrow in self.hypothesis_axes:
            self.plotter.remove(arrow)
        self.hypothesis_axes = []

        for points in self.hypothesis_neighbor_points:
            self.plotter.remove(points)
        self.hypothesis_neighbor_points = []

        for arrow in self.mlh_axes:
            self.plotter.remove(arrow)
        self.mlh_axes = []

        # Remove MLH ellipsoid and cube
        if self.mlh_ellipsoid is not None:
            self.plotter.remove(self.mlh_ellipsoid)
            self.mlh_ellipsoid = None
        if self.mlh_cube is not None:
            self.plotter.remove(self.mlh_cube)
            self.mlh_cube = None

        # Select a new random hypothesis and add ellipsoid, center point, nearest neighbors, and axes arrows
        idx = np.random.choice(len(self.current_hypotheses_locations), 1)[0]
        self.current_hypothesis_index = idx
        self._add_ellipsoid(
            self.current_hypotheses_locations[idx],
            self.current_hypotheses_rotations[idx],
            self.hypotheses_inside_reference_frame[idx],
            is_mlh=False,
        )
        self._add_hypothesis_center_point(self.current_hypotheses_locations[idx])
        self._add_nearest_neighbor_points(self.hypotheses_nearest_node_locs[idx])
        self._add_axes_arrows(
            self.current_hypotheses_locations[idx],
            self.current_hypotheses_rotations[idx],
        )

        # Also re-add MLH ellipsoid, cube, nearest neighbors, and axes
        self._add_ellipsoid(
            self.current_highest_evidence_location,
            self.current_highest_evidence_rotation,
            self.is_mlh_inside_reference_frame,
            is_mlh=True,
        )
        self._add_mlh_cube(
            self.current_highest_evidence_location, self.is_mlh_inside_reference_frame
        )
        self._add_nearest_neighbor_points(self.mlh_nearest_node_locs)
        self._add_axes_arrows(
            self.current_highest_evidence_location,
            self.current_highest_evidence_rotation,
        )

        # Update statistics text with new ellipsoid and NN info
        if hasattr(self, "stats_text"):
            self.plotter.remove(self.stats_text)

        stats_text = self._create_summary_text(
            self.current_timestep,
        )
        # Left-justified stats text in top-left, lowered to avoid cutoff
        self.stats_text = Text2D(
            stats_text,
            pos="top-left",
            s=0.8,
            font="Calco",
        )
        self.plotter.at(self.main_renderer_ix).add(self.stats_text)

        self.plotter.at(self.sm0_renderer_ix).render()
        self.plotter.at(self.sm1_renderer_ix).render()
        self.plotter.at(self.main_renderer_ix).render()

    def toggle_pose_vectors_callback(self, _widget: Button, _event: str) -> None:
        """Toggle visibility of pose vector arrows."""
        self.pose_vectors_visible = not self.pose_vectors_visible

        if self.pose_vectors_visible:
            self._add_pose_vector_arrows()
        else:
            for arrow in self.pose_vector_arrows:
                self.plotter.at(self.main_renderer_ix).remove(arrow)
            self.pose_vector_arrows = []

        self.plotter.at(self.main_renderer_ix).render()

    def _initialize_timestep_data(self, timestep: int) -> tuple:
        """Get data for a specific timestep.

        Args:
            timestep: The timestep to retrieve data for
        """
        self.current_hypotheses_locations = self.data_loader.all_hyp_locations[timestep]
        self.current_hypotheses_rotations = (
            self.data_loader.all_hyp_object_orientations[timestep]
        )
        self.current_highest_evidence_location = (
            self.data_loader.highest_evidence_location[timestep]
        )
        self.current_highest_evidence_rotation = (
            self.data_loader.highest_evidence_object_orientation[timestep]
        )
        self.current_mlh_graph_id = self.data_loader.all_mlh_graph_ids[timestep]
        self.current_sm0_location = self.data_loader.all_sm0_locations[timestep]
        self.current_sm1_location = self.data_loader.all_sm1_locations[timestep]
        self.current_sm0_rgba = self.data_loader.all_sm0_rgba[timestep]
        self.current_sm1_rgba = self.data_loader.all_sm1_rgba[timestep]
        self.current_sensed_curvature = self.data_loader.max_abs_curvatures[timestep]
        self.current_sensed_rotation = self.data_loader.sensed_orientations[timestep]

    def _cleanup_previous_visualizations(self) -> None:
        """Remove previous visualization objects from the plotter."""
        if self.hypotheses_points is not None:
            if isinstance(self.hypotheses_points, list):
                for pts in self.hypotheses_points:
                    self.plotter.at(self.main_renderer_ix).remove(pts)
            else:
                self.plotter.at(self.main_renderer_ix).remove(self.hypotheses_points)
        if self.stats_text is not None:
            self.plotter.at(self.main_renderer_ix).remove(self.stats_text)
        for ellipsoid in self.hypothesis_ellipsoids:
            self.plotter.at(self.main_renderer_ix).remove(ellipsoid)
        self.hypothesis_ellipsoids = []
        for point in self.ellipsoid_centers:
            self.plotter.at(self.main_renderer_ix).remove(point)
        self.ellipsoid_centers = []
        for arrow in self.hypothesis_axes:
            self.plotter.at(self.main_renderer_ix).remove(arrow)
        self.hypothesis_axes = []
        for points in self.hypothesis_neighbor_points:
            self.plotter.at(self.main_renderer_ix).remove(points)
        self.hypothesis_neighbor_points = []
        for arrow in self.mlh_axes:
            self.plotter.at(self.main_renderer_ix).remove(arrow)
        self.mlh_axes = []
        if self.mlh_cube is not None:
            self.plotter.at(self.main_renderer_ix).remove(self.mlh_cube)
            self.mlh_cube = None
        if self.mlh_ellipsoid is not None:
            self.plotter.at(self.main_renderer_ix).remove(self.mlh_ellipsoid)
            self.mlh_ellipsoid = None
        if self.sm0_sphere is not None:
            self.plotter.at(self.main_renderer_ix).remove(self.sm0_sphere)
            self.sm0_sphere = None
        if self.sm1_sphere is not None:
            self.plotter.at(self.main_renderer_ix).remove(self.sm1_sphere)
            self.sm1_sphere = None
        if self.sm0_image is not None:
            self.plotter.at(self.sm0_renderer_ix).remove(self.sm0_image)
            self.sm0_image = None
        if self.sm0_label is not None:
            self.plotter.at(self.sm0_renderer_ix).remove(self.sm0_label)
            self.sm0_label = None
        if self.sm1_image is not None:
            self.plotter.at(self.sm1_renderer_ix).remove(self.sm1_image)
            self.sm1_image = None
        if self.sm1_label is not None:
            self.plotter.at(self.sm1_renderer_ix).remove(self.sm1_label)
            self.sm1_label = None
        for arrow in self.pose_vector_arrows:
            self.plotter.at(self.main_renderer_ix).remove(arrow)
        self.pose_vector_arrows = []

    def _initialize_target_visualization(self) -> None:
        """Initialize object model and convex hull visualization."""
        self.object_model = self.data_loader.object_model

        self.target_pointcloud = Points(self.object_model.object_points_wrt_world, c="gray")
        self.target_pointcloud.point_size(8)
        self.plotter.at(self.main_renderer_ix).add(self.target_pointcloud)

    def _add_hypotheses_points(
        self,
    ) -> None:
        """Plot hypotheses in world frame colored by OORF status.

        Args:
            hypotheses_locations: Array of hypothesis locations in world frame
            hypotheses_rotations: Array of rotation matrices for each hypothesis in world frame
        """
        inside_hypotheses = self.current_hypotheses_locations[
            self.hypotheses_inside_reference_frame
        ]
        outside_hypotheses = self.current_hypotheses_locations[
            ~self.hypotheses_inside_reference_frame
        ]

        if len(inside_hypotheses) > 0:
            inside_points = Points(inside_hypotheses, c=TBP_COLORS["blue"])
            inside_points.point_size(8)
            self.hypotheses_points.append(inside_points)
            self.plotter.at(self.main_renderer_ix).add(inside_points)

        if len(outside_hypotheses) > 0:
            outside_points = Points(outside_hypotheses, c=TBP_COLORS["pink"])
            outside_points.point_size(3)
            self.hypotheses_points.append(outside_points)
            self.plotter.at(self.main_renderer_ix).add(outside_points)

    def _add_sensor_spheres(self, timestep: int) -> None:
        """Add spheres to visualize sensor locations."""
        self.sm0_sphere = Sphere(
            self.current_sm0_location,
            r=0.003,
            c=TBP_COLORS["green"],
            alpha=0.8,
        )
        self.plotter.at(self.main_renderer_ix).add(self.sm0_sphere)

        self.sm1_sphere = Sphere(
            self.current_sm1_location,
            r=0.005,
            c=TBP_COLORS["yellow"],
            alpha=0.6,
        )
        self.plotter.at(self.main_renderer_ix).add(self.sm1_sphere)

    def _add_sensor_images(self, timestep: int) -> None:
        """Add sensor RGB patch visualizations to separate renderers."""
        rgba_patch = self.current_sm0_rgba
        rgb_patch = rgba_patch[:, :, :3]

        self.sm0_image = Image(rgb_patch)
        self.plotter.at(self.sm0_renderer_ix).add(self.sm0_image)

        self.sm0_label = Text2D("SM_0", pos="top-center", c="black", font="Calco")
        self.plotter.at(self.sm0_renderer_ix).add(self.sm0_label)

        rgba_patch = self.current_sm1_rgba
        rgb_patch = rgba_patch[:, :, :3]

        self.sm1_image = Image(rgb_patch)
        self.plotter.at(self.sm1_renderer_ix).add(self.sm1_image)

        self.sm1_label = Text2D("SM_1", pos="top-center", c="black", font="Calco")
        self.plotter.at(self.sm1_renderer_ix).add(self.sm1_label)

    def _add_ellipsoid(
        self,
        location: np.ndarray,
        hypothesis_rotation: np.ndarray,
        is_inside_reference_frame: bool,
        is_mlh: bool = False,
    ) -> None:
        """Add ellipsoid at given location with orientation from transformed sensed rotation."""
        # Transform sensed pose vectors by hypothesis rotation using spatial_arithmetics
        features = {"pose_vectors": self.current_sensed_rotation}
        # Reshape single rotation matrix to have batch dimension for consistency
        hypothesis_rotation_batch = hypothesis_rotation.reshape(1, 3, 3)
        rotated_features = rotate_pose_dependent_features(
            features, hypothesis_rotation_batch
        )
        transformed_pose_vectors = rotated_features["pose_vectors"]  # Shape: (1, 3, 3)

        surface_normal = transformed_pose_vectors[0, 0, :]  # Transformed surface normal
        tangent1 = transformed_pose_vectors[0, 1, :]  # Transformed PC1 (dir1)
        tangent2 = transformed_pose_vectors[0, 2, :]  # Transformed PC2 (dir2)

        stretch_factor = 1.0 / (np.abs(self.current_sensed_curvature) + 0.5)
        semi_axis_tangent = self.max_match_distance
        semi_axis_normal = self.max_match_distance / (1 + stretch_factor)

        color = TBP_COLORS["blue"] if is_inside_reference_frame else TBP_COLORS["pink"]

        ellipsoid = Ellipsoid(
            pos=location,
            axis1=tangent1 * semi_axis_tangent,
            axis2=tangent2 * semi_axis_tangent,
            axis3=surface_normal * semi_axis_normal,
            c=color,
        )
        ellipsoid.alpha(0.15)

        if is_mlh:
            self.mlh_ellipsoid = ellipsoid
        else:
            self.hypothesis_ellipsoids.append(ellipsoid)

        self.plotter.at(self.main_renderer_ix).add(ellipsoid)

    def _add_hypothesis_center_point(self, location: np.ndarray) -> None:
        """Add a black point at the hypothesis center."""
        hyp_point = Point(location, c="black")
        hyp_point.point_size(25)
        self.ellipsoid_centers.append(hyp_point)
        self.plotter.at(self.main_renderer_ix).add(hyp_point)

    def _add_mlh_cube(
        self, location: np.ndarray, is_inside_reference_frame: bool
    ) -> None:
        """Add a cube at the MLH location."""
        self.mlh_cube = Cube(location, side=0.003, c="black", alpha=0.6)
        self.plotter.at(self.main_renderer_ix).add(self.mlh_cube)

    def _add_nearest_neighbor_points(self, nearest_node_locs: np.ndarray) -> None:
        """Add yellow points for nearest neighbors."""
        nearest_node_points = Points(
            nearest_node_locs.squeeze(), c=TBP_COLORS["yellow"]
        )
        nearest_node_points.point_size(15)
        self.hypothesis_neighbor_points.append(nearest_node_points)
        self.plotter.at(self.main_renderer_ix).add(nearest_node_points)

    def _add_axes_arrows(
        self,
        location: np.ndarray,
        hypothesis_rotation: np.ndarray,
    ) -> None:
        """Add arrows showing transformed sensed tangent and normal directions."""
        arrow_length = 0.02

        # Transform sensed pose vectors by hypothesis rotation using spatial_arithmetics
        features = {"pose_vectors": self.current_sensed_rotation}
        # Reshape single rotation matrix to have batch dimension for consistency
        hypothesis_rotation_batch = hypothesis_rotation.reshape(1, 3, 3)
        rotated_features = rotate_pose_dependent_features(
            features, hypothesis_rotation_batch
        )
        transformed_pose_vectors = rotated_features["pose_vectors"]  # Shape: (1, 3, 3)

        # Pose vectors are in Darboux Frame
        surface_normal = transformed_pose_vectors[0, 0, :]  # Transformed surface normal
        tangent1 = transformed_pose_vectors[0, 1, :]  # Transformed PC1 (dir1)
        tangent2 = transformed_pose_vectors[0, 2, :]  # Transformed PC2 (dir2)

        arrow1 = Arrow(
            location,
            location + tangent1 * arrow_length,
            c=TBP_COLORS["purple"],
        )
        arrow1.alpha(0.7)
        self.hypothesis_axes.append(arrow1)
        self.plotter.at(self.main_renderer_ix).add(arrow1)

        arrow2 = Arrow(
            location,
            location + tangent2 * arrow_length,
            c=TBP_COLORS["green"],
        )
        arrow2.alpha(0.7)
        self.hypothesis_axes.append(arrow2)
        self.plotter.at(self.main_renderer_ix).add(arrow2)

        arrow3 = Arrow(
            location,
            location + surface_normal * arrow_length,
            c=TBP_COLORS["yellow"],
        )
        arrow3.alpha(0.9)
        self.hypothesis_axes.append(arrow3)
        self.plotter.at(self.main_renderer_ix).add(arrow3)

    def _add_pose_vector_arrows(self) -> None:
        """Add arrows showing surface normals from object_model pose vectors."""
        if not hasattr(self.object_model, "pose_vectors"):
            logger.warning("Object model does not have pose_vectors attribute")
            return

        arrow_length = 0.01
        # Sample more points for better coverage (every 10th point)
        sample_indices = np.arange(0, len(self.object_model.object_points_wrt_world), 4)

        locations = self.object_model.object_points_wrt_world[sample_indices]
        pose_vectors = self.object_model.object_feature_orientations_wrt_world[
            sample_indices
        ]  # Shape: (n_sampled, 9)

        # Reshape pose vectors to matrices (n_sampled, 3, 3)
        pose_matrices = pose_vectors.reshape(-1, 3, 3)

        for i, (location, pose_matrix) in enumerate(zip(locations, pose_matrices)):
            # Extract only surface normal (first row)
            surface_normal = pose_matrix[0, :]  # First row

            # Surface normal arrow (gray)
            arrow_normal = Arrow(
                location,
                location + surface_normal * arrow_length,
                c="gray",
            )
            arrow_normal.alpha(0.4)
            self.pose_vector_arrows.append(arrow_normal)
            self.plotter.at(self.main_renderer_ix).add(arrow_normal)

    def _compute_surface_normal_comparison(
        self, hypothesis_location: np.ndarray, hypothesis_rotation: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Compare hypothesis surface normal with nearest object model surface normal.

        Args:
            hypothesis_location: Location of the hypothesis (3,)
            hypothesis_rotation: Rotation matrix of the hypothesis (3, 3)

        Returns:
            Tuple of (angle_degrees, hypothesis_surface_normal, nearest_model_surface_normal)
        """
        # Get transformed surface normal from hypothesis
        features = {"pose_vectors": self.current_sensed_rotation}
        hypothesis_rotation_batch = hypothesis_rotation.reshape(1, 3, 3)
        rotated_features = rotate_pose_dependent_features(
            features, hypothesis_rotation_batch
        )
        transformed_pose_vectors = rotated_features["pose_vectors"]
        hypothesis_surface_normal = transformed_pose_vectors[0, 0, :]  # First row

        # Find nearest point in object model
        distance, nearest_idx = self.object_model.kd_tree.query(
            hypothesis_location, k=1
        )

        # Get surface normal from object model at nearest point
        nearest_pose_vectors = self.object_model.pose_vectors[nearest_idx].reshape(3, 3)
        model_surface_normal = nearest_pose_vectors[0, :]  # First row

        # Compute angle between surface normals
        dot_product = np.clip(
            np.dot(hypothesis_surface_normal, model_surface_normal), -1.0, 1.0
        )
        angle_radians = np.arccos(np.abs(dot_product))  # Use abs to get acute angle
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees, hypothesis_surface_normal, model_surface_normal

    def _create_summary_text(
        self,
        timestep: int,
    ) -> str:
        """Create summary text for current timestep.

        Args:
            timestep: Current timestep

        Returns:
            Formatted statistics text
        """
        num_hypotheses_in_reference_frame = sum(self.hypotheses_inside_reference_frame)
        num_hypotheses_outside_reference_frame = sum(
            ~self.hypotheses_inside_reference_frame
        )

        formatted_position = np.array2string(
            np.array(self.data_loader.ground_truth_position),
            precision=2,  # show up to 2 decimal places
            separator=", ",
            suppress_small=False,
        )
        formatted_rotation = np.array2string(
            np.array(self.data_loader.ground_truth_rotation),
            precision=2,
            separator=", ",
            suppress_small=False,
        )

        sm_step = self.data_loader.lm_to_sm_mapping[timestep]
        object_summary = [
            f"Object: {self.data_loader.target_name}",
            f"Object position: {formatted_position}",
            f"Object rotation: {formatted_rotation}",
            f"LM Step: {timestep}",
            f"SM Step: {sm_step}",
        ]

        hypotheses_summary = [
            f"Num Hyp. Inside Ref. Frame: {num_hypotheses_in_reference_frame}",
            f"Num Hyp. Outside Ref. Frame: {num_hypotheses_outside_reference_frame}",
            f"Current MLH: {self.current_mlh_graph_id}",
        ]

        summary_text = "\n".join(object_summary + hypotheses_summary)
        return summary_text

    def _add_legend(self) -> None:
        """Add a legend with color-coded text."""
        legend_title = Text2D(
            "Legend",
            pos=(0.02, 0.25),
            s=0.8,
            font="Calco",
            c="black",
        )
        self.plotter.at(self.main_renderer_ix).add(legend_title)

        legend_items = [
            ("Inside RF (Point)", TBP_COLORS["blue"], 0.22),
            ("Outside RF (Point)", TBP_COLORS["pink"], 0.20),
            ("PC1 Axis (Arrow)", TBP_COLORS["purple"], 0.18),
            ("PC2 Axis (Arrow)", TBP_COLORS["green"], 0.16),
            ("Surface Normal (Arrow)", TBP_COLORS["yellow"], 0.14),
            ("MLH (Cube)", "black", 0.12),
            ("Nearest Neighbor (Point)", TBP_COLORS["yellow"], 0.1),
        ]

        for text, color, y_pos in legend_items:
            item = Text2D(
                text,
                pos=(0.02, y_pos),
                s=0.65,
                font="Courier",
                c=color,
            )
            self.plotter.at(self.main_renderer_ix).add(item)


def plot_target_hypotheses(
    exp_path: Path,
    episode_id: int = 0,
) -> int:
    """Plot target object hypotheses with interactive timestep slider.

    Args:
        exp_path: Path to experiment directory containing detailed_run_stats.json
        episode_id: Episode ID to visualize (default: 0)

    Returns:
        Exit code
    """
    json_path = Path(exp_path) / "detailed_run_stats.json"

    if not json_path.exists():
        logger.error(f"Could not find detailed_run_stats.json at {json_path}")
        return 1

    model_path = get_model_path(Path(exp_path))

    visualizer = HypothesesOORFVisualizer(json_path, model_path, episode_id)
    visualizer.create_interactive_visualization()

    return 0


def add_subparser(
    subparsers: argparse._SubParsersAction,
    parent_parser: argparse.ArgumentParser | None = None,
) -> None:
    """Add the hypothesis_oorf subparser to the main parser.

    Args:
        subparsers: The subparsers object from the main parser.
        parent_parser: Optional parent parser for shared arguments.
    """
    parser = subparsers.add_parser(
        "hypothesis_oorf",
        help="Interactive tool to visualize hypotheses' locations and rotations.",
        parents=[parent_parser] if parent_parser else [],
    )
    parser.add_argument(
        "experiment_log_dir",
        help="The directory containing the detailed_run_stats.json file.",
    )

    parser.add_argument(
        "--episode_id",
        type=int,
        default=0,
        help="The episode ID to visualize.",
    )
    parser.set_defaults(
        func=lambda args: sys.exit(
            plot_target_hypotheses(
                args.experiment_log_dir,
                args.episode_id,
            )
        )
    )
