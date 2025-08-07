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
    python tools/plot/cli.py hypothesis_out_of_frame_radius <experiment_log_dir>
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
from vedo import (
    Ellipsoid,
    Image,
    Plotter,
    Points,
    Sphere,
    Text2D,
    settings,
)

from tbp.monty.frameworks.run_env import setup_env

from .hypothesis_out_of_frame_data_models import (
    EpisodeDataLoader,
    ObjectModel,
    load_object_model,
)
from .hypothesis_out_of_frame_geometry import is_hypothesis_in_object_reference_frame

if TYPE_CHECKING:
    import argparse

    from vedo import Button, Slider2D

logger = logging.getLogger(__name__)

# Vedo settings
settings.immediate_rendering = False
settings.default_font = "Theemim"

setup_env()

TBP_COLORS = {
    "black": "#000000",
    "blue": "#00A0DF",
    "pink": "#F737BD",
    "purple": "#5D11BF",
    "green": "#008E42",
    "yellow": "#FFBE31",
}

def is_hypothesis_in_object_reference_frame(
    hypothesis_locations: np.ndarray,
    target_model: ObjectModel,
    hypothesis_rotations: np.ndarray,
    max_nneighbors: int = 3,
    max_match_distance: float = 0.001,
) -> np.ndarray:
    """Check if hypotheses are in the object reference frame.

    Args:
        hypothesis_locations: Array of hypothesis locations (n_hypotheses, 3).
        target_model: ObjectModel containing points and features.
        hypothesis_rotations: Rotation matrices for each hypothesis (n_hypotheses, 3, 3).
        max_nneighbors: Maximum number of nearest neighbors to consider (default: 3).
        max_match_distance: Maximum distance for matching (default: 0.001).

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
    node_distance_weights = (
        max_match_distance - custom_nearest_node_dists
    ) / max_match_distance
    mask = node_distance_weights <= 0

    # A hypothesis is outside if ALL its nearest neighbors are outside.
    is_outside = np.all(mask, axis=1)
    return is_outside


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
        json_path: str,
        model_name: str = "dist_agent_1lm",
        max_match_distance: float = 0.01,
        max_nneighbors: int = 3,
    ):
        # ============ CONFIGURATION PARAMETERS ============
        self.json_path = json_path
        self.model_name = model_name
        self.max_match_distance = max_match_distance
        self.max_nneighbors = max_nneighbors
        self.max_ellipsoids_to_show = 1
        self.max_hypotheses_to_show = None  # None means show all

        # ============ APPLICATION STATE ============
        self.current_timestep = 0
        self.current_sampled_indices = None
        self.ellipsoids_visible = True
        self.total_hypotheses_count = 0

        # ============ DATA LOADER ============
        self.data_loader = EpisodeDataLoader(json_path)

        # ============ CURRENT TIMESTEP DATA ============
        self.current_target_locations = None
        self.current_target_evidences = None
        self.current_hypotheses_rotations = None
        self.current_ellipsoid_info = None

        # ============ 3D VEDO OBJECTS ============
        self.hypotheses = None
        self.object_points = None
        self.hypothesis_ellipsoids = []
        self.hypothesis_spheres = []
        self.mlh_cube = None
        self.sm0_sphere = None
        self.sm1_sphere = None

        # ============ UI ============
        self.plotter = None
        self.slider = None  # For timestep
        self.ellipsoid_button = None  # Toggle ellipsoid visibility
        self.resample_button = None  # Trigger hypothesis resampling
        self.stats_text = None
        self.sm0_image = None
        self.sm1_image = None
        self.sm0_label = None
        self.sm1_label = None

        # ============ RENDERER ============
        self.main_renderer_ix = 0
        self.sm0_renderer_ix = 1
        self.sm1_renderer_ix = 2

        # Load data
        self.data_loader.load_episode_data()
        self.load_target_model()


    def load_target_model(self) -> None:
        """Load the target object model."""
        self.target_model = load_object_model(
            self.model_name, self.data_loader.target_object_name
        )
        logger.info(
            f"Loaded {self.data_loader.target_object_name} model with "
            f"{len(self.target_model.pos)} points and features: "
            f"{list(self.target_model.__dict__.keys())}"
        )

    def create_interactive_visualization(self) -> None:
        """Create interactive visualization with slider for timestep navigation."""
        # Create plotter with multiple renderers: main view as background,
        # 2 small overlays in top-left for sensor images
        # Define custom layout with overlapping renderers
        custom_shape = [
            dict(bottomleft=(0.0, 0.0), topright=(1.0, 1.0)),  # Main view (full window)
            dict(
                bottomleft=(0.02, 0.72), topright=(0.17, 0.92), bg="white"
            ),  # SM_0 (top-left, smaller)
            dict(
                bottomleft=(0.19, 0.72), topright=(0.34, 0.92), bg="white"
            ),  # SM_1 (next to SM_0, smaller)
        ]

        self.plotter = Plotter(
            shape=custom_shape,
            size=(1400, 1000),
            sharecam=False,
            title=(
                f"Sensor Location Hypotheses (Radius-based) for "
                f"{self.data_loader.target_object_name.title()} - Episode 0"
            ),
        )

        self.update_visualization(timestep=0)

        # Add slider to the main renderer
        self.slider = self.plotter.at(self.main_renderer_ix).add_slider(
            self.slider_callback,
            xmin=0,
            xmax=self.data_loader.num_lm_steps - 1,
            value=0,
            pos=[(0.25, 0.05), (0.75, 0.05)],
            title="LM step",
        )
        self.ellipsoid_button = self.plotter.at(self.main_renderer_ix).add_button(
            self.toggle_ellipsoid_callback,
            pos=(0.85, 0.15),
            states=["Hide Ellipsoids", "Show Ellipsoids"],
            font="Calco",
            size=20,
        )
        self.resample_button = self.plotter.at(self.main_renderer_ix).add_button(
            self.resample_ellipsoids_callback,
            pos=(0.85, 0.1),
            states=["Select Different Ellipsoids"],
            font="Calco",
            size=20,
        )
        self.hypotheses_filter_slider = self.plotter.at(
            self.main_renderer_ix
        ).add_slider(
            self.hypotheses_filter_slider_callback,
            xmin=0,
            xmax=100,
            value=1,
            pos=[(0.25, 0.10), (0.65, 0.10)],
            title="Top N% Hypotheses",
            show_value=True,
        )

        # Configure each renderer
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

        # Show the plotter with all renderers
        self.plotter.at(self.main_renderer_ix).show(
            axes=True,
            viewup="y",
            camera=dict(pos=(0.5, 1.5, 2.0), focal_point=(0, 1.5, 0)),
        )
        self.plotter.show(interactive=True)

        # Clean up sensor images when vedo window is closed
        self._cleanup_sensor_images()

    def _cleanup_sensor_images(self) -> None:
        """Clean up sensor images when the visualization is closed."""
        # Cleanup handled by vedo plotter
        pass

    def update_visualization(self, timestep: int) -> None:
        """Update visualization for given timestep."""
        self.current_timestep = timestep

        self.current_sampled_indices = None

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
        self._add_sensor_spheres(timestep)
        self._add_sensor_images(timestep)

        stats_text = self._create_statistics_text(
            timestep,
            hypotheses_locations,
            hypotheses_evidences,
            mlh_location,
            mlh_graph_id,
        )
        self.stats_text = Text2D(stats_text, pos="top-right", s=0.7)
        self.plotter.at(self.main_renderer_ix).add(self.stats_text)

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

        self.current_sampled_indices = None

        self.update_visualization(self.current_timestep)
        self.plotter.render()

    def resample_ellipsoids_callback(self, _widget: Button, _event: str) -> None:
        """Resample the ellipsoids to show a different random selection."""
        # Force new random sampling
        self.current_sampled_indices = None
        self.current_ellipsoid_info = None

        # Re-render just the ellipsoids and spheres
        # First clean up existing ellipsoids and spheres
        for ellipsoid in self.hypothesis_ellipsoids:
            self.plotter.remove(ellipsoid)
        self.hypothesis_ellipsoids = []

        for sphere in self.hypothesis_spheres:
            self.plotter.remove(sphere)
        self.hypothesis_spheres = []

        # Re-add ellipsoids with new random selection
        if hasattr(self, "_current_filtered_locations"):
            self._add_hypothesis_ellipsoids(
                self._current_filtered_locations, self._current_filtered_rotations
            )

            # Update statistics text
            self.update_visualization(self.current_timestep)

        self.plotter.at(self.sm0_renderer_ix).render()
        self.plotter.at(self.sm1_renderer_ix).render()
        self.plotter.at(self.main_renderer_ix).render()

    def _get_timestep_data(self, timestep: int) -> tuple:
        """Get data for a specific timestep.

        Args:
            timestep: The timestep to retrieve data for

        Returns:
            Tuple of (hypotheses_locations, hypotheses_evidences, hypotheses_rotations,
            mlh_location, mlh_graph_id)
        """
        hypotheses_locations = self.data_loader.all_hypotheses_locations[timestep]
        hypotheses_evidences = self.data_loader.all_hypotheses_evidences[timestep]
        hypotheses_rotations = self.data_loader.all_hypotheses_rotations[timestep]
        mlh_location = self.data_loader.all_mlh_locations[timestep]
        mlh_graph_id = self.data_loader.all_mlh_graph_ids[timestep]

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
        # Clean up main renderer objects
        if self.hypotheses is not None:
            if isinstance(self.hypotheses, list):
                for pts in self.hypotheses:
                    self.plotter.at(self.main_renderer_ix).remove(pts)
            else:
                self.plotter.at(self.main_renderer_ix).remove(self.hypotheses)
        if self.mlh_sphere is not None:
            self.plotter.at(self.main_renderer_ix).remove(self.mlh_sphere)
        if self.stats_text is not None:
            self.plotter.at(self.main_renderer_ix).remove(self.stats_text)
        for ellipsoid in self.hypothesis_ellipsoids:
            self.plotter.at(self.main_renderer_ix).remove(ellipsoid)
        self.hypothesis_ellipsoids = []
        for sphere in self.hypothesis_spheres:
            self.plotter.at(self.main_renderer_ix).remove(sphere)
        self.hypothesis_spheres = []
        if self.sm0_sphere is not None:
            self.plotter.at(self.main_renderer_ix).remove(self.sm0_sphere)
            self.sm0_sphere = None
        if self.sm1_sphere is not None:
            self.plotter.at(self.main_renderer_ix).remove(self.sm1_sphere)
            self.sm1_sphere = None

        # Clean up sensor renderer objects
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

    def _initialize_object_visualization(self) -> None:
        """Initialize object model and convex hull visualization."""
        model = self.target_model.copy()

        # The pretrained models are stored at learned position [0, 1.5, 0]
        # We need to move them to origin before rotating
        learned_position = np.array([0, 1.5, 0])

        # Move object from learned position to origin
        model = model - learned_position

        # Apply ground truth rotation
        model = model.rotated(self.data_loader.target_rotation, degrees=True)

        # Translate to the ground truth world position
        model = model + self.data_loader.target_position

        # Rebuild KDTree with transformed positions for accurate distance queries
        model.kd_tree = KDTree(model.pos, leafsize=40)

        # Store the transformed model for distance calculations
        self.transformed_target_model = model

        self.object_points = Points(model.pos, c="gray")
        self.object_points.point_size(10)
        self.plotter.at(self.main_renderer_ix).add(self.object_points)

        self.plotter.at(self.main_renderer_ix).add(
            Text2D(
                f"{self.data_loader.target_object_name.title()} Object (Ground Truth)",
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
        # Use transformed model if available, otherwise use original
        target_model = getattr(self, "transformed_target_model", self.target_model)

        has_object_points = ~is_hypothesis_in_object_reference_frame(
            points,
            target_model,
            rotations,
            self.max_nneighbors,
            self.max_match_distance,
        )
        num_within = np.sum(has_object_points)
        num_outside = len(points) - num_within

        return num_within, num_outside

    def _sample_ellipsoid_indices(self, num_hypotheses: int) -> np.ndarray:
        """Randomly sample indices for ellipsoid visualization.

        Args:
            num_hypotheses: Total number of hypotheses available

        Returns:
            Array of sampled indices
        """
        if num_hypotheses <= self.max_ellipsoids_to_show:
            return np.arange(num_hypotheses)
        else:
            return np.random.choice(
                num_hypotheses, self.max_ellipsoids_to_show, replace=False
            )

    def _add_hypotheses(
        self,
        hypotheses_locations: np.ndarray,
        hypotheses_evidences: np.ndarray,
        hypotheses_rotations: np.ndarray,
    ) -> None:
        """Add hypotheses by their location and ellipsoid based on the custom
        distance metric.

        Args:
            hypotheses_locations: Array of hypothesis locations
            hypotheses_evidences: Array of evidence values for each hypothesis
            hypotheses_rotations: Array of rotation matrices for each hypothesis
        """
        # Transform hypotheses from learned object frame to world frame
        # The hypotheses are relative to the learned object at [0, 1.5, 0]
        learned_position = np.array([0, 1.5, 0])

        # Move hypotheses to origin
        transformed_locations = hypotheses_locations - learned_position

        # Apply ground truth rotation to both locations and rotation matrices
        ground_truth_rotation = R.from_euler(
            "xyz", self.data_loader.target_rotation, degrees=True
        )
        transformed_locations = ground_truth_rotation.apply(transformed_locations)

        # For rotation matrices, we need to apply the rotation transformation
        # R_world = R_ground_truth @ R_hypothesis
        transformed_rotations = np.zeros_like(hypotheses_rotations)
        ground_truth_matrix = ground_truth_rotation.as_matrix()
        for i in range(len(hypotheses_rotations)):
            transformed_rotations[i] = ground_truth_matrix @ hypotheses_rotations[i]

        # Translate to ground truth position
        transformed_locations = transformed_locations + self.data_loader.target_position

        # Use transformed values for the rest of the method
        hypotheses_locations = transformed_locations
        hypotheses_rotations = transformed_rotations

        self.total_hypotheses_count = len(hypotheses_locations)

        # Use transformed model if available, otherwise use original
        target_model = getattr(self, "transformed_target_model", self.target_model)

        # Check which hypotheses are in/out of object reference frame
        # Returns True for hypotheses OUTSIDE the reference frame
        is_outside = is_hypothesis_in_object_reference_frame(
            hypotheses_locations,
            target_model,
            hypotheses_rotations,
            self.max_nneighbors,
            self.max_match_distance,
        )

        if (
            self.max_hypotheses_to_show is not None
            and len(hypotheses_locations) > self.max_hypotheses_to_show
        ):
            # Sort by evidence values (descending) and take top N
            sorted_indices = np.argsort(hypotheses_evidences)[::-1]
            top_indices = sorted_indices[: self.max_hypotheses_to_show]

            filtered_locations = hypotheses_locations[top_indices]
            filtered_rotations = hypotheses_rotations[top_indices]
            filtered_is_outside = is_outside[top_indices]
        else:
            filtered_locations = hypotheses_locations
            filtered_rotations = hypotheses_rotations
            filtered_is_outside = is_outside

        # Store filtered data for resampling
        self._current_filtered_locations = filtered_locations
        self._current_filtered_rotations = filtered_rotations

        # Separate hypotheses into two groups based on whether they're in/out of object reference frame
        # Blue for hypotheses within object frame (is_outside=False)
        # Pink for hypotheses outside object frame (is_outside=True)
        inside_mask = ~filtered_is_outside
        outside_mask = filtered_is_outside

        inside_locations = filtered_locations[inside_mask]
        outside_locations = filtered_locations[outside_mask]

        # Create separate Points objects for each color group
        self.hypotheses = []

        if len(inside_locations) > 0:
            inside_points = Points(inside_locations, c=TBP_COLORS["blue"])
            inside_points.point_size(8)  # Slightly bigger for inside points
            self.hypotheses.append(inside_points)
            self.plotter.at(self.main_renderer_ix).add(inside_points)

        if len(outside_locations) > 0:
            outside_points = Points(outside_locations, c=TBP_COLORS["pink"])
            outside_points.point_size(5)  # Smaller for outside points
            self.hypotheses.append(outside_points)
            self.plotter.at(self.main_renderer_ix).add(outside_points)

        self._add_hypothesis_ellipsoids(filtered_locations, filtered_rotations)

    def _add_mlh_sphere(self, mlh_location: np.ndarray) -> None:
        """Add MLH sphere visualization.

        Args:
            mlh_location: 3D location of the most likely hypothesis
        """
        # Transform MLH location from learned object frame to world frame
        learned_position = np.array([0, 1.5, 0])

        # Move MLH to origin
        transformed_mlh = mlh_location - learned_position

        # Apply ground truth rotation
        ground_truth_rotation = R.from_euler(
            "xyz", self.data_loader.target_rotation, degrees=True
        )
        transformed_mlh = ground_truth_rotation.apply(transformed_mlh)

        # Translate to ground truth position
        transformed_mlh = transformed_mlh + self.data_loader.target_position

        # Check if MLH is within object reference frame
        # Use transformed model if available, otherwise use original
        target_model = getattr(self, "transformed_target_model", self.target_model)

        # Get MLH rotation from current timestep data (stored as Euler angles)
        mlh_rotation_euler = self.data_loader.all_mlh_rotations[self.current_timestep]

        # Convert Euler angles to rotation matrix
        mlh_rotation_matrix = R.from_euler(
            "xyz", mlh_rotation_euler, degrees=True
        ).as_matrix()

        # Transform MLH rotation to world frame
        mlh_rotation_transformed = (
            ground_truth_rotation.as_matrix() @ mlh_rotation_matrix
        )

        # Check if MLH is in object reference frame
        is_outside = is_hypothesis_in_object_reference_frame(
            transformed_mlh.reshape(1, 3),
            target_model,
            mlh_rotation_transformed.reshape(1, 3, 3),
            self.max_nneighbors,
            self.max_match_distance,
        )[0]

        # Set color based on whether MLH is in/out of object frame
        mlh_color = TBP_COLORS["pink"] if is_outside else TBP_COLORS["blue"]

        # MLH sphere is slightly smaller than regular hypothesis points
        self.mlh_sphere = Sphere(transformed_mlh, r=0.003, c=mlh_color)
        self.plotter.at(self.main_renderer_ix).add(self.mlh_sphere)

    def _add_sensor_spheres(self, timestep: int) -> None:
        """Add spheres to visualize sensor locations."""
        # Add SM_0 sphere if location exists
        if (
            timestep < len(self.data_loader.all_sm0_locations)
            and len(self.data_loader.all_sm0_locations[timestep]) > 0
        ):
            self.sm0_sphere = Sphere(
                self.data_loader.all_sm0_locations[timestep],
                r=0.003,
                c=TBP_COLORS["green"],
                alpha=0.8,
            )
            self.plotter.at(self.main_renderer_ix).add(self.sm0_sphere)

        # Add SM_1 sphere if location exists
        if (
            timestep < len(self.data_loader.all_sm1_locations)
            and len(self.data_loader.all_sm1_locations[timestep]) > 0
        ):
            self.sm1_sphere = Sphere(
                self.data_loader.all_sm1_locations[timestep],
                r=0.005,
                c=TBP_COLORS["yellow"],
                alpha=0.6,
            )
            self.plotter.at(self.main_renderer_ix).add(self.sm1_sphere)

    def _add_sensor_images(self, timestep: int) -> None:
        """Add sensor RGB patch visualizations to separate renderers."""
        # Update SM_0 image if available
        if timestep < len(self.data_loader.all_sm0_rgba):
            rgba_patch = self.data_loader.all_sm0_rgba[timestep]  # (64, 64, 4), 0-255 range
            rgb_patch = rgba_patch[:, :, :3]  # Extract RGB channels

            # Create Image object that fills the renderer
            self.sm0_image = Image(rgb_patch)
            # No need to scale or position - it will fill the renderer
            self.plotter.at(self.sm0_renderer_ix).add(self.sm0_image)

            # Add label
            self.sm0_label = Text2D(
                f"SM_0 (LM step {timestep})", pos="top-center", s=0.8, c="black"
            )
            self.plotter.at(self.sm0_renderer_ix).add(self.sm0_label)

        # Update SM_1 image if available
        if timestep < len(self.data_loader.all_sm1_rgba):
            rgba_patch = self.data_loader.all_sm1_rgba[timestep]
            rgb_patch = rgba_patch[:, :, :3]  # Extract RGB channels

            # Create Image object that fills the renderer
            self.sm1_image = Image(rgb_patch)
            # No need to scale or position - it will fill the renderer
            self.plotter.at(self.sm1_renderer_ix).add(self.sm1_image)

            # Add label
            self.sm1_label = Text2D(
                f"SM_1 (LM step {timestep})", pos="top-center", s=0.8, c="black"
            )
            self.plotter.at(self.sm1_renderer_ix).add(self.sm1_label)

    def _add_hypothesis_ellipsoids(
        self,
        locations: np.ndarray,
        rotations: np.ndarray,
    ) -> None:
        """Add ellipsoids around hypotheses showing their anisotropic reach."""
        if not self.ellipsoids_visible:
            return

        # Note: We'll get curvature per hypothesis location below

        # Use transformed model if available, otherwise use original
        target_model = getattr(self, "transformed_target_model", self.target_model)

        # Check which hypotheses have object points within their ellipsoids
        has_object_points = ~is_hypothesis_in_object_reference_frame(
            locations,
            target_model,
            rotations,
            self.max_nneighbors,
            self.max_match_distance,
        )

        # Sample indices if not already done
        if self.current_sampled_indices is None:
            self.current_sampled_indices = self._sample_ellipsoid_indices(
                len(locations)
            )

        # Use the sampled indices
        show_locations = locations[self.current_sampled_indices]
        show_rotations = rotations[self.current_sampled_indices]
        show_has_objects = has_object_points[self.current_sampled_indices]

        # Clear ellipsoid info for new display
        self.current_ellipsoid_info = []

        for location, rotation, has_object in zip(
            show_locations, show_rotations, show_has_objects
        ):
            # Extract the local coordinate frame from rotation matrix
            # rotation[:, 0] = first principal direction (e.g., PC1)
            # rotation[:, 1] = second principal direction (e.g., PC2)
            # rotation[:, 2] = surface normal
            tangent1 = rotation[:, 0]  # First tangent direction
            tangent2 = rotation[:, 1]  # Second tangent direction
            surface_normal = rotation[:, 2]  # Normal direction

            # Debug: Check magnitudes of the vectors
            tangent1_mag = np.linalg.norm(tangent1)
            tangent2_mag = np.linalg.norm(tangent2)
            normal_mag = np.linalg.norm(surface_normal)

            # Only print debug info for first ellipsoid to avoid spam
            if len(self.current_ellipsoid_info) == 0:
                print(f"\n=== Ellipsoid Debug Info ===")
                print(f"tangent1 magnitude: {tangent1_mag:.6f}")
                print(f"tangent2 magnitude: {tangent2_mag:.6f}")
                print(f"surface_normal magnitude: {normal_mag:.6f}")
                print(
                    f"Is rotation orthonormal? tangent1·tangent2 = {np.dot(tangent1, tangent2):.6f}"
                )

            # Get sensed curvature for current timestep
            sensed_curvature = self.data_loader.sensed_curvatures[self.current_timestep]

            # Calculate ellipsoid axes based on the distance metric
            # Exaggerate the stretch factor to make the effect more visible
            stretch_factor = 1.0 / (np.abs(sensed_curvature) + 0.5)
            semi_axis_tangent = self.max_match_distance
            semi_axis_normal = self.max_match_distance / (1 + stretch_factor)

            # Color based on whether hypothesis has object points in its ellipsoid
            # Blue for hypotheses within object frame, Pink for outside
            color = TBP_COLORS["blue"] if has_object else TBP_COLORS["pink"]

            # Calculate actual axis lengths
            axis1_length = np.linalg.norm(tangent1 * semi_axis_tangent)
            axis2_length = np.linalg.norm(tangent2 * semi_axis_tangent)
            axis3_length = np.linalg.norm(surface_normal * semi_axis_normal)

            # Debug: Print axis information for first ellipsoid
            if len(self.current_ellipsoid_info) == 0:
                print(
                    f"semi_axis_tangent (max_match_distance): {semi_axis_tangent:.6f}"
                )
                print(f"semi_axis_normal: {semi_axis_normal:.6f}")
                print(f"Actual axis lengths:")
                print(f"  axis1: {axis1_length:.6f}")
                print(f"  axis2: {axis2_length:.6f}")
                print(f"  axis3: {axis3_length:.6f}")
                print(
                    f"  Largest axis: axis{np.argmax([axis1_length, axis2_length, axis3_length]) + 1}"
                )
                print(
                    f"  Max radius: {max(axis1_length, axis2_length, axis3_length):.6f}"
                )

            # Store ellipsoid info for display
            ellipsoid_info = {
                "location": location,
                "curvature": sensed_curvature,
                "stretch_factor": stretch_factor,
                "semi_axis_tangent": semi_axis_tangent,
                "semi_axis_normal": semi_axis_normal,
                "has_object": has_object,
                "index": self.current_sampled_indices[len(self.current_ellipsoid_info)],
            }
            self.current_ellipsoid_info.append(ellipsoid_info)

            # Create ellipsoid with axes aligned to the local surface frame
            # axis1 and axis2 lie in the tangent plane
            # axis3 is along the surface normal
            ellipsoid = Ellipsoid(
                pos=location,
                axis1=tangent1 * semi_axis_tangent,
                axis2=tangent2 * semi_axis_tangent,
                axis3=surface_normal * semi_axis_normal,
                c=color,
            )

            # Add sphere at hypothesis location with matching color
            sphere = Sphere(location, r=0.005, c=color)
            self.hypothesis_spheres.append(sphere)
            self.plotter.at(self.main_renderer_ix).add(sphere)

            ellipsoid.alpha(0.15)
            ellipsoid.wireframe(True)
            self.hypothesis_ellipsoids.append(ellipsoid)
            self.plotter.at(self.main_renderer_ix).add(ellipsoid)

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
            filtered_rotations = self.current_hypotheses_rotations[top_indices]
            num_within, num_outside = self._count_points_within_ellipsoids(
                filtered_locations, filtered_rotations
            )
        else:
            # Get all hypothesis rotations
            all_rotations = self.current_hypotheses_rotations
            num_within, num_outside = self._count_points_within_ellipsoids(
                hypotheses_locations, all_rotations
            )

        # Build statistics text
        stats_lines = [
            f"Target: {self.data_loader.target_object_name}",
            f"Object position: [{self.data_loader.target_position[0]:.3f}, "
            f"{self.data_loader.target_position[1]:.3f}, {self.data_loader.target_position[2]:.3f}]",
            f"Object rotation: [{self.data_loader.target_rotation[0]:.3f}, "
            f"{self.data_loader.target_rotation[1]:.3f}, {self.data_loader.target_rotation[2]:.3f}]",
            "",
            f"LM step: {timestep}",
            f"Total hypotheses: {self.total_hypotheses_count}",
        ]

        # Add displayed count info if filtering
        if self.max_hypotheses_to_show is not None:
            displayed_count = min(
                self.max_hypotheses_to_show, len(hypotheses_locations)
            )
            stats_lines.append(
                f"Displayed: {displayed_count} (top {int(self.max_hypotheses_to_show * 100 / self.total_hypotheses_count)}%)"
            )

        stats_lines.extend(
            [
                f"Within object reach: {num_within}",
                f"Outside object reach: {num_outside}",
                f"Evidence range: [{hypotheses_evidences.min():.4f}, "
                f"{hypotheses_evidences.max():.4f}]",
                f"Current MLH object: {mlh_graph_id}",
                f"MLH location: [{mlh_location[0]:.3f}, {mlh_location[1]:.3f}, "
                f"{mlh_location[2]:.3f}]",
            ]
        )

        # Add ellipsoid information if available
        if self.current_ellipsoid_info and len(self.current_ellipsoid_info) > 0:
            stats_lines.append("")
            stats_lines.append("=== Ellipsoid Details ===")
            for i, info in enumerate(self.current_ellipsoid_info):
                stats_lines.append(f"Ellipsoid {i + 1} (Hyp #{info['index']})")
                stats_lines.append(f"  Curvature: {info['curvature']:.4f}")
                stats_lines.append(f"  Stretch factor: {info['stretch_factor']:.4f}")
                stats_lines.append(
                    f"  Tangent semi-axis: {info['semi_axis_tangent']:.6f}"
                )
                stats_lines.append(
                    f"  Normal semi-axis: {info['semi_axis_normal']:.6f}"
                )
                stats_lines.append(
                    f"  Ratio (normal/tangent): {info['semi_axis_normal'] / info['semi_axis_tangent']:.3f}"
                )
                stats_lines.append(f"  Has object points: {info['has_object']}")
                if i < len(self.current_ellipsoid_info) - 1:
                    stats_lines.append("")

        # Add sensor location info
        stats_lines.append("")
        stats_lines.append("=== Sensor Locations ===")
        if (
            timestep < len(self.data_loader.all_sm0_locations)
            and len(self.data_loader.all_sm0_locations[timestep]) > 0
        ):
            sm0_loc = self.data_loader.all_sm0_locations[timestep]
            stats_lines.append(
                f"SM_0 (yellow): ({sm0_loc[0]:.4f}, {sm0_loc[1]:.4f}, {sm0_loc[2]:.4f})"
            )
        else:
            stats_lines.append("SM_0: No data")

        if (
            timestep < len(self.data_loader.all_sm1_locations)
            and len(self.data_loader.all_sm1_locations[timestep]) > 0
        ):
            sm1_loc = self.data_loader.all_sm1_locations[timestep]
            stats_lines.append(
                f"SM_1 (green): ({sm1_loc[0]:.4f}, {sm1_loc[1]:.4f}, {sm1_loc[2]:.4f})"
            )
        else:
            stats_lines.append("SM_1: No data")

        # Add sensed curvature info
        if timestep < len(self.data_loader.sensed_curvatures):
            stats_lines.append(
                f"Sensed curvature: {self.data_loader.sensed_curvatures[timestep]:.4f}"
            )

        stats_text = "\n".join(stats_lines)

        return stats_text


def plot_target_hypotheses(
    exp_path: str,
    model_name: str = "dist_agent_1lm",
    max_match_distance: float = 0.01,
    max_nneighbors: int = 3,
) -> int:
    """Plot target object hypotheses with interactive timestep slider.

    Args:
        exp_path: Path to experiment directory containing detailed_run_stats.json
        model_name: Name of pretrained model to load object from
        max_match_distance: Maximum distance for matching (default: 0.01)
        max_nneighbors: Maximum number of nearest neighbors to consider (default: 3)

    Returns:
        Exit code
    """
    json_path = Path(exp_path) / "detailed_run_stats.json"

    if not json_path.exists():
        logger.error(f"Could not find detailed_run_stats.json at {json_path}")
        return 1

    visualizer = HypothesesOORFVisualizer(
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
        default=0.01,
        help="Maximum distance for matching (default: 0.01).",
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
