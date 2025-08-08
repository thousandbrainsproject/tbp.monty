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
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.spatial.transform import Rotation as R
from vedo import (
    Arrow,
    Cube,
    Ellipsoid,
    Image,
    Lines,
    Plotter,
    Point,
    Points,
    Sphere,
    Text2D,
    settings,
)

from tbp.monty.frameworks.run_env import setup_env
from tbp.monty.frameworks.utils.graph_matching_utils import get_custom_distances

from .data_utils import (
    EpisodeDataLoader,
    ObjectModel,
    get_pretrained_model_path,
)

if TYPE_CHECKING:
    import argparse

    from vedo import Button, Slider2D

logger = logging.getLogger(__name__)

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


def is_hypothesis_inside_object_reference_frame(
    target_model: ObjectModel,
    hypothesis_locations: np.ndarray,
    hypothesis_rotations: np.ndarray,
    max_abs_curvature: float,
    max_nneighbors: int = 3,
    max_match_distance: float = 0.01,
) -> np.ndarray:
    """Check if hypotheses are in the object reference frame.

    Args:
        target_model: ObjectModel containing points and features.
        hypothesis_locations: Array of hypothesis locations (n_hypotheses, 3).
        hypothesis_rotations: Rotation matrices for each hypothesis (n_hypotheses, 3, 3).
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

    nearest_node_locs = target_model.locations[nearest_node_ids]
    surface_normals = hypothesis_rotations[:, :, 2]

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
    return ~is_outside # Return TRUE if INSIDE reference frame


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
    ):
        # ============ CONFIGURATION PARAMETERS ============
        self.json_path = json_path
        self.model_path = model_path
        self.max_match_distance = 0.01
        self.max_nneighbors = 3
        self.current_timestep = 0

        # ============ DATA ============
        self.data_loader = EpisodeDataLoader(self.json_path, self.model_path)
        self.current_hypotheses_locations = None
        self.current_hypotheses_rotations = None
        self.current_mlh_location = None
        self.current_mlh_rotation = None
        self.current_mlh_graph_id = None
        self.current_sm0_location = None
        self.current_sm1_location = None
        self.current_sm0_rgba = None
        self.current_sm1_rgba = None
        self.current_sensed_curvature = None
        self.current_ellipsoid_info = None

        # ============ 3D VEDO OBJECTS ============
        self.target_pointcloud = None
        self.hypotheses_points = []
        self.hypothesis_ellipsoids = []
        self.ellipsoid_centers = []
        self.hypothesis_axes = []
        self.hypothesis_neighbor_points = []  # Track nearest neighbor points
        self.mlh_cube = None
        self.mlh_ellipsoid = None
        self.mlh_axes = []
        self.sm0_sphere = None
        self.sm1_sphere = None
        self.sm0_image = None
        self.sm1_image = None
        self.sm0_label = None
        self.sm1_label = None

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
            dict(bottomleft=(0.02, 0.72), topright=(0.17, 0.92)),  # SM_0
            dict(bottomleft=(0.19, 0.72), topright=(0.34, 0.92)),  # SM_1
        ]

        self.plotter = Plotter(
            shape=custom_shape,
            size=(1400, 1000),
            sharecam=False,
            title=f"Hypotheses Out of Reference Frame for {self.data_loader.target_object_name.title()}",
        )

        self.update_visualization(timestep=0)

        # Timestep slider
        self.slider = self.plotter.at(self.main_renderer_ix).add_slider(
            self.slider_callback,
            xmin=0,
            xmax=self.data_loader.num_lm_steps - 1,
            value=0,
            pos=[(0.25, 0.05), (0.75, 0.05)],
            title="LM step",
        )
        self.resample_button = self.plotter.at(self.main_renderer_ix).add_button(
            self.resample_ellipsoids_callback,
            pos=(0.85, 0.15),
            states=["Select Different Ellipsoids"],
            font="Calco",
            size=20,
        )

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

        self.is_inside_reference_frame = is_hypothesis_inside_object_reference_frame(
            self.object_model,
            self.current_hypotheses_locations,
            self.current_hypotheses_rotations,
            self.current_sensed_curvatures,
            self.max_nneighbors,
            self.max_match_distance,
        )
        self._add_hypotheses_points()
        
        hyp_location, hyp_rotation, _, _ = self._add_ellipsoid()
        self._add_nearest_neighbors(hyp_location, hyp_rotation)
        self._add_axes_arrows(hyp_location, hyp_rotation)

        self.is_mlh_inside_reference_frame = is_hypothesis_inside_object_reference_frame(
            self.object_model,
            self.current_mlh_location.reshape(1, 3),
            self.current_mlh_rotation.reshape(1, 3, 3),
            self.current_sensed_curvatures,
            self.max_nneighbors,
            self.max_match_distance,
        )[0]
        self._add_mlh_cube(self.current_mlh_location)
        self._add_nearest_neighbors(self.current_mlh_location, self.current_mlh_rotation)
        self._add_axes_arrows(
            self.current_mlh_location,
            self.current_mlh_rotation,
        )

        self._add_sensor_spheres(timestep)
        self._add_sensor_images(timestep)

        stats_text = self._create_statistics_text(
            timestep,
            self.current_hypotheses_locations,
            self.current_mlh_location,
        )
        self.stats_text = Text2D(stats_text, pos="top-right", s=0.7)
        self.plotter.at(self.main_renderer_ix).add(self.stats_text)

    def slider_callback(self, widget: Slider2D, _event: str) -> None:
        """Respond to slider step by updating the visualization."""
        timestep = round(widget.GetRepresentation().GetValue())
        if timestep != self.current_timestep:
            self.update_visualization(timestep)
            self.plotter.render()

    def resample_ellipsoids_callback(self, _widget: Button, _event: str) -> None:
        """Resample the ellipsoids to show a different random selection."""
        self.current_ellipsoid_info = None

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

        hyp_location, hyp_rotation, _, _ = self._add_ellipsoid()
        self._add_nearest_neighbors(hyp_location, hyp_rotation)
        self._add_axes_arrows(hyp_location, hyp_rotation)
        self._add_axes_arrows(self.current_mlh_location, self.current_mlh_rotation)

        # Update statistics text with new ellipsoid and NN info
        if hasattr(self, "stats_text"):
            self.plotter.remove(self.stats_text)

        stats_text = self._create_statistics_text(
            self.current_timestep,
            self.current_hypotheses_locations,
            self.current_mlh_location,
        )
        self.stats_text = Text2D(stats_text, pos="top-right", s=0.7)
        self.plotter.at(self.main_renderer_ix).add(self.stats_text)

        self.plotter.at(self.sm0_renderer_ix).render()
        self.plotter.at(self.sm1_renderer_ix).render()
        self.plotter.at(self.main_renderer_ix).render()

    def _initialize_timestep_data(self, timestep: int) -> tuple:
        """Get data for a specific timestep.

        Args:
            timestep: The timestep to retrieve data for
        """
        self.current_hypotheses_locations = self.data_loader.all_hypotheses_locations[
            timestep
        ]
        self.current_hypotheses_rotations = self.data_loader.all_hypotheses_rotations[
            timestep
        ]
        self.current_mlh_location = self.data_loader.all_mlh_locations[timestep]
        self.current_mlh_rotation = self.data_loader.all_mlh_rotations[timestep]
        self.current_mlh_graph_id = self.data_loader.all_mlh_graph_ids[timestep]
        self.current_sm0_location = self.data_loader.all_sm0_locations[timestep]
        self.current_sm1_location = self.data_loader.all_sm1_locations[timestep]
        self.current_sm0_rgba = self.data_loader.all_sm0_rgba[timestep]
        self.current_sm1_rgba = self.data_loader.all_sm1_rgba[timestep]
        self.current_sensed_curvatures = self.data_loader.sensed_curvatures[timestep]

    def _cleanup_previous_visualizations(self) -> None:
        """Remove previous visualization objects from the plotter."""
        # Clean up main renderer objects
        if self.hypotheses_points is not None:
            if isinstance(self.hypotheses_points, list):
                for pts in self.hypotheses_points:
                    self.plotter.at(self.main_renderer_ix).remove(pts)
            else:
                self.plotter.at(self.main_renderer_ix).remove(self.hypotheses_points)
        if self.mlh_cube is not None:
            self.plotter.at(self.main_renderer_ix).remove(self.mlh_cube)
            self.mlh_cube = None
        if self.mlh_ellipsoid is not None:
            self.plotter.at(self.main_renderer_ix).remove(self.mlh_ellipsoid)
            self.mlh_ellipsoid = None
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

    def _initialize_target_visualization(self) -> None:
        """Initialize object model and convex hull visualization."""
        self.object_model = self.data_loader.object_model

        self.target_pointcloud = Points(self.object_model.locations, c="gray")
        self.target_pointcloud.point_size(10)
        self.plotter.at(self.main_renderer_ix).add(self.target_pointcloud)

        self.plotter.at(self.main_renderer_ix).add(
            Text2D(
                f"{self.data_loader.target_name.title()}",
                pos="top-left",
                s=1,
            )
        )

    def _add_hypotheses_points(
        self,
    ) -> None:
        """Plot hypotheses in world frame colored by OORF status.

        Args:
            hypotheses_locations: Array of hypothesis locations in world frame
            hypotheses_rotations: Array of rotation matrices for each hypothesis in world frame
        """
        inside_hypotheses = self.current_hypotheses_locations[self.is_inside_reference_frame]
        outside_hypotheses = self.current_hypotheses_locations[~self.is_inside_reference_frame]

        if len(inside_hypotheses) > 0:
            inside_points = Points(inside_hypotheses, c=TBP_COLORS["blue"])
            inside_points.point_size(8)
            self.hypotheses_points.append(inside_points)
            self.plotter.at(self.main_renderer_ix).add(inside_points)

        if len(outside_hypotheses) > 0:
            outside_points = Points(outside_hypotheses, c=TBP_COLORS["pink"])
            outside_points.point_size(5)
            self.hypotheses_points.append(outside_points)
            self.plotter.at(self.main_renderer_ix).add(outside_points)

    def _add_mlh_cube(self, mlh_location: np.ndarray) -> None:
        """Add MLH sphere visualization.

        Args:
            mlh_location: 3D location of the most likely hypothesis (already in world frame)
        """
        mlh_color = TBP_COLORS["blue"] if self.is_mlh_inside_reference_frame else TBP_COLORS["pink"]

        # Add smaller cube for MLH
        self.mlh_cube = Cube(mlh_location, side=0.005, c=mlh_color, alpha=0.6)
        self.plotter.at(self.main_renderer_ix).add(self.mlh_cube)

        # Add ellipsoid around MLH (always present)
        self.mlh_ellipsoid = Ellipsoid(
            pos=mlh_location,
            axis1=[0.012, 0, 0],  # Semi-axis along X
            axis2=[0, 0.010, 0],  # Semi-axis along Y
            axis3=[0, 0, 0.008],  # Semi-axis along Z
            c=mlh_color,
            alpha=0.3
        )
        self.plotter.at(self.main_renderer_ix).add(self.mlh_ellipsoid)

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

        self.sm0_label = Text2D("SM_0", pos="top-center", s=0.8, c="black")
        self.plotter.at(self.sm0_renderer_ix).add(self.sm0_label)

        rgba_patch = self.current_sm1_rgba
        rgb_patch = rgba_patch[:, :, :3]

        self.sm1_image = Image(rgb_patch)
        self.plotter.at(self.sm1_renderer_ix).add(self.sm1_image)

        self.sm1_label = Text2D("SM_1", pos="top-center", s=0.8, c="black")
        self.plotter.at(self.sm1_renderer_ix).add(self.sm1_label)

    def _add_ellipsoid(self) -> None:
        """Add ellipsoids around a randomly selected hypothesis."""
        idx = np.random.choice(len(self.current_hypotheses_locations), 1)[0]
        hyp_location = self.current_hypotheses_locations[idx]
        hyp_rotation = self.current_hypotheses_rotations[idx]
        hyp_is_inside_reference_frame = self.is_inside_reference_frame[idx]

        tangent1 = hyp_rotation[:, 0]  # First tangent direction, PC1
        tangent2 = hyp_rotation[:, 1]  # Second tangent direction, PC2
        surface_normal = hyp_rotation[:, 2]  # Surface normal

        stretch_factor = 1.0 / (np.abs(self.current_sensed_curvatures) + 0.5)
        semi_axis_tangent = self.max_match_distance
        semi_axis_normal = self.max_match_distance / (1 + stretch_factor)

        color = TBP_COLORS["blue"] if hyp_is_inside_reference_frame else TBP_COLORS["pink"]

        ellipsoid = Ellipsoid(
            pos=hyp_location,
            axis1=tangent1 * semi_axis_tangent,
            axis2=tangent2 * semi_axis_tangent,
            axis3=surface_normal * semi_axis_normal,
            c=color,
        )

        self.current_ellipsoid_info = {
            "location": hyp_location,
            "rotation": hyp_rotation,
            "curvature": self.current_sensed_curvatures,
            "stretch_factor": stretch_factor,
            "semi_axis_tangent": semi_axis_tangent,
            "semi_axis_normal": semi_axis_normal,
            "is_inside_reference_frame": hyp_is_inside_reference_frame,
            "index": idx,
        }

        hyp_point = Point(hyp_location, c="darkblue")
        hyp_point.point_size(25)
        self.ellipsoid_centers.append(hyp_point)
        self.plotter.at(self.main_renderer_ix).add(hyp_point)

        ellipsoid.alpha(0.15)
        ellipsoid.wireframe(True)
        self.hypothesis_ellipsoids.append(ellipsoid)
        self.plotter.at(self.main_renderer_ix).add(ellipsoid)

        return hyp_location, hyp_rotation, hyp_is_inside_reference_frame, idx

    def _add_axes_arrows(
        self,
        location: np.ndarray,
        rotation: np.ndarray,
    ) -> None:
        """Add arrows showing tangent and normal directions for a hypothesis."""
        arrow_length = 0.02

        tangent1 = rotation[:, 0]
        tangent2 = rotation[:, 1]
        surface_normal = rotation[:, 2]

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

    def _add_nearest_neighbors(
        self,
        location: np.ndarray,
        rotation: np.ndarray,
    ) -> None:
        """Add visualization of nearest neighbors for a hypothesis."""
        # Find nearest neighbors
        distances, nearest_node_ids = self.object_model.kd_tree.query(
            location.reshape(1, -1),
            k=self.max_nneighbors,
            p=2,
            workers=1,
        )

        if self.max_nneighbors == 1:
            nearest_node_ids = np.expand_dims(nearest_node_ids, axis=0)
            distances = np.expand_dims(distances, axis=0)
        else:
            nearest_node_ids = nearest_node_ids[0]
            distances = distances[0]

        nearest_node_locs = self.object_model.locations[nearest_node_ids]
        surface_normal = rotation[:, 2]

        # Color the nearest node neighbors in yellow points
        nearest_node_points = Points(nearest_node_locs, c=TBP_COLORS["yellow"])
        nearest_node_points.point_size(15)
        self.hypothesis_neighbor_points.append(nearest_node_points)
        self.plotter.at(self.main_renderer_ix).add(nearest_node_points)

        # Calculate custom distances for visualization
        custom_dists = get_custom_distances(
            nearest_node_locs.reshape(1, self.max_nneighbors, 3),
            location.reshape(1, 3),
            surface_normal.reshape(1, 3),
            self.current_sensed_curvatures,
        )[0]

        # Store for statistics
        self.hypothesis_nn_info = {
            "locations": nearest_node_locs,
            "euclidean_distances": distances,
            "custom_distances": custom_dists,
        }

        # # Add lines and spheres for each neighbor
        # for neighbor_loc, custom_dist in zip(nearest_node_locs, custom_dists):
        #     # Color based on whether it's within match distance
        #     is_within = custom_dist <= self.max_match_distance
        #     line_color = TBP_COLORS["blue"] if is_within else TBP_COLORS["pink"]

        #     # # Line to neighbor
        #     # line = Lines(
        #     #     [location, neighbor_loc],
        #     #     c=line_color,
        #     #     lw=1 if is_within else 0.5,
        #     # )
        #     # line.alpha(0.5 if is_within else 0.3)
        #     # self.hypothesis_neighbor_lines.append(line)
        #     # self.plotter.at(self.main_renderer_ix).add(line)

        #     # Small sphere at neighbor
        #     sphere = Sphere(
        #         neighbor_loc,
        #         r=0.002,
        #         c=line_color,
        #     )
        #     sphere.alpha(0.6)
        #     self.hypothesis_neighbor_spheres.append(sphere)
        #     self.plotter.at(self.main_renderer_ix).add(sphere)

    def _create_statistics_text(
        self,
        timestep: int,
        hypotheses_locations: np.ndarray,
        mlh_location: np.ndarray,
    ) -> str:
        """Create statistics text for display.

        Args:
            timestep: Current timestep
            hypotheses_locations: Array of hypothesis locations
            mlh_location: Location of most likely hypothesis

        Returns:
            Formatted statistics text
        """
        num_hypotheses_within_reach = sum(self.is_inside_reference_frame)
        num_hypotheses_outside_reach = sum(~self.is_inside_reference_frame)

        target_stats = [
            f"Target: {self.data_loader.target_name}",
            f"Object position: {self.data_loader.ground_truth_position}",
            f"Object rotation: {self.data_loader.ground_truth_rotation}",
        ]

        hypotheses_stats = [
            f"LM step: {timestep}",
            f"Total hypotheses at step: {len(self.current_hypotheses_locations)}",
            f"Within object reach: {num_hypotheses_within_reach}",
            f"Outside object reach: {num_hypotheses_outside_reach}",
            f"Current MLH object: {self.current_mlh_graph_id}",
            f"MLH location: {mlh_location}",
        ]

        ellipsoid_stats = [
            "",
            "=== Selected Hypothesis (Ellipsoid) ===",
            f"Hypothesis #{self.current_ellipsoid_info['index']}:",
            f"  Max Abs Curvature: {self.current_ellipsoid_info['curvature']:.4f}",
            f"  Stretch factor: {self.current_ellipsoid_info['stretch_factor']:.4f}",
            f"  Tangent semi-axis: {self.current_ellipsoid_info['semi_axis_tangent']:.6f}",
            f"  Normal semi-axis: {self.current_ellipsoid_info['semi_axis_normal']:.6f}",
            f"  Ratio (normal/tangent): {self.current_ellipsoid_info['semi_axis_normal'] / self.current_ellipsoid_info['semi_axis_tangent']:.3f}",
            f"  Is inside reference frame: {self.current_ellipsoid_info['is_inside_reference_frame']}",
        ]

        # Add selected hypothesis nearest neighbor stats
        if hasattr(self, "hypothesis_nn_info"):
            ellipsoid_stats.extend(
                [
                    "  Nearest Neighbors:",
                ]
            )
            for i, (dist, custom_dist) in enumerate(
                zip(
                    self.hypothesis_nn_info["euclidean_distances"],
                    self.hypothesis_nn_info["custom_distances"],
                )
            ):
                within = custom_dist <= self.max_match_distance
                ellipsoid_stats.append(
                    f"    NN{i + 1}: Eucl={dist:.4f}, Custom={custom_dist:.4f} {'✓' if within else '✗'}"
                )

        # Add MLH nearest neighbor stats
        mlh_nn_stats = []
        if hasattr(self, "mlh_nearest_neighbors_info"):
            mlh_nn_stats = [
                "",
                "=== MLH Nearest Neighbors ===",
            ]
            for i, (dist, custom_dist) in enumerate(
                zip(
                    self.mlh_nearest_neighbors_info["euclidean_distances"],
                    self.mlh_nearest_neighbors_info["custom_distances"],
                )
            ):
                within = custom_dist <= self.max_match_distance
                mlh_nn_stats.append(
                    f"  NN{i + 1}: Eucl={dist:.4f}, Custom={custom_dist:.4f} {'✓' if within else '✗'}"
                )

        sensor_stats = [
            "",
            "=== Sensor Locations ===",
            f"SM_0 (green): ({self.current_sm0_location[0]:.4f}, {self.current_sm0_location[1]:.4f}, {self.current_sm0_location[2]:.4f})",
            f"SM_1 (yellow): ({self.current_sm1_location[0]:.4f}, {self.current_sm1_location[1]:.4f}, {self.current_sm1_location[2]:.4f})",
        ]

        arrow_legend = [
            "",
            "=== Arrow Legend ===",
            "Purple: 1st Tangent (PC1)",
            "Green: 2nd Tangent (PC2)",
            "Yellow: Surface Normal",
        ]

        stats_text = "\n".join(
            target_stats
            + hypotheses_stats
            + ellipsoid_stats
            + mlh_nn_stats
            + sensor_stats
            + arrow_legend
        )

        return stats_text


def plot_target_hypotheses(
    exp_path: Path,
    model_type: Literal["dist", "surf"] = "dist",
) -> int:
    """Plot target object hypotheses with interactive timestep slider.

    Args:
        exp_path: Path to experiment directory containing detailed_run_stats.json
        model_type: Type of pretrained model to load object from
        max_match_distance: Maximum distance for matching (default: 0.01)
        max_nneighbors: Maximum number of nearest neighbors to consider (default: 3)

    Returns:
        Exit code
    """
    json_path = Path(exp_path) / "detailed_run_stats.json"

    if not json_path.exists():
        logger.error(f"Could not find detailed_run_stats.json at {json_path}")
        return 1

    model_path = get_pretrained_model_path(model_type)

    visualizer = HypothesesOORFVisualizer(json_path, model_path)
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
        help="Interactive tool to visualize which hypotheses are out of object reference frame.",
        parents=[parent_parser] if parent_parser else [],
    )
    parser.add_argument(
        "experiment_log_dir",
        help="The directory containing the detailed_run_stats.json file.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="dist",
        help="Type of pretrained model to load target object from.",
        choices=["dist", "surf"],
    )
    parser.set_defaults(
        func=lambda args: sys.exit(
            plot_target_hypotheses(
                args.experiment_log_dir,
                args.model_type,
            )
        )
    )
