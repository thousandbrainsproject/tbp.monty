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


def compute_reference_frame_analysis(
    object_model: ObjectModelForVisualization,
    hypothesis_locations: np.ndarray,
    hypothesis_rotations: np.ndarray,
    sensed_orientation: np.ndarray,
    max_abs_curvature: float,
    max_nneighbors: int = 3,
    max_match_distance: float = 0.01,
) -> dict:
    """Compute reference frame analysis for hypotheses.

    Args:
        object_model: ObjectModel containing points and features.
        hypothesis_locations: Array of hypothesis locations (n_hypotheses, 3).
        hypothesis_rotations: Rotation matrix for each hypothesis (n_hypotheses, 3, 3).
            This is the hypothesized rotation of the object.
        sensed_orientation: Sensed pose vectors (3, 3) to be transformed by hypotheses.
            At each timestep, the Sensor Module extracts pose_vectors from RGBD image,
            where pose_vectors represent row vectors of [surface_normal, pc1, pc2].
        max_abs_curvature: Maximum absolute value of principal_curvature_logs.
        max_nneighbors: Maximum number of nearest neighbors to consider (default: 3).
        max_match_distance: Maximum distance for matching (default: 0.01).

    Returns:
        Dictionary containing:
            - is_inside_reference_frame: Array of booleans indicating if hypothesis is
                inside the object's reference frame.
            - nearest_node_locs: Locations of nearest neighbors for each hypothesis
                just for visualizing nearest neighbor points.
    """
    _, nearest_node_ids = object_model.kd_tree.query(
        hypothesis_locations,
        k=max_nneighbors,
        p=2,
        workers=1,
    )
    nearest_node_locs = object_model.object_points_wrt_world[nearest_node_ids]

    # Transform sensed orientation by all hypothesized object rotations
    features = {"pose_vectors": sensed_orientation}
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
        "nearest_node_locs": nearest_node_locs,
    }


class HypothesesOORFVisualizer:
    """Interactive visualizer for hypotheses that are out of object's reference frame.

    Args:
        json_path: Path to the detailed_run_stats.json file containing episode data.
        model_path: Path to the pretrained model to load object from.
        episode_id: Episode ID to visualize.
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
        self.object_pointcloud = None
        self.hypotheses_points = []
        self.hypothesis_ellipsoid = None
        self.ellipsoid_center_point = None
        self.random_hypothesis_neighbor_points = []
        self.heh_neighbor_points = []
        self.random_hypothesis_pose_arrows = []
        self.heh_pose_arrows = []
        self.highest_evidence_ellipsoid = None
        self.highest_evidence_cube = None
        self.sm0_sphere = None
        self.sm1_sphere = None
        self.sm0_image = None
        self.sm1_image = None
        self.object_pose_vector_arrows = []
        self.pose_vectors_visible = False
        self.current_hypothesis_index = 0

        # ============ UI ============
        self.plotter = None
        self.timestep_slider = None
        self.summary_text = None

        # ============ RENDERERS ============
        self.main_renderer_ix = 0
        self.sm0_renderer_ix = 1
        self.sm1_renderer_ix = 2

        self.data_loader.load_episode_data()

    def create_interactive_visualization(self) -> None:
        """Create interactive visualization with slider for timestep navigation."""
        # Create plotter main view and 2 small overlays for sensor images
        custom_view_areas = [
            dict(bottomleft=(0.0, 0.0), topright=(1.0, 1.0)),  # Main view (full window)
            dict(bottomleft=(0.73, 0.79), topright=(0.86, 0.99)),  # SM_0 (top-right)
            dict(bottomleft=(0.86, 0.79), topright=(0.99, 0.99)),  # SM_1 (top-right)
        ]

        self.plotter = Plotter(
            shape=custom_view_areas,
            size=(1400, 1000),
            sharecam=False,
            title=f"Hypotheses Out of Reference Frame",
        )
        # Add static elements to main renderer
        self._add_object_pointcloud()
        self._add_legend()

        # Add timestep dependent elements to main renderer
        self.update_visualization(timestep=0)

        # Add widgets to main renderer
        self.timestep_slider = self.plotter.at(self.main_renderer_ix).add_slider(
            self.timestep_slider_callback,
            xmin=0,
            xmax=self.data_loader.num_lm_steps - 1,
            value=0,
            pos=[(0.2, 0.05), (0.8, 0.05)],
            title="LM step",
            show_value=True,
        )
        self.resample_ellipsoid_button = self.plotter.at(
            self.main_renderer_ix
        ).add_button(
            self.resample_ellipsoid_callback,
            pos=(0.40, 0.14),
            states=[" Resample Hypothesis "],
            size=20,
            font="Calco",
        )
        self.pose_vectors_button = self.plotter.at(self.main_renderer_ix).add_button(
            self.toggle_pose_vectors_callback,
            pos=(0.6, 0.14),
            states=[" Show Pose Vectors ", " Hide Pose Vectors "],
            size=20,
            font="Calco",
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
            camera=dict(pos=(0.5, 1.5, 0.5), focal_point=(0, 1.5, 0)),
        )
        self.plotter.show(interactive=True)

    def update_visualization(self, timestep: int) -> None:
        """Update visualization for given timestep."""
        self._clear_visualizations()

        self.current_timestep = timestep
        self._initialize_timestep_data(self.current_timestep)

        hypotheses_oorf_info = compute_reference_frame_analysis(
            self.object_model,
            self.current_hypotheses_locations,
            self.current_hypotheses_rotations,
            self.current_sensed_orientation,
            self.current_sensed_curvature,
            self.max_nneighbors,
            self.max_match_distance,
        )
        self.hypotheses_inside_reference_frame = hypotheses_oorf_info[
            "is_inside_reference_frame"
        ]
        self.hypotheses_nearest_node_locs = hypotheses_oorf_info["nearest_node_locs"]

        self._add_hypotheses_points()
        self._add_random_hypothesis_visualization()
        self._add_highest_evidence_hypothesis_visualization()
        self._add_sensor_location_spheres()
        self._add_sensor_images()
        self._add_summary_text()

    def timestep_slider_callback(self, widget: Slider2D, _event: str) -> None:
        """Respond to slider step by updating the visualization."""
        timestep = round(widget.GetRepresentation().GetValue())
        if timestep != self.current_timestep:
            self.update_visualization(timestep)
            self.plotter.render()

    def resample_ellipsoid_callback(self, _widget: Button, _event: str) -> None:
        """Resample hypothesis (and its ellipsoid) to show a different example."""
        self.plotter.remove(self.hypothesis_ellipsoid)
        self.plotter.remove(self.ellipsoid_center_point)
        self.plotter.remove(self.random_hypothesis_neighbor_points)
        self.plotter.remove(self.random_hypothesis_pose_arrows)

        self.random_hypothesis_neighbor_points.clear()
        self.random_hypothesis_pose_arrows.clear()

        self._add_random_hypothesis_visualization()
        self.plotter.at(self.main_renderer_ix).render()

    def toggle_pose_vectors_callback(self, _widget: Button, _event: str) -> None:
        """Toggle visibility of pose vector arrows."""
        self.pose_vectors_visible = not self.pose_vectors_visible
        self.pose_vectors_button.switch()

        if self.pose_vectors_visible:
            self._add_object_surface_normal_arrows()
        else:
            self.plotter.at(self.main_renderer_ix).remove(
                self.object_pose_vector_arrows
            )

        self.plotter.at(self.main_renderer_ix).render()

    def _initialize_timestep_data(self, timestep: int) -> tuple:
        """Initialize data for a specific timestep.

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
        self.current_highest_evidence_index = self.data_loader.highest_evidence_indices[
            timestep
        ]
        self.current_mlh_graph_id = self.data_loader.all_mlh_graph_ids[timestep]
        self.current_sm0_location = self.data_loader.all_sm0_locations[timestep]
        self.current_sm1_location = self.data_loader.all_sm1_locations[timestep]
        self.current_sm0_rgba = self.data_loader.all_sm0_rgba[timestep]
        self.current_sm1_rgba = self.data_loader.all_sm1_rgba[timestep]
        self.current_sensed_curvature = self.data_loader.max_abs_curvatures[timestep]
        self.current_sensed_orientation = self.data_loader.sensed_orientations[timestep]

    def _clear_visualizations(self) -> None:
        """Remove visualization objects from the plotter.

        Note that it does not remove static elements like object
        pointcloud, object pose vector arrows, and legend.
        """
        self.plotter.at(self.main_renderer_ix).remove(self.hypotheses_points)
        self.plotter.at(self.main_renderer_ix).remove(self.summary_text)
        self.plotter.at(self.main_renderer_ix).remove(self.hypothesis_ellipsoid)
        self.plotter.at(self.main_renderer_ix).remove(self.ellipsoid_center_point)
        self.plotter.at(self.main_renderer_ix).remove(
            self.random_hypothesis_pose_arrows
        )
        self.plotter.at(self.main_renderer_ix).remove(self.heh_pose_arrows)
        self.plotter.at(self.main_renderer_ix).remove(
            self.random_hypothesis_neighbor_points
        )
        self.plotter.at(self.main_renderer_ix).remove(self.heh_neighbor_points)
        self.plotter.at(self.main_renderer_ix).remove(self.highest_evidence_cube)
        self.plotter.at(self.main_renderer_ix).remove(self.highest_evidence_ellipsoid)
        self.plotter.at(self.main_renderer_ix).remove(self.sm0_sphere)
        self.plotter.at(self.main_renderer_ix).remove(self.sm1_sphere)
        self.plotter.at(self.sm0_renderer_ix).remove(self.sm0_image)
        self.plotter.at(self.sm1_renderer_ix).remove(self.sm1_image)

        self.hypotheses_points.clear()
        self.random_hypothesis_neighbor_points.clear()
        self.heh_neighbor_points.clear()
        self.random_hypothesis_pose_arrows.clear()
        self.heh_pose_arrows.clear()

    def _add_object_pointcloud(self) -> None:
        """Add object pointcloud to visualization."""
        self.object_model = self.data_loader.object_model

        self.object_pointcloud = Points(
            self.object_model.object_points_wrt_world,
            c="gray",
        )
        self.object_pointcloud.point_size(8)
        self.plotter.at(self.main_renderer_ix).add(self.object_pointcloud)

    def _add_hypotheses_points(self) -> None:
        """Add hypotheses' locations in world frame colored by OORF status.

        The points are colored blue if they are inside the object's reference frame,
        and pink if they are outside.
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

    def _add_random_hypothesis_visualization(self) -> None:
        """Select a random hypothesis and add its visualization elements.

        Adds ellipsoid, center point, nearest neighbors, and sensed pose vector arrows
        for a randomly selected hypothesis.
        """
        idx = np.random.choice(len(self.current_hypotheses_locations), 1)[0]
        self.current_hypothesis_index = idx
        self._add_ellipsoid(
            self.current_hypotheses_locations[idx],
            self.current_hypotheses_rotations[idx],
            self.hypotheses_inside_reference_frame[idx],
            is_heh=False,
        )
        self._add_hypothesis_center_point(self.current_hypotheses_locations[idx])
        self._add_nearest_neighbor_points(
            self.hypotheses_nearest_node_locs[idx], is_heh=False
        )
        self._add_sensed_pose_vector_arrows(
            self.current_hypotheses_locations[idx],
            self.current_hypotheses_rotations[idx],
            is_heh=False,
        )

    def _add_highest_evidence_hypothesis_visualization(self) -> None:
        """Add visualization elements for the highest evidence hypothesis (HEH).

        Adds ellipsoid, center cube, nearest neighbors, and sensed pose vector arrows
        for the highest evidence hypothesis (HEH).
        """
        is_inside_reference_frame = self.hypotheses_inside_reference_frame[
            self.current_highest_evidence_index
        ]
        highest_evidence_nearest_node_locs = self.hypotheses_nearest_node_locs[
            self.current_highest_evidence_index
        ]
        self._add_ellipsoid(
            self.current_highest_evidence_location,
            self.current_highest_evidence_rotation,
            is_inside_reference_frame,
            is_heh=True,
        )
        self._add_highest_hypothesis_center_cube(
            self.current_highest_evidence_location, is_inside_reference_frame
        )
        self._add_nearest_neighbor_points(
            highest_evidence_nearest_node_locs, is_heh=True
        )
        self._add_sensed_pose_vector_arrows(
            self.current_highest_evidence_location,
            self.current_highest_evidence_rotation,
            is_heh=True,
        )

    def _add_ellipsoid(
        self,
        location: np.ndarray,
        hypothesis_rotation: np.ndarray,
        is_inside_reference_frame: bool,
        is_heh: bool = False,
    ) -> None:
        """Add ellipsoid around hypothesis based on sensed orientation.

        The ellipsoid represents the region where a hypothesis is considered valid
        based on the custom distance metric that incorporates surface curvature.

        Derivation of the normal axis length:
        Let d be the distance in the direction of the surface normal.
        The custom distance metric is:
            custom_nearest_node_dists = d + d * stretch_factor
        where stretch_factor = 1 / (|search_curvature| + 0.5)

        For a point to be within the ellipsoid:
            custom_nearest_node_dists <= max_match_distance
            d * (1 + stretch_factor) <= max_match_distance
            d <= max_match_distance / (1 + stretch_factor)

        Therefore, the normal axis length is: max_match_distance / (1 + stretch_factor)

        This creates an ellipsoid that accounts for the curvature-dependent stretching
        of the distance metric in the normal direction.
        """
        surface_normal, tangent1, tangent2 = (
            self._transform_sensed_orientation_by_hypothesis(hypothesis_rotation)
        )

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

        if is_heh:
            self.highest_evidence_ellipsoid = ellipsoid
        else:
            self.hypothesis_ellipsoid = ellipsoid

        self.plotter.at(self.main_renderer_ix).add(ellipsoid)

    def _add_hypothesis_center_point(self, location: np.ndarray) -> None:
        """Add a black point at the hypothesis center.

        This point is a randomly selected hypothesis that goes along with the ellipsoid,
        and changes when the hypothesis is resampled within a step.
        """
        hyp_point = Point(location, c="black")
        hyp_point.point_size(25)
        self.ellipsoid_center_point = hyp_point
        self.plotter.at(self.main_renderer_ix).add(hyp_point)

    def _add_highest_hypothesis_center_cube(
        self, location: np.ndarray, is_inside_reference_frame: bool
    ) -> None:
        """Add a cube at the location of the highest evidence hypothesis (HEH).

        The highest evidence hypothesis is the one with the highest evidence score
        for the subset of hypotheses whose target is the same as the object. It can be
        different from the MLH, which is the highest evidence hypothesis across entire
        set of objects.

        The cube denoting this location is fixed within a step.
        """
        self.highest_evidence_cube = Cube(location, side=0.003, c="black", alpha=0.6)
        self.plotter.at(self.main_renderer_ix).add(self.highest_evidence_cube)

    def _add_nearest_neighbor_points(
        self, nearest_node_locs: np.ndarray, is_heh: bool = False
    ) -> None:
        """Add yellow points indicating the nearest neighbors of a hypothesis.

        The nearest neighbors are points within the object model. It is used for
        debugging to verify that a hypothesis is considered within an object's
        reference frame if at least one of its nearest neighbors is within the
        ellipsoid of the hypothesis.
        """
        nearest_node_points = Points(
            nearest_node_locs.squeeze(), c=TBP_COLORS["yellow"]
        )
        nearest_node_points.point_size(15)
        if is_heh:
            self.heh_neighbor_points.append(nearest_node_points)
        else:
            self.random_hypothesis_neighbor_points.append(nearest_node_points)

        self.plotter.at(self.main_renderer_ix).add(nearest_node_points)

    def _add_sensed_pose_vector_arrows(
        self,
        location: np.ndarray,
        hypothesis_rotation: np.ndarray,
        is_heh: bool = False,
    ) -> None:
        """Add arrows showing transformed sensed tangent and normal directions."""
        arrow_length = 0.02

        surface_normal, tangent1, tangent2 = (
            self._transform_sensed_orientation_by_hypothesis(hypothesis_rotation)
        )

        arrow1 = Arrow(
            location,
            location + tangent1 * arrow_length,
            c=TBP_COLORS["purple"],
        )
        arrow1.alpha(0.7)

        arrow2 = Arrow(
            location,
            location + tangent2 * arrow_length,
            c=TBP_COLORS["green"],
        )
        arrow2.alpha(0.7)

        arrow3 = Arrow(
            location,
            location + surface_normal * arrow_length,
            c=TBP_COLORS["yellow"],
        )
        arrow3.alpha(0.9)

        if is_heh:
            self.heh_pose_arrows.extend([arrow1, arrow2, arrow3])
        else:
            self.random_hypothesis_pose_arrows.extend([arrow1, arrow2, arrow3])

        self.plotter.at(self.main_renderer_ix).add([arrow1, arrow2, arrow3])

    def _add_object_surface_normal_arrows(self) -> None:
        """Add arrows showing surface normals from object_model's pose vectors.

        These are the ground truth surface normals useful for debugging. Note that
        the object_model's pose vectors is a 3x3 matrix of stacked
        [surface_normal, pc1, pc2] row vectors for each point.
        """
        arrow_length = 0.01

        # Sample because showing all surface normals is too cluttered for visualization
        sample_indices = np.arange(0, len(self.object_model.object_points_wrt_world), 4)

        locations = self.object_model.object_points_wrt_world[sample_indices]
        feature_orientations = self.object_model.object_feature_orientations_wrt_world[
            sample_indices
        ]  # Shape: (n_sampled, 3, 3)

        for location, feature_orientation in zip(locations, feature_orientations):
            surface_normal = feature_orientation[0, :]

            arrow_normal = Arrow(
                location,
                location + surface_normal * arrow_length,
                c="gray",
            )
            arrow_normal.alpha(0.4)
            self.object_pose_vector_arrows.append(arrow_normal)
            self.plotter.at(self.main_renderer_ix).add(arrow_normal)

    def _add_sensor_location_spheres(self) -> None:
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

    def _add_sensor_images(self) -> None:
        """Add sensor_0 (patch) and sensor_1 (view_finder) RGB images."""
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

    def _add_summary_text(self) -> None:
        """Create summary text for current timestep."""
        num_hypotheses_in_reference_frame = sum(self.hypotheses_inside_reference_frame)
        num_hypotheses_outside_reference_frame = sum(
            ~self.hypotheses_inside_reference_frame
        )

        formatted_position = np.array2string(
            np.array(self.data_loader.ground_truth_position),
            precision=2,
            separator=", ",
            suppress_small=False,
        )
        formatted_rotation = np.array2string(
            np.array(self.data_loader.ground_truth_rotation),
            precision=2,
            separator=", ",
            suppress_small=False,
        )

        sm_step = self.data_loader.lm_to_sm_mapping[self.current_timestep]
        object_summary = [
            f"Object: {self.data_loader.target_name}",
            f"Object position: {formatted_position}",
            f"Object rotation: {formatted_rotation}",
            f"LM Step: {self.current_timestep}",
            f"SM Step: {sm_step}",
            f"Current Sensed Curvature: {self.current_sensed_curvature}",
        ]

        hypotheses_summary = [
            f"Num Hyp. Inside Ref. Frame: {num_hypotheses_in_reference_frame}",
            f"Num Hyp. Outside Ref. Frame: {num_hypotheses_outside_reference_frame}",
            f"Current MLH: {self.current_mlh_graph_id}",
        ]

        combined_text = "\n".join(object_summary + hypotheses_summary)

        self.summary_text = Text2D(
            combined_text,
            pos="top-left",
            s=0.8,
            font="Calco",
        )
        self.plotter.at(self.main_renderer_ix).add(self.summary_text)

    def _transform_sensed_orientation_by_hypothesis(
        self, hypothesis_rotation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform sensed orientation by hypothesis rotation.

        Args:
            hypothesis_rotation: Rotation matrix (3, 3) for the hypothesis.

        Returns:
            Tuple of (surface_normal, tangent1, tangent2) vectors.
        """
        features = {"pose_vectors": self.current_sensed_orientation}

        # Reshape single rotation matrix to have batch dimension
        hypothesis_rotation_batch = hypothesis_rotation.reshape(1, 3, 3)
        rotated_features = rotate_pose_dependent_features(
            features, hypothesis_rotation_batch
        )
        transformed_pose_vectors = rotated_features["pose_vectors"][0]

        surface_normal = transformed_pose_vectors[0, :]
        tangent1 = transformed_pose_vectors[1, :]
        tangent2 = transformed_pose_vectors[2, :]

        return surface_normal, tangent1, tangent2

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
            ("Highest Evidence Hypothesis (Cube)", "black", 0.12),
            ("Nearest Neighbors (Point)", TBP_COLORS["yellow"], 0.1),
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


def plot_target_hypotheses(exp_path: Path, episode_id: int = 0) -> int:
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
