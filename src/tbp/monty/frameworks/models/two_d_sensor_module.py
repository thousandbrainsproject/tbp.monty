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
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from tbp.monty.frameworks.models.abstract_monty_classes import SensorModule
from tbp.monty.frameworks.models.sensor_modules import (
    DefaultMessageNoise,
    FeatureChangeFilter,
    HabitatObservationProcessor,
    MessageNoise,
    PassthroughStateFilter,
    SnapshotTelemetry,
    StateFilter,
    no_message_noise,
)
from tbp.monty.frameworks.models.states import State
from tbp.monty.frameworks.utils.edge_detection_utils import (
    DEFAULT_KERNEL_SIZE,
    DEFAULT_WINDOW_SIGMA,
    compute_arc_length_correction,
    compute_weighted_structure_tensor_edge_features,
    draw_2d_pose_on_patch,
    is_geometric_edge,
    normalize,
    project_onto_tangent_plane,
    save_raw_rgb_if_needed,
)

logger = logging.getLogger(__name__)


class TwoDPoseSM(SensorModule):
    """Sensor Module that turns Habitat camera obs into features at locations.

    Currently extracts all the same features as HabitatSM with addition features
    related to 2D edges.
    """

    def __init__(
        self,
        rng,
        sensor_module_id: str,
        features: list[str],
        save_raw_obs: bool = False,
        edge_detection_params: dict[str, Any] | None = None,
        noise_params: dict[str, Any] | None = None,
        delta_thresholds: dict[str, Any] | None = None,
        debug_visualize=False,
        debug_save_dir=None,
        save_raw_rgb: bool = False,
        raw_rgb_base_dir: str | None = None,
        use_arc_length_correction: bool = True,
    ):
        """Initialize 2D Sensor Module.

        Args:
            rng: Random number generator.
            sensor_module_id: Name of sensor module.
            features: Which features to extract.
            save_raw_obs: Whether to save raw sensory input for logging.
            edge_detection_params: Dictionary of edge detection parameters:
                - gaussian_sigma: Standard deviation for Gaussian smoothing
                  (default: DEFAULT_WINDOW_SIGMA from edge_detection_utils)
                - kernel_size: Kernel size for Gaussian blur
                  (default: DEFAULT_KERNEL_SIZE from edge_detection_utils)
                - edge_threshold: Minimum edge strength threshold (default: 0.1)
                - coherence_threshold: Minimum coherence threshold (default: 0.05)
                - radius: Radius of influence around center in pixels (default: 14.0)
                - sigma_r: Radial falloff parameter for center weighting (default: 7.0)
                - c_min: Minimum coherence threshold to accept edge (default: 0.75)
                - e_min: Minimum local gradient energy threshold (default: 0.01)
            noise_params: Dictionary of noise amount for each feature.
            delta_thresholds: If given, a FeatureChangeFilter will be used to
                check whether the current state's features are significantly different
                from the previous with tolerances set according to `delta_thresholds`.
                Defaults to None.
            debug_visualize: Whether to save debug visualizations of edge detection.
            debug_save_dir: Directory to save debug visualizations.
            save_raw_rgb: Whether to save raw RGB patches without annotations.
            raw_rgb_base_dir: Directory for saving raw RGB images. Images will be
                saved directly in this directory with filenames
                ep{episode:02d}_step{step:03d}.png. If None, defaults to
                current_working_directory/raw_rgb.
            use_arc_length_correction: Whether to apply curvature-based arc length
                correction to 2D displacements. When enabled, compensates for the
                underestimation that occurs when projecting chord length onto tangent
                plane on curved surfaces. Requires principal_curvatures in features.
        """
        self.sensor_module_id = sensor_module_id
        self.save_raw_obs = save_raw_obs

        self._habitat_observation_processor = HabitatObservationProcessor(
            features=features,
            sensor_module_id=sensor_module_id,
        )

        if noise_params:
            self._message_noise: MessageNoise = DefaultMessageNoise(
                noise_params=noise_params, rng=rng
            )
        else:
            self._message_noise = no_message_noise

        if delta_thresholds:
            self._state_filter: StateFilter = FeatureChangeFilter(
                delta_thresholds=delta_thresholds
            )
        else:
            self._state_filter = PassthroughStateFilter()

        self._snapshot_telemetry = SnapshotTelemetry()

        self.features = features
        self.processed_obs = []
        self.states = []
        self.visited_locs = []
        self.visited_normals = []

        default_edge_params = {
            "gaussian_sigma": DEFAULT_WINDOW_SIGMA,
            "kernel_size": DEFAULT_KERNEL_SIZE,
            "edge_threshold": 0.1,
            "coherence_threshold": 0.05,
            "radius": 14.0,
            "sigma_r": 7.0,
            "c_min": 0.75,
            "e_min": 0.01,
            "depth_edge_threshold": 0.01,
        }
        self.edge_params = {**default_edge_params, **(edge_detection_params or {})}
        self.depth_edge_threshold = self.edge_params["depth_edge_threshold"]

        self.debug_visualize = debug_visualize
        self.debug_save_dir = debug_save_dir
        if self.debug_visualize:
            if self.debug_save_dir:
                self.debug_save_dir = Path(self.debug_save_dir)
            else:
                self.debug_save_dir = Path.cwd() / "debug_visualizations"
            self.debug_save_dir.mkdir(parents=True, exist_ok=True)

        self.save_raw_rgb = save_raw_rgb
        self.raw_rgb_base_dir = raw_rgb_base_dir
        if self.save_raw_rgb:
            if self.raw_rgb_base_dir:
                self.raw_rgb_base_dir = Path(self.raw_rgb_base_dir)
            else:
                self.raw_rgb_base_dir = Path.cwd() / "raw_rgb"
            self.raw_rgb_base_dir.mkdir(parents=True, exist_ok=True)

        # Initialize counters if either debug_visualize or save_raw_rgb is enabled
        if self.debug_visualize or self.save_raw_rgb:
            self.episode_counter = 0
            self.step_counter = 0

        # For 2D displacement tracking
        self._previous_location: np.ndarray | None = None
        self._cumulative_2d_position: np.ndarray = np.zeros(2)
        self.use_arc_length_correction = use_arc_length_correction
        self._previous_t1: np.ndarray | None = None
        self._previous_t2: np.ndarray | None = None
        self._R_local_to_global_uv: np.ndarray = np.eye(2, dtype=np.float64)

        # Reference frame for cross-episode consistency
        # These persist across episodes to ensure patches have same orientation
        self._reference_u: np.ndarray | None = None  # Global U axis (world-aligned)
        self._reference_v: np.ndarray | None = None  # Global V axis (world-aligned)

        # For parallel transport of tangent frame
        self._previous_normal: np.ndarray | None = None

    def pre_episode(self):
        """Reset buffer and is_exploring flag."""
        super().pre_episode()
        self._snapshot_telemetry.reset()
        self._state_filter.reset()
        self.is_exploring = False
        self.processed_obs = []
        self.states = []
        self.visited_locs = []
        self.visited_normals = []
        self._previous_location = None
        self._cumulative_2d_position = np.zeros(2)
        self._previous_t1 = None
        self._previous_t2 = None
        self._previous_normal = None
        self._R_local_to_global_uv = np.eye(2, dtype=np.float64)
        # Note: _reference_u, _reference_v, _mapped_points persist across episodes
        if self.debug_visualize or self.save_raw_rgb:
            self.episode_counter += 1
            self.step_counter = 0

    def update_state(self, state):
        """Update information about the sensor's location and rotation."""
        agent_position = state["position"]
        sensor_position = state["sensors"][self.sensor_module_id + ".rgba"]["position"]
        if "motor_only_step" in state.keys():
            self.motor_only_step = state["motor_only_step"]
        else:
            self.motor_only_step = False

        agent_rotation = state["rotation"]
        sensor_rotation = state["sensors"][self.sensor_module_id + ".rgba"]["rotation"]
        self.state = {
            "location": agent_position + sensor_position,
            "rotation": agent_rotation * sensor_rotation,
        }


    def state_dict(self):
        state_dict = self._snapshot_telemetry.state_dict()
        state_dict.update(processed_observations=self.processed_obs)
        return state_dict

    def step(self, data):
        """Turn raw observations into dict of features at location.

        Args:
            data: Raw observations.

        Returns:
            State with features and morphological features. Noise may be added.
            use_state flag may be set.
        """
        if self.save_raw_obs and not self.is_exploring:
            self._snapshot_telemetry.raw_observation(
                data,
                self.state["rotation"],
                self.state["location"]
                if "location" in self.state.keys()
                else self.state["position"],
            )

        # TODO: Long-term refactoring idea - Implement a FeatureRegistry pattern
        # Currently, TwoDPoseSM does a two-step process: (1) extract standard features
        # via HabitatObservationProcessor.process(), then (2) enhance with edge-based
        # pose. A more extensible design would allow registering feature extractors
        # that the processor calls based on the requested features list.

        # Pseudocode of the idea
        # feature_registry = FeatureRegistry()
        # feature_registry.register("hsv", HSVExtractor())
        # feature_registry.register("feature_name", FeatureExtractor(feature_params))

        # processor = HabitatObservationProcessor(
        #     features=["hsv", "feature_name", "etc."],
        #     feature_registry=feature_registry
        # )
        # observed_state, telemetry = processor.process(data)

        observed_state, telemetry = self._habitat_observation_processor.process(data)

        curvature_pose_vectors = observed_state.morphological_features.get(
            "pose_vectors"
        )
        if curvature_pose_vectors is not None:
            curvature_pose_vectors = curvature_pose_vectors.copy()

        if self.save_raw_rgb or self.debug_visualize:
            self.step_counter = save_raw_rgb_if_needed(
                save_raw_rgb=self.save_raw_rgb,
                is_exploring=self.is_exploring,
                observed_state=observed_state,
                rgba_image=data["rgba"],
                raw_rgb_base_dir=self.raw_rgb_base_dir,
                episode_counter=self.episode_counter,
                step_counter=self.step_counter,
            )

        if observed_state.use_state and observed_state.get_on_object():
            observed_state = self.extract_2d_edge(
                observed_state,
                data["rgba"],
                data["world_camera"],
                depth_image=data.get("depth"),
            )

        if observed_state.use_state:
            observed_state = self._message_noise(observed_state)

        if self.motor_only_step:
            # Set interesting-features flag to False, as should not be passed to
            # LM, even in e.g. pre-training experiments that might otherwise do so
            observed_state.use_state = False

        # Calculate 2D displacement and accumulate cumulative 2D position
        # Using Option A (local tangent coords) + Option C
        # (incremental rotation tracking)
        self._update_2d_position_and_displacement(
            observed_state, curvature_pose_vectors
        )

        observed_state = self._state_filter(observed_state)

        if not self.is_exploring:
            self.processed_obs.append(telemetry.processed_obs.__dict__)
            self.states.append(self.state)
            self.visited_locs.append(telemetry.visited_loc)
            self.visited_normals.append(telemetry.visited_normal)

        return observed_state

    def _update_2d_position_and_displacement(
        self,
        observed_state: State,
        curvature_pose_vectors: np.ndarray | None,
    ) -> None:
        """Calculate 2D displacement and accumulate cumulative 2D position.

        Updates the observed_state with displacement and location information.

        Args:
            observed_state: State to update with 2D displacement and position.
            curvature_pose_vectors: Optional curvature-based pose vectors from
                morphological features.
        """
        if observed_state.get_on_object():
            surface_normal = observed_state.get_surface_normal()
            n = normalize(surface_normal)

            # Get principal curvatures for isotropic detection and arc length correction
            principal_curvatures = observed_state.non_morphological_features.get(
                "principal_curvatures"
            )

            # Check for isotropic surface (sphere, plane) where k1 ≈ k2
            isotropic_threshold = 0.5  # curvature units (1/m)
            is_isotropic = False
            if principal_curvatures is not None:
                k1, k2 = principal_curvatures[0], principal_curvatures[1]
                is_isotropic = abs(k1 - k2) < isotropic_threshold

            # Build stable local tangent basis (t1, t2)
            # Use parallel transport for subsequent steps (Rodrigues rotation)
            # This avoids hourglass distortion from re-computing world alignment
            if (
                self._previous_t1 is not None
                and self._previous_t2 is not None
                and self._previous_normal is not None
            ):
                # Subsequent step: parallel transport previous frame
                t1, t2 = self._parallel_transport_frame(
                    self._previous_t1, self._previous_t2, self._previous_normal, n
                )
            else:
                # First step of episode: initialize tangent frame
                if is_isotropic:
                    # Isotropic surface - use world-aligned
                    t1_raw, _ = self._get_world_aligned_tangent_frame(n)
                elif (
                    curvature_pose_vectors is not None
                    and len(curvature_pose_vectors) >= 3
                ):
                    # Anisotropic surface - use direction with MINIMUM curvature magnitude
                    # This is more stable across positions (e.g., axial direction on cylinder)
                    # dir1 (index 1) corresponds to k1, dir2 (index 2) corresponds to k2
                    if principal_curvatures is not None:
                        if abs(principal_curvatures[1]) < abs(principal_curvatures[0]):
                            t1_raw = curvature_pose_vectors[2].copy()
                        else:
                            t1_raw = curvature_pose_vectors[1].copy()
                    else:
                        t1_raw = curvature_pose_vectors[2].copy()

                    # Align sign to closest world axis for cross-epoch consistency
                    # The curvature direction has arbitrary sign due to right-hand rule
                    # coupling with the other principal direction in sensor_processing.py
                    world_axes = np.array(
                        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
                    )
                    dots = world_axes @ t1_raw
                    max_idx = np.argmax(np.abs(dots))
                    if dots[max_idx] < 0:
                        t1_raw = -t1_raw
                else:
                    # Fallback - use world X
                    t1_raw = np.array([1.0, 0.0, 0.0], dtype=np.float64)

                # Enforce orthonormal tangent basis
                t1 = normalize(project_onto_tangent_plane(t1_raw, n))

                # Handle degenerate case (t1_raw parallel to normal)
                if np.linalg.norm(t1) < 1e-12:
                    fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                    if abs(np.dot(fallback, n)) > 0.99:
                        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                    t1 = normalize(project_onto_tangent_plane(fallback, n))

                # Right-handed tangent basis: t2 = n × t1, then re-orthogonalize t1
                t2 = normalize(np.cross(n, t1))
                t1 = normalize(np.cross(t2, n))

                # Compute deterministic global UV basis from world axes
                u_global, v_global = self._get_world_aligned_tangent_frame(n)

                if self._reference_u is None:
                    # First episode - establish reference frame
                    self._reference_u = u_global.copy()
                    self._reference_v = v_global.copy()

                # Align (t1, t2) to REFERENCE basis for cross-episode consistency
                if np.dot(t1, self._reference_u) < 0:
                    t1 = -t1
                    t2 = -t2
                # Handle axis swap: check if t1 aligns better with reference_v
                if abs(np.dot(t1, self._reference_u)) < abs(np.dot(t2, self._reference_u)):
                    t1, t2 = t2, -t1  # swap and keep right-handed

            # Compute tangent displacement in world coords
            if self._previous_location is None:
                displacement_3d = np.zeros(3)
            else:
                displacement_3d = observed_state.location - self._previous_location

            d_tan = project_onto_tangent_plane(displacement_3d, n)

            # Apply arc length correction if enabled (Part 4)
            if (
                self.use_arc_length_correction
                and principal_curvatures is not None
                and curvature_pose_vectors is not None
                and len(curvature_pose_vectors) >= 3
            ):
                chord_length = np.linalg.norm(d_tan)
                if chord_length > 1e-12:
                    dir1 = curvature_pose_vectors[1]
                    dir2 = curvature_pose_vectors[2]
                    directional_curvature = self._get_directional_curvature(
                        d_tan,
                        principal_curvatures[0],
                        principal_curvatures[1],
                        dir1,
                        dir2,
                    )
                    arc_length = compute_arc_length_correction(
                        chord_length, directional_curvature
                    )
                    d_tan = d_tan * (arc_length / chord_length)

            # Local displacement in (t1, t2) basis
            du = float(np.dot(d_tan, t1))
            dv = float(np.dot(d_tan, t2))
            local_disp = np.array([du, dv], dtype=np.float64)

            # Incremental rotation tracking (Option C)
            if self._previous_t1 is None:
                # First step: compute R_local_to_global = B_global.T @ B_local
                # B_local = [t1, t2] as columns, B_global = [u, v] as columns
                # R[i,j] = global_i · local_j
                self._R_local_to_global_uv = np.array(
                    [
                        [np.dot(self._reference_u, t1), np.dot(self._reference_u, t2)],
                        [np.dot(self._reference_v, t1), np.dot(self._reference_v, t2)],
                    ],
                    dtype=np.float64,
                )
                global_disp = self._R_local_to_global_uv @ local_disp
            else:
                # Subsequent step - incremental rotation tracking
                # Build relative rotation: current local -> previous local
                R_rel = np.array(
                    [
                        [np.dot(self._previous_t1, t1), np.dot(self._previous_t1, t2)],
                        [np.dot(self._previous_t2, t1), np.dot(self._previous_t2, t2)],
                    ],
                    dtype=np.float64,
                )

                # Compose: current local -> global =
                # (prev local -> global) @ (curr -> prev)
                self._R_local_to_global_uv = self._R_local_to_global_uv @ R_rel

                # Transform local displacement to global UV
                global_disp = self._R_local_to_global_uv @ local_disp

            # Accumulate in global UV frame
            self._cumulative_2d_position += global_disp

            # Save for next step (including normal for parallel transport)
            self._previous_location = observed_state.location.copy()
            self._previous_t1 = t1.copy()
            self._previous_t2 = t2.copy()
            self._previous_normal = n.copy()

            # Store displacement and update location for graph building
            observed_state.set_displacement(
                np.array([global_disp[0], global_disp[1], 0.0], dtype=np.float64)
            )
            observed_state.location = np.array(
                [self._cumulative_2d_position[0], self._cumulative_2d_position[1], 0.0],
                dtype=np.float64,
            )
        else:
            observed_state.set_displacement(np.zeros(3))
            # Don't update previous state when off-object

    def extract_2d_edge(
        self,
        state: State,
        rgba_image: np.ndarray,
        world_camera: np.ndarray,
        depth_image: np.ndarray | None = None,
    ) -> State:
        """Extract 2D edge-based pose if edge is detected.

        This method attempts to create a fully-defined pose (normal + 2 tangents)
        using edge detection, replacing the standard curvature-based tangents.

        Args:
            state: State with standard features from HabitatObservationProcessor
            rgba_image: RGBA image patch
            world_camera: World to camera transformation matrix
            depth_image: Optional depth image patch for filtering geometric edges.
                If provided, edges at depth discontinuities (object boundaries,
                surface creases) are filtered out, keeping only texture edges.

        Returns:
            State with edge-based pose vectors if edge detected,
            otherwise returns the original state unchanged.
        """
        if "pose_vectors" not in state.morphological_features:
            state.morphological_features["pose_from_edge"] = False
            return state

        surface_normal = normalize(state.morphological_features["pose_vectors"][0])

        if rgba_image.shape[2] == 4:
            patch = rgba_image[:, :, :3]
        else:
            patch = rgba_image

        win_sigma = self.edge_params.get("gaussian_sigma", DEFAULT_WINDOW_SIGMA)
        ksize = self.edge_params.get("kernel_size", DEFAULT_KERNEL_SIZE)
        radius = self.edge_params.get("radius", 14.0)
        sigma_r = self.edge_params.get("sigma_r", 7.0)
        c_min = self.edge_params.get("c_min", 0.75)
        e_min = self.edge_params.get("e_min", 0.01)
        edge_strength, coherence, edge_orientation = (
            compute_weighted_structure_tensor_edge_features(
                patch,
                win_sigma=win_sigma,
                ksize=ksize,
                radius=radius,
                sigma_r=sigma_r,
                c_min=c_min,
                e_min=e_min,
            )
        )

        # Filter out geometric edges (depth discontinuities) if depth is available
        if edge_strength > 0 and depth_image is not None:
            if is_geometric_edge(
                depth_image, edge_orientation, self.depth_edge_threshold
            ):
                edge_strength = 0.0
                coherence = 0.0
                edge_orientation = 0.0

        strength_threshold = self.edge_params.get("edge_threshold", 0.1)
        coherence_threshold = self.edge_params.get("coherence_threshold", 0.05)
        has_edge = (edge_strength > strength_threshold) and (
            coherence > coherence_threshold
        )

        if self.debug_visualize:
            patch_with_debug = draw_2d_pose_on_patch(
                patch.copy(), edge_orientation, label_text=None
            )

            angle_deg = np.degrees(edge_orientation)
            filename = (
                f"ep{self.episode_counter:02d}_"
                f"step{self.step_counter:03d}_"
                f"theta{angle_deg:.1f}_"
                f"coh{coherence:.2f}_"
                f"str{edge_strength:.2f}.png"
            )
            filepath = self.debug_save_dir / filename
            plt.imsave(filepath, patch_with_debug)

            self.step_counter += 1

        if not has_edge:
            state.morphological_features["pose_from_edge"] = False
            return state

        edge_tangent = self.edge_angle_to_3d_tangent(
            edge_orientation, surface_normal, world_camera
        )
        edge_tangent = normalize(edge_tangent)
        edge_perp = normalize(np.cross(surface_normal, edge_tangent))

        state.morphological_features["pose_vectors"] = np.vstack(
            [
                surface_normal,
                edge_tangent,
                edge_perp,
            ]
        )
        state.morphological_features["pose_fully_defined"] = True
        state.morphological_features["pose_from_edge"] = True
        state.non_morphological_features["pose_fully_defined"] = True

        if "edge_strength" in self.features:
            state.non_morphological_features["edge_strength"] = edge_strength
        if "coherence" in self.features:
            state.non_morphological_features["coherence"] = coherence

        return state

    def edge_angle_to_3d_tangent(self, theta, normal, world_camera):
        """Projects a 2D edge angle from an image to a 3D tangent vector on a surface.

        This function performs the following steps to convert an edge detected in a
        2D image into a 3D tangent vector on the object's surface:

        1. Transform the surface normal from world coordinates to camera coordinates
        2. Construct an orthonormal tangent basis (tx, ty) on the surface tangent
           plane in camera coordinates, aligned with the image coordinate system:
           - tx aligns with image +x (rightward)
           - ty aligns with image +y (downward, since image y-axis points down)
        3. Express the edge direction in this tangent basis using the angle theta
        4. Transform the resulting tangent vector back to world coordinates

        An edge in the image lies on the projection of a 3D curve on the surface.
        Since the surface is locally planar, the edge must be tangent to that surface.
        By building a tangent basis aligned with the image axes, we can "lift" the 2D
        edge angle back to 3D.

        Args:
            theta: Edge angle in radians, measured counterclockwise from the image
                +x axis (rightward). In image coordinates, +x is right and +y is down.
            normal: Surface normal vector in world frame.
            world_camera: 3x3 or 4x4 rotation matrix transforming from world
                coordinates to camera coordinates.

        Returns:
            3D unit tangent vector in world frame, representing the direction of the
            edge on the surface.

        Raises:
            ValueError: If the input normal has zero or near-zero length.
        """
        n_world = normalize(normal)
        if np.allclose(n_world, 0.0):
            raise ValueError(
                "Cannot compute tangent vector: input normal has zero or "
                "near-zero length"
            )

        world_camera = (
            world_camera[:3, :3] if world_camera.shape == (4, 4) else world_camera
        )

        n_cam = world_camera @ n_world

        image_x = np.array([1.0, 0.0, 0.0])
        image_y = np.array([0.0, -1.0, 0.0])

        tx = project_onto_tangent_plane(image_x, n_cam)
        if np.linalg.norm(tx) < 1e-12:
            # If image x-axis is nearly parallel to normal,
            # use a different reference axis
            fallback_axis = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(fallback_axis, n_cam)) > 0.99:
                fallback_axis = np.array([0.0, 1.0, 0.0])
            tx = project_onto_tangent_plane(fallback_axis, n_cam)
        tx = normalize(tx)

        ty = normalize(np.cross(n_cam, tx))

        # Ensure ty points in the same direction as image y-axis (down)
        if np.dot(ty, image_y) < 0:
            ty = -ty

        t_cam = np.cos(theta) * tx + np.sin(theta) * ty
        t_cam = normalize(t_cam)

        t_world = world_camera.T @ t_cam
        return normalize(t_world)

    def _compute_2d_displacement(
        self,
        current_location: np.ndarray,
        surface_normal: np.ndarray | None,
        principal_curvatures: np.ndarray | None = None,
        curvature_direction: np.ndarray | None = None,
    ) -> np.ndarray:
        """Project 3D displacement onto tangent plane for 2D surface movement.

        Optionally applies arc length correction using surface curvature to
        compensate for underestimation on curved surfaces.

        Args:
            current_location: Current sensor location in world coordinates.
            surface_normal: Surface normal at current location.
            principal_curvatures: Optional array [k1, k2] of principal curvatures.
                If provided along with curvature_direction, arc length correction
                is applied.
            curvature_direction: Optional first principal curvature direction
                (unit vector in tangent plane). Required for arc length correction.

        Returns:
            Displacement vector projected onto the tangent plane (perpendicular
            to the surface normal). If curvature info is provided, the magnitude
            is corrected from chord length to arc length. Returns zeros if no
            previous location exists or if surface_normal is None.
        """
        if self._previous_location is None or surface_normal is None:
            return np.zeros(3)

        displacement_3d = current_location - self._previous_location
        displacement_2d = project_onto_tangent_plane(displacement_3d, surface_normal)

        # Apply arc length correction if curvature info is available
        if principal_curvatures is not None and curvature_direction is not None:
            chord_length = np.linalg.norm(displacement_2d)
            if chord_length > 1e-12:
                k1, k2 = principal_curvatures[0], principal_curvatures[1]
                dir1 = project_onto_tangent_plane(curvature_direction, surface_normal)
                dir1 = normalize(dir1)

                directional_curvature = self._get_directional_curvature(
                    displacement_2d, k1, k2, dir1
                )
                arc_length = compute_arc_length_correction(
                    chord_length, directional_curvature
                )
                # Scale displacement to arc length
                scale_factor = arc_length / chord_length
                print(f"scale_factor: {scale_factor}")
                displacement_2d = displacement_2d * scale_factor

        return displacement_2d

    def _get_directional_curvature(
        self,
        movement_direction: np.ndarray,
        k1: float,
        k2: float,
        dir1: np.ndarray,
        dir2: np.ndarray,
    ) -> float:
        """Compute normal curvature in the direction of movement using Euler's formula.

        Uses both principal directions explicitly:
            k_direction = k1*cos²(angle with dir1) + k2*cos²(angle with dir2)

        This avoids assumptions about direction ordering and is cleaner since
        dir1 and dir2 are orthonormal in the tangent plane.

        Args:
            movement_direction: Vector indicating movement direction (will be normalized).
            k1: First principal curvature (corresponds to dir1).
            k2: Second principal curvature (corresponds to dir2).
            dir1: First principal curvature direction (unit vector in tangent plane).
            dir2: Second principal curvature direction (unit vector in tangent plane).

        Returns:
            Normal curvature in the movement direction.
        """
        movement_norm = np.linalg.norm(movement_direction)
        if movement_norm < 1e-12:
            return 0.0

        move_hat = movement_direction / movement_norm

        # Project movement onto both principal directions
        cos_a = np.dot(move_hat, dir1)
        cos_b = np.dot(move_hat, dir2)

        # Euler's formula with both directions explicitly
        return k1 * cos_a**2 + k2 * cos_b**2

    def _stabilize_tangent_frame(
        self,
        t1_new: np.ndarray,
        t2_new: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stabilize tangent frame against sign flips and axis swaps.

        Ensures continuity with previous frame by:
        1. Checking for axis swap (t1/t2 roles exchanged)
        2. Correcting sign flips (t1 vs -t1)

        Args:
            t1_new: First tangent direction from current pose_vectors.
            t2_new: Second tangent direction from current pose_vectors.

        Returns:
            Stabilized (t1, t2) that maximizes alignment with previous frame.
        """
        if self._previous_t1 is None:
            # First step - no previous frame to align to
            return t1_new, t2_new

        # Compute alignment scores for both axes in both configurations
        dot_t1_t1 = np.dot(t1_new, self._previous_t1)  # t1 aligned with prev t1
        dot_t1_t2 = np.dot(t1_new, self._previous_t2)  # t1 aligned with prev t2
        dot_t2_t1 = np.dot(t2_new, self._previous_t1)  # t2 aligned with prev t1
        dot_t2_t2 = np.dot(t2_new, self._previous_t2)  # t2 aligned with prev t2

        # Check for axis swap: compare total alignment with and without swap
        # Use sum of absolute alignments - swap only if it improves BOTH axes
        current_alignment = abs(dot_t1_t1) + abs(dot_t2_t2)
        swapped_alignment = abs(dot_t1_t2) + abs(dot_t2_t1)

        # Add hysteresis (0.1) to prevent oscillation near ambiguous orientations
        if swapped_alignment > current_alignment + 0.1:
            # Swap detected - exchange t1 and t2
            t1_new, t2_new = t2_new.copy(), t1_new.copy()
            # Update alignment scores after swap
            dot_t1_t1 = dot_t2_t1
            dot_t2_t2 = dot_t1_t2

        # Correct sign flips
        if dot_t1_t1 < 0:
            t1_new = -t1_new
        if dot_t2_t2 < 0:
            t2_new = -t2_new

        return t1_new, t2_new

    def _parallel_transport_frame(
        self,
        t1_prev: np.ndarray,
        t2_prev: np.ndarray,
        n_prev: np.ndarray,
        n_curr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transport tangent frame from previous to current position via Rodrigues.

        Uses the minimal rotation (Rodrigues formula) that maps n_prev -> n_curr,
        then applies that rotation to the previous tangent basis. This preserves
        parallel transport along the surface and avoids the "hourglass" distortion
        that occurs when re-computing world-aligned frames at each step.

        Args:
            t1_prev: Previous first tangent direction.
            t2_prev: Previous second tangent direction.
            n_prev: Previous surface normal (unit vector).
            n_curr: Current surface normal (unit vector).

        Returns:
            (t1_transported, t2_transported): Tangent frame transported to current
                position, re-orthonormalized to lie exactly in the tangent plane.
        """
        # Rotation axis is n_prev x n_curr
        cross = np.cross(n_prev, n_curr)
        sin_theta = np.linalg.norm(cross)

        if sin_theta < 1e-10:
            # Normals are parallel - no rotation needed
            # Just project onto current tangent plane (handles numerical drift)
            t1 = t1_prev - np.dot(t1_prev, n_curr) * n_curr
            t1_norm = np.linalg.norm(t1)
            if t1_norm < 1e-12:
                return t1_prev.copy(), t2_prev.copy()
            t1 = t1 / t1_norm
            t2 = np.cross(n_curr, t1)
            return t1, t2

        k = cross / sin_theta  # Unit rotation axis
        cos_theta = np.clip(np.dot(n_prev, n_curr), -1.0, 1.0)

        # Rodrigues formula: R = I + K*sin(theta) + K^2*(1-cos(theta))
        # where K is the skew-symmetric matrix of k
        K = np.array(
            [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float64
        )
        R = np.eye(3) + K * sin_theta + K @ K * (1 - cos_theta)

        # Rotate previous tangent vectors
        t1_rotated = R @ t1_prev
        t2_rotated = R @ t2_prev

        # Project back onto tangent plane and re-orthonormalize
        # (should be nearly there already, but ensures numerical stability)
        t1 = t1_rotated - np.dot(t1_rotated, n_curr) * n_curr
        t1_norm = np.linalg.norm(t1)
        if t1_norm < 1e-12:
            # Degenerate case - fall back to world-aligned
            return self._get_world_aligned_tangent_frame(n_curr)
        t1 = t1 / t1_norm

        # Re-orthogonalize t2 using cross product
        t2 = np.cross(n_curr, t1)
        t2 = t2 / np.linalg.norm(t2)

        return t1, t2

    def _get_world_aligned_tangent_frame(
        self,
        surface_normal: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute tangent frame aligned with world X axis.

        Used as fallback for isotropic surfaces (spheres, planes) where
        principal curvature directions are undefined/arbitrary.

        Projects world X onto the tangent plane. If surface_normal is
        parallel to world X, falls back to world Y.

        Args:
            surface_normal: Unit surface normal vector.

        Returns:
            (t1, t2): Orthonormal tangent vectors, t1 aligned with world X
                      projection onto tangent plane.
        """
        world_x = np.array([1.0, 0.0, 0.0])
        world_y = np.array([0.0, 1.0, 0.0])

        # Project world X onto tangent plane
        t1 = world_x - np.dot(world_x, surface_normal) * surface_normal
        t1_norm = np.linalg.norm(t1)

        # If surface normal is nearly parallel to world X, use world Y
        if t1_norm < 0.1:
            t1 = world_y - np.dot(world_y, surface_normal) * surface_normal
            t1_norm = np.linalg.norm(t1)

        t1 = t1 / t1_norm

        # t2 = surface_normal × t1 (perpendicular to both)
        t2 = np.cross(surface_normal, t1)
        t2 = t2 / np.linalg.norm(t2)

        return t1, t2

    def _to_local_tangent_coords(
        self,
        displacement_world: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
    ) -> np.ndarray:
        """Express world-space displacement in local tangent coordinates.

        Args:
            displacement_world: Displacement vector in world coordinates
                (should already be in the tangent plane).
            t1: First tangent direction (unit vector).
            t2: Second tangent direction (unit vector).

        Returns:
            Array [u, v] where u is component along t1, v along t2.
        """
        u = np.dot(displacement_world, t1)
        v = np.dot(displacement_world, t2)
        return np.array([u, v])
