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
import quaternion as qt

from tbp.monty.frameworks.models.abstract_monty_classes import SensorID, SensorModule
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    SensorState,
)
from tbp.monty.frameworks.models.sensor_modules import (
    DefaultMessageNoise,
    FeatureChangeFilter,
    HabitatObservationProcessor,
    MessageNoise,
    NoMessageNoise,
    PassthroughStateFilter,
    SnapshotTelemetry,
    StateFilter,
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
        self._rng = rng
        self._habitat_observation_processor = HabitatObservationProcessor(
            features=features,
            sensor_module_id=sensor_module_id,
        )
        if noise_params:
            self._message_noise: MessageNoise = DefaultMessageNoise(
                noise_params=noise_params, rng=rng
            )
        else:
            self._message_noise = NoMessageNoise()
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
        self.sensor_module_id = sensor_module_id
        self.save_raw_obs = save_raw_obs
        self.visited_locs = []
        self.visited_normals = []

        default_edge_params = {
            "gaussian_sigma": DEFAULT_WINDOW_SIGMA,
            "kernel_size": DEFAULT_KERNEL_SIZE,
            "edge_threshold": 0.1,
            "coherence_threshold": 0.05,
            "radius": 14.0,
            "sigma_r": 7.0,
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

        if self.debug_visualize or self.save_raw_rgb:
            self.episode_counter = 0
            self.step_counter = 0

        # For 2D displacement tracking
        self._previous_location: np.ndarray | None = None
        self._cumulative_2d_position: np.ndarray = np.zeros(2)
        self.use_arc_length_correction = use_arc_length_correction

        # TODO: Reference frame for cross-episode consistency (not yet implemented)
        # These could be used to align the initial u-v frame with a canonical
        # direction (e.g., world axis or primary curvature direction) to make
        # 2D maps deterministic across exploration sessions.
        self._reference_u: np.ndarray | None = None  # Global U axis (world-aligned)
        self._reference_v: np.ndarray | None = None  # Global V axis (world-aligned)

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
        # Reset reference frame for new object (each episode = new object)
        self._reference_u = None
        self._reference_v = None
        if self.debug_visualize or self.save_raw_rgb:
            self.episode_counter += 1
            self.step_counter = 0

    def update_state(self, agent: AgentState):
        """Update information about the sensor's location and rotation."""
        sensor = agent.sensors[SensorID(self.sensor_module_id + ".rgba")]
        self.state = SensorState(
            position=agent.position
            + qt.rotate_vectors(agent.rotation, sensor.position),
            rotation=agent.rotation * sensor.rotation,
        )
        self.motor_only_step = agent.motor_only_step

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
                data, self.state.rotation, self.state.position
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
            observed_state = self._message_noise(observed_state, rng=self._rng)

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

        Uses a world-anchored tangent frame approach: at each position, the tangent
        frame is computed by projecting fixed world axes onto the local tangent plane.
        This ensures globally consistent coordinates without rotation accumulation.

        For developable surfaces (cylinder, cone): produces correct flat representation.
        For non-developable surfaces (sphere): produces Mercator-like projection.

        Args:
            observed_state: State to update with 2D displacement and position.
            curvature_pose_vectors: Optional curvature-based pose vectors from
                morphological features (used for arc length correction).
        """
        if not observed_state.get_on_object():
            observed_state.set_displacement(np.zeros(3))
            return

        current_location = observed_state.location.copy()
        surface_normal = observed_state.get_surface_normal()

        if self._previous_location is None or surface_normal is None:
            self._initialize_basis(surface_normal)
            self._previous_location = current_location
            self._cumulative_2d_position = np.zeros(2)
            observed_state.location = np.array([0.0, 0.0, 0.0])
            return

        # Compute raw 3D displacement
        displacement_3d = current_location - self._previous_location
        chord_length = np.linalg.norm(displacement_3d)

        # 2. Parallel Transport: Update basis vectors to the new tangent plane
        # Find the rotation that maps the old normal to the new normal
        self._update_basis(surface_normal)

        du_raw = np.dot(displacement_3d, self._basis_u)
        dv_raw = np.dot(displacement_3d, self._basis_v)
        direction_uv = np.array([du_raw, dv_raw])
        dir_norm = np.linalg.norm(direction_uv)
        if dir_norm > 1e-12:
            direction_uv /= dir_norm

        # Arc length correction
        step_magnitude = chord_length
        principal_curvatures = observed_state.morphological_features.get(
            "principal_curvatures"
        )
        if principal_curvatures is not None and curvature_pose_vectors is not None:
            # Project movement onto tangent plane for curvature calculation
            d_tan = project_onto_tangent_plane(displacement_3d, surface_normal)
            tan_length = np.linalg.norm(d_tan)
            if tan_length > 1e-12:
                directional_curvature = self._get_directional_curvature(
                    d_tan,
                    principal_curvatures[0],
                    principal_curvatures[1],
                    curvature_pose_vectors[1],
                    curvature_pose_vectors[2],
                )
                step_magnitude = compute_arc_length_correction(
                    tan_length,
                    directional_curvature,
                )

        du = direction_uv[0] * step_magnitude
        dv = direction_uv[1] * step_magnitude
        self._cumulative_2d_position[0] += du
        self._cumulative_2d_position[1] += dv

        observed_state.set_displacement(np.array([du, dv, 0.0]))
        observed_state.location = np.array(
            [self._cumulative_2d_position[0], self._cumulative_2d_position[1], 0.0]
        )

        # Project 3D pose vectors into local UV basis
        pose_vecs = observed_state.morphological_features.get("pose_vectors")
        if pose_vecs is not None and self._basis_u is not None:
            uv_pose_vecs = []
            for v_3d in pose_vecs:
                # Project the 3D vector onto local u, v and the normal basis
                u_comp = np.dot(v_3d, self._basis_u)
                v_comp = np.dot(v_3d, self._basis_v)
                n_comp = np.dot(v_3d, self._previous_normal)
                uv_pose_vecs.append([u_comp, v_comp, n_comp])
            observed_state.morphological_features["pose_vectors"] = np.array(
                uv_pose_vecs
            )

        # Save current location for next step
        self._previous_location = current_location.copy()

    def _initialize_basis(self, surface_normal: np.ndarray) -> None:
        # Use World Up as the reference to keep the 'V' axis generally 'Up'
        world_up = np.array([0, 1, 0])

        # If the normal is pointing along the Y axis, use Z as fallback
        if abs(np.dot(world_up, surface_normal)) > 0.95:
            world_up = np.array([0, 0, 1])

        # 1. Basis U is perpendicular to normal and the 'Up' vector
        # This usually aligns U with the 'Horizon'
        self._basis_u = np.cross(world_up, surface_normal)
        self._basis_u /= np.linalg.norm(self._basis_u)

        # 2. Basis V completes the right-handed system
        # Given normal points to camera, this makes V point 'Up' on the surface
        self._basis_v = np.cross(surface_normal, self._basis_u)

        self._previous_normal = surface_normal

    def _update_basis(self, new_normal: np.ndarray) -> None:
        """Rotates the basis vectors to stay tangent to the new normal."""
        # Standard Parallel Transport (The 'Levi-Civita' connection)
        # We rotate the basis vectors around the axis perpendicular to both normals
        old_n = self._previous_normal
        new_n = new_normal

        axis = np.cross(old_n, new_n)
        axis_len = np.linalg.norm(axis)

        if axis_len > 1e-9:
            axis /= axis_len
            angle = np.arccos(np.clip(np.dot(old_n, new_n), -1.0, 1.0))

            # Rodrigues' rotation formula to tilt the basis
            def rotate(v, a, theta):
                return (
                    v * np.cos(theta)
                    + np.cross(a, v) * np.sin(theta)
                    + a * np.dot(a, v) * (1 - np.cos(theta))
                )

            self._basis_u = rotate(self._basis_u, axis, angle)
            self._basis_v = rotate(self._basis_v, axis, angle)

        # Ensure basis stays perfectly orthogonal to new normal
        self._basis_u -= np.dot(self._basis_u, new_n) * new_n
        self._basis_u /= np.linalg.norm(self._basis_u)
        self._basis_v = np.cross(new_n, self._basis_u)

        self._previous_normal = new_n

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
        edge_strength, coherence, edge_orientation = (
            compute_weighted_structure_tensor_edge_features(
                patch,
                win_sigma=win_sigma,
                ksize=ksize,
                radius=radius,
                sigma_r=sigma_r,
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
