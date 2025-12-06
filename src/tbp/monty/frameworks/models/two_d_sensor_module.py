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
    compute_edge_features_center_weighted,
    draw_2d_pose_on_patch,
    normalize,
    project_onto_tangent_plane,
    save_raw_rgb_patch,
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
        }
        self.edge_params = {**default_edge_params, **(edge_detection_params or {})}

        self.debug_visualize = debug_visualize
        self.debug_save_dir = debug_save_dir
        if self.debug_visualize:
            self.debug_counter = 0

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

        # Save raw RGB patch if enabled and conditions are met
        self._save_raw_rgb_if_needed(data, observed_state)

        if observed_state.use_state and observed_state.get_on_object():
            observed_state = self.extract_2d_edge(
                observed_state,
                data["rgba"],
                data["world_camera"],
            )

        if observed_state.use_state:
            observed_state = self._message_noise(observed_state)

        if self.motor_only_step:
            # Set interesting-features flag to False, as should not be passed to
            # LM, even in e.g. pre-training experiments that might otherwise do so
            observed_state.use_state = False

        observed_state = self._state_filter(observed_state)

        if not self.is_exploring:
            self.processed_obs.append(telemetry.processed_obs.__dict__)
            self.states.append(self.state)
            self.visited_locs.append(telemetry.visited_loc)
            self.visited_normals.append(telemetry.visited_normal)

        return observed_state

    def _save_raw_rgb_if_needed(
        self, data: dict[str, Any], observed_state: State
    ) -> None:
        """Save raw RGB patch if saving is enabled and conditions are met.

        Args:
            data: Raw observation data containing 'rgba' key.
            observed_state: Processed state from observation processor.
        """
        if not (
            self.save_raw_rgb
            and not self.is_exploring
            and observed_state.get_on_object()
        ):
            return

        # Extract RGB patch from RGBA if needed
        if data["rgba"].shape[2] == 4:
            patch = data["rgba"][:, :, :3]
        else:
            patch = data["rgba"]

        # Create directory if it doesn't exist
        self.raw_rgb_base_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename: ep{episode:02d}_step{step:03d}.png
        # Uses same counters as debug_visualize
        filename = (
            f"ep{self.episode_counter:02d}_"
            f"step{self.step_counter:03d}.png"
        )
        filepath = self.raw_rgb_base_dir / filename
        save_raw_rgb_patch(patch, str(filepath))

        self.step_counter += 1

    def extract_2d_edge(
        self,
        state: State,
        rgba_image: np.ndarray,
        world_camera: np.ndarray,
    ) -> State:
        """Extract 2D edge-based pose if edge is detected.

        This method attempts to create a fully-defined pose (normal + 2 tangents)
        using edge detection, replacing the standard curvature-based tangents.

        Args:
            state: State with standard features from HabitatObservationProcessor
            rgba_image: RGBA image patch
            world_camera: World to camera transformation matrix

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
            compute_edge_features_center_weighted(
                patch,
                win_sigma=win_sigma,
                ksize=ksize,
                radius=radius,
                sigma_r=sigma_r,
                c_min=c_min,
                e_min=e_min,
            )
        )

        strength_threshold = self.edge_params.get("edge_threshold", 0.1)
        coherence_threshold = self.edge_params.get("coherence_threshold", 0.05)
        has_edge = (edge_strength > strength_threshold) and (
            coherence > coherence_threshold
        )

        if self.debug_visualize:
            angle_deg = np.degrees(edge_orientation)
            label_text = f"{angle_deg:.1f}"
            patch_with_debug = draw_2d_pose_on_patch(
                patch.copy(), edge_orientation, label_text
            )

            filename = (
                f"ep{self.episode_counter:02d}_"
                f"step{self.step_counter:03d}_"
                f"{self.debug_counter:04d}.png"
            )
            filepath = self.debug_save_dir / filename
            plt.imsave(filepath, patch_with_debug)

            self.step_counter += 1
            self.debug_counter += 1

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
