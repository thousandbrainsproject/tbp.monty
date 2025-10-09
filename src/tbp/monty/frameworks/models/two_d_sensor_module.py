# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""2D Sensor Module for extracting 2D pose information from RGB images."""

import csv
import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import quaternion

from tbp.monty.frameworks.models.sensor_modules import DetailedLoggingSM, NoiseMixin
from tbp.monty.frameworks.models.states import State
from tbp.monty.frameworks.utils.edge_detection_utils import structure_tensor_center

logger = logging.getLogger(__name__)


class TwoDPoseSM(DetailedLoggingSM, NoiseMixin):
    """Sensor Module that extracts 2D edges."""

    def __init__(
        self,
        sensor_module_id,
        features,
        save_raw_obs=False,
        edge_detection_params=None,
        noise_params=None,
        process_all_obs=False,
        debug_visualize=False,
        debug_save_dir=None,
        **kwargs,
    ):
        """Initialize 2D Pose Sensor Module.

        Args:
            sensor_module_id: Name of sensor module.
            features: Which features to extract. Should include "pose_vectors" and
                "on_object". Additional features: "edge_strength", "coherence".
            save_raw_obs: Whether to save raw sensory input for logging.
            edge_detection_params: Dictionary of edge detection parameters:
                - gaussian_sigma: Standard deviation for Gaussian smoothing (default: 1.0)
                - edge_threshold: Minimum edge strength threshold (default: 0.1)
                - non_max_radius: Radius for non-maximum suppression (default: 2)
                - fallback_to_normal: Use surface normal when no edge detected (default: True)
            noise_params: Dictionary of noise amount for each feature.
            process_all_obs: Enable explicitly to enforce that off-observations are
                still processed by LMs, primarily for the purpose of unit testing.
            debug_visualize: Whether to save debug visualizations of edge detection.
            debug_save_dir: Directory to save debug visualizations (required if debug_visualize=True).
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(
            sensor_module_id,
            save_raw_obs,
            noise_params=noise_params,
            **kwargs,
        )

        # Validate features
        possible_features = [
            "on_object",
            "object_coverage",
            "pose_vectors",
            "pose_fully_defined",
            "edge_strength",
            "edge_orientation",
            "edge_tangent",
            "coherence",
            "rgba",
            "hsv",
        ]
        for feature in features:
            assert feature in possible_features, (
                f"{feature} not part of {possible_features}"
            )

        self.features = features
        self.processed_obs = []
        self.states = []
        self.on_object_obs_only = True
        self.process_all_obs = process_all_obs

        # Set default edge detection parameters
        default_edge_params = {
            "gaussian_sigma": 1.0,
            "edge_threshold": 0.1,
            "non_max_radius": 2,
            "fallback_to_normal": True,
        }
        self.edge_params = edge_detection_params or default_edge_params
        self.edge_params.update(edge_detection_params or {})

        # Canonical frame tracking for feature-aligned pose vectors
        self._canonical_tangent = None
        self._canonical_perpendicular = None
        self._canonical_normal = None
        self._pose_frame_confidence = 0.0
        self._edge_confidence_threshold = (
            self.edge_params.get("canonical_confidence_threshold")
            or self.edge_params.get("edge_threshold", 0.1)
        )

        # Debug visualization setup
        self.debug_visualize = debug_visualize
        self.debug_save_dir = Path(debug_save_dir) if debug_save_dir else None
        self.episode_counter = 0
        self.step_counter = 0
        self.debug_counter = 0

        if self.debug_visualize:
            assert self.debug_save_dir is not None, (
                "debug_save_dir must be specified when debug_visualize=True"
            )
            self.debug_save_dir.mkdir(parents=True, exist_ok=True)

    def pre_episode(self):
        """Reset buffer and episode-specific variables."""
        super().pre_episode()
        self.processed_obs = []
        self.states = []
        self.episode_counter += 1
        self.step_counter = 0
        self._reset_canonical_frame()

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

    def step(self, data):
        """Process RGB image to extract 2D pose and features.

        Args:
            data: Raw observations containing rgba, depth, semantic_3d, etc.

        Returns:
            State with 2D pose vectors and features. Noise may be added.
            use_state flag may be set.
        """
        super().step(data)  # for logging

        # Process observations to extract 2D features
        observed_state = self.observations_to_comunication_protocol(data)

        # Add noise if specified
        if self.noise_params is not None and observed_state.use_state:
            observed_state = self.add_noise_to_sensor_data(observed_state)

        # Process all observations if explicitly requested (e.g., for testing)
        if self.process_all_obs:
            observed_state.use_state = True

        # Motor-only steps should not be passed to learning modules
        if self.motor_only_step:
            # Set interesting-features flag to False, as should not be passed to
            # LM, even in e.g. pre-training experiments that might otherwise do so
            observed_state.use_state = False

        return observed_state

    def observations_to_comunication_protocol(self, data) -> State:
        """Turn raw observations into State following CMP.

        This method uses the parent class's pose vector computation (surface normals
        and curvature directions) and additionally extracts 2D edge features from
        RGB images as supplementary non-morphological features.

        Args:
            data: Raw observations containing rgba, depth, semantic_3d, sensor_frame_data,
                world_camera, etc.

        Returns:
            State with proper 3D pose vectors and supplementary edge features.
        """
        # Extract basic data
        obs_3d = data["semantic_3d"]
        sensor_frame_data = data["sensor_frame_data"]
        world_camera = data["world_camera"]
        rgba_feat = data["rgba"]
        depth_feat = data["depth"].reshape(data["depth"].size, 1).astype(np.float64)

        # Calculate center coordinates
        center_row_col = rgba_feat.shape[0] // 2
        obs_dim = int(np.sqrt(obs_3d.shape[0]))
        half_obs_dim = obs_dim // 2
        center_id = half_obs_dim + obs_dim * half_obs_dim

        # Initialize features dictionary
        features = {}

        # Calculate object coverage if requested
        if "object_coverage" in self.features:
            features["object_coverage"] = sum(obs_3d[:, 3] > 0) / len(obs_3d[:, 3])

        # Get center location and check if on object
        obs_3d_center = obs_3d[center_id]
        x, y, z, semantic_id = obs_3d_center

        if semantic_id > 0:  # On object
            # Use parent's extract_and_add_features for proper pose vectors
            (
                features,
                morphological_features,
                invalid_signals,
            ) = self.extract_and_add_features(
                features,
                obs_3d,
                rgba_feat,
                depth_feat,
                center_id,
                center_row_col,
                sensor_frame_data,
                world_camera,
            )

            # Additionally extract edge features from RGB image
            edge_info = self.extract_2d_edge_features(rgba_feat, center_row_col)

            surface_normal = morphological_features["pose_vectors"][0]
            edge_confidence = 0.0
            edge_tangent_world = None
            if edge_info["has_edge"]:
                theta = edge_info["edge_orientation"]
                edge_tangent_camera = np.array([
                    np.cos(theta),
                    np.sin(theta),
                    0.0,
                ])

                edge_tangent_world = quaternion.rotate_vectors(
                    self.state["rotation"], edge_tangent_camera
                )

                normal_component = np.dot(edge_tangent_world, surface_normal)
                edge_tangent_world = edge_tangent_world - normal_component * surface_normal
                edge_tangent_world = self._normalize(edge_tangent_world)

                edge_confidence = edge_info["coherence"]

                if edge_tangent_world is not None:
                    self._maybe_initialize_canonical_frame(
                        edge_tangent_world, surface_normal, edge_confidence
                    )

            # Propagate canonical frame to pose vectors
            canonical_tangent = None
            canonical_perp = None
            if self._canonical_tangent is not None:
                canonical_tangent, canonical_perp = self._canonical_axes_for_normal(
                    surface_normal
                )
                if canonical_tangent is not None and canonical_perp is not None:
                    morphological_features["pose_vectors"] = np.vstack(
                        [
                            surface_normal,
                            canonical_tangent,
                            canonical_perp,
                        ]
                    )
                    morphological_features["pose_fully_defined"] = True

            # Add edge features to non_morphological_features
            if "edge_strength" in self.features:
                features["edge_strength"] = edge_info["edge_strength"]
            if "edge_orientation" in self.features:
                features["edge_orientation"] = edge_info["edge_orientation"]
            if "coherence" in self.features:
                features["coherence"] = edge_info["coherence"]
            features["pose_frame_confidence"] = float(
                edge_confidence if not invalid_signals else 0.0
            )
            self._pose_frame_confidence = features["pose_frame_confidence"]

            # Optionally expose edge tangent if requested
            if "edge_tangent" in self.features:
                if edge_tangent_world is not None:
                    features["edge_tangent"] = edge_tangent_world
                else:
                    features["edge_tangent"] = np.zeros(3)

            use_state = not invalid_signals
        else:
            # Not on object
            morphological_features = {}
            invalid_signals = True
            use_state = False

        # Add on_object to morphological features
        if "on_object" in self.features:
            morphological_features["on_object"] = float(semantic_id > 0)

        # Add standard color features if requested
        if "rgba" in self.features:
            features["rgba"] = rgba_feat[center_row_col, center_row_col]
        if "hsv" in self.features:
            from skimage.color import rgb2hsv

            rgba = rgba_feat[center_row_col, center_row_col]
            hsv = rgb2hsv(rgba[:3])
            features["hsv"] = hsv

        # Create and return State object
        observed_state = State(
            location=np.array([x, y, z]),
            morphological_features=morphological_features,
            non_morphological_features=features,
            confidence=1.0,
            use_state=bool(semantic_id > 0) and use_state,
            sender_id=self.sensor_module_id,
            sender_type="SM",
        )

        # Store for logging
        if not self.is_exploring:
            self.processed_obs.append(observed_state.__dict__)
            self.states.append(self.state)
            self.visited_locs.append(observed_state.location)
            if "pose_vectors" in morphological_features:
                self.visited_normals.append(morphological_features["pose_vectors"][0])
            else:
                self.visited_normals.append(None)

        return observed_state

    def _draw_2d_pose_on_patch(
        self,
        patch,
        edge_direction,
        tangent_color=(255, 255, 0),
        normal_color=(0, 255, 255),
        arrow_length=20,
    ):
        """Draw both tangent and normal arrows to show 2D pose.

        Args:
            patch: RGB patch of shape (H, W, 3)
            edge_direction: Edge tangent direction in radians
            tangent_color: RGB color for tangent arrow (default: yellow)
            normal_color: RGB color for normal arrow (default: cyan)
            arrow_length: Length of arrows in pixels

        Returns:
            Patch with 2D pose arrows drawn on it
        """
        # Create a copy to avoid modifying original
        patch_with_pose = patch.copy()

        # Center of patch
        center_y, center_x = patch.shape[0] // 2, patch.shape[1] // 2

        # Tangent arrow (edge direction)
        tangent_end_x = int(center_x + arrow_length * np.cos(edge_direction))
        tangent_end_y = int(center_y + arrow_length * np.sin(edge_direction))

        # Normal arrow (perpendicular to edge, 90 degree rotation)
        normal_direction = edge_direction + np.pi / 2
        normal_length = arrow_length * 0.7  # Slightly shorter
        normal_end_x = int(center_x + normal_length * np.cos(normal_direction))
        normal_end_y = int(center_y + normal_length * np.sin(normal_direction))

        # Draw tangent arrow (edge direction)
        cv2.arrowedLine(
            patch_with_pose,
            (center_x, center_y),
            (tangent_end_x, tangent_end_y),
            tangent_color,
            thickness=3,
            tipLength=0.3,
        )

        # Draw normal arrow (perpendicular)
        cv2.arrowedLine(
            patch_with_pose,
            (center_x, center_y),
            (normal_end_x, normal_end_y),
            normal_color,
            thickness=3,
            tipLength=0.3,
        )

        # Draw center point
        cv2.circle(patch_with_pose, (center_x, center_y), 4, (255, 255, 255), -1)

        return patch_with_pose

    def extract_2d_edge_features(self, rgba_image, center_coord):
        """Extract 2D edge features from RGB image using structure tensor.

        This method applies the enhanced structure tensor method to detect edges
        and their orientation in the input image.

        Args:
            rgba_image: rgba patch of size (64, 64, 4)
            center_coord: Center coordinate (row, col) for patch extraction

        Returns:
            Dictionary containing:
                - has_edge: Boolean indicating if edge is detected
                - edge_orientation: Edge tangent angle in radians [0, 2π)
                - edge_strength: Magnitude of edge strength
                - coherence: Edge quality metric [0, 1]
                - edge_curvature: Curvature estimate (0.0 for structure tensor)
        """
        # Convert RGBA to RGB if needed
        if rgba_image.shape[2] == 4:
            patch = rgba_image[:, :, :3]
        else:
            patch = rgba_image

        # Apply structure tensor edge detection
        win_sigma = self.edge_params.get("gaussian_sigma", 1.0)
        ksize = 7  # Standard kernel size for structure tensor

        edge_strength, coherence, edge_orientation = structure_tensor_center(
            patch, win_sigma=win_sigma, ksize=ksize
        )

        # Determine if edge exists based on strength and coherence thresholds
        strength_threshold = self.edge_params.get("edge_threshold", 0.1)
        coherence_threshold = 0.05  # Minimum coherence for edge-like structure

        has_edge = (edge_strength > strength_threshold) and (
            coherence > coherence_threshold
        )

        # Build and return edge info dictionary
        edge_info = {
            "has_edge": has_edge,
            "edge_orientation": edge_orientation,
            "edge_strength": edge_strength,
            "coherence": coherence,
        }

        # Debug visualization: save patch with edge arrows
        if self.debug_visualize and has_edge:
            # Convert RGBA to RGB for visualization
            if rgba_image.shape[2] == 4:
                rgb_patch = rgba_image[:, :, :3]
            else:
                rgb_patch = rgba_image

            # Draw edge arrows on patch
            patch_with_arrows = self._draw_2d_pose_on_patch(
                rgb_patch.copy(), edge_orientation
            )

            # Create filename
            filename = (
                f"ep{self.episode_counter:04d}_"
                f"step{self.step_counter:04d}_"
                f"{self.debug_counter:04d}.png"
            )
            filepath = self.debug_save_dir / filename

            # Save image
            plt.imsave(filepath, patch_with_arrows)

            # Increment counters
            self.step_counter += 1
            self.debug_counter += 1

        return edge_info

    # ---------------------------------------------------------------------
    # Canonical frame helpers
    # ---------------------------------------------------------------------

    def _reset_canonical_frame(self):
        """Clear cached canonical frame information."""
        self._canonical_tangent = None
        self._canonical_perpendicular = None
        self._canonical_normal = None
        self._pose_frame_confidence = 0.0

    @staticmethod
    def _normalize(vector):
        """Return unit vector or None if norm is ~0."""
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            return None
        return vector / norm

    def _rotation_between(self, source, target):
        """Quaternion rotating source vector onto target vector."""
        src = self._normalize(source)
        tgt = self._normalize(target)
        if src is None or tgt is None:
            return quaternion.one

        dot_val = np.clip(np.dot(src, tgt), -1.0, 1.0)
        if dot_val > 1.0 - 1e-6:
            return quaternion.one
        if dot_val < -1.0 + 1e-6:
            # 180-degree rotation: choose an arbitrary orthogonal axis
            axis = np.cross(src, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(src, np.array([0.0, 1.0, 0.0]))
            axis = self._normalize(axis)
            angle = np.pi
            return quaternion.from_rotation_vector(axis * angle)

        axis = self._normalize(np.cross(src, tgt))
        angle = np.arccos(dot_val)
        return quaternion.from_rotation_vector(axis * angle)

    def _maybe_initialize_canonical_frame(self, tangent, surface_normal, confidence):
        """Cache canonical tangent/perpendicular if not set and confidence high."""
        if self._canonical_tangent is not None:
            return

        if confidence < self._edge_confidence_threshold:
            return

        tangent = self._normalize(tangent - np.dot(tangent, surface_normal) * surface_normal)
        if tangent is None:
            return

        perpendicular = np.cross(surface_normal, tangent)
        perpendicular = self._normalize(perpendicular)
        if perpendicular is None:
            return

        self._canonical_tangent = tangent
        self._canonical_perpendicular = perpendicular
        self._canonical_normal = self._normalize(surface_normal)

    def _canonical_axes_for_normal(self, surface_normal):
        """Return tangent/perpendicular aligned with cached canonical frame."""
        if self._canonical_tangent is None or self._canonical_normal is None:
            return None, None

        rot = self._rotation_between(self._canonical_normal, surface_normal)
        tangent = quaternion.rotate_vectors(rot, self._canonical_tangent)
        perpendicular = quaternion.rotate_vectors(rot, self._canonical_perpendicular)

        tangent = self._normalize(
            tangent - np.dot(tangent, surface_normal) * surface_normal
        )
        perpendicular = self._normalize(perpendicular)

        if tangent is None or perpendicular is None:
            return None, None

        return tangent, perpendicular

    def state_dict(self):
        """Return state_dict for logging."""
        return {
            **super().state_dict(),
            "processed_observations": self.processed_obs,
        }
