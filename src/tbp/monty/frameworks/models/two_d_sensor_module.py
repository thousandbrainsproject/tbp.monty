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
from skimage.color import rgb2hsv

from tbp.monty.frameworks.models.sensor_modules import DetailedLoggingSM, NoiseMixin
from tbp.monty.frameworks.models.states import State
from tbp.monty.frameworks.utils.edge_detection_utils import structure_tensor_center

logger = logging.getLogger(__name__)

def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > eps else v * 0.0

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
            "coherence",
            "rgba",
            "hsv",
            "pose_from_edge",
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
        self._pose_frame_confidence = 0.0

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

    def edge_angle_to_3d_tangent(
        self, theta, normal, *, normal_frame="world", world_camera=None, out_frame="world"
    ):
        """
        Lift the edge angle (theta from the 2D image x-axis) into a 3D tangent vector
        on the surface, aligned with the camera's image axes.
        """
        n = normal / np.linalg.norm(normal)

        R_wc = None
        if world_camera is not None:
            R_wc = (
                world_camera[:3, :3] if world_camera.shape == (4, 4) else world_camera
            )
        R_cw = R_wc.T if R_wc is not None else None

        # Put normal in camera frame to build a basis consistent with image axes
        if normal_frame == "camera":
            n_cam = n
        elif normal_frame == "world":
            if R_cw is None:
                raise ValueError(
                    "world_camera must be provided if normal_frame is 'world'"
                )
            n_cam = R_cw @ n
        else:
            raise ValueError(f"Invalid normal_frame: {normal_frame}")

        n_cam = n_cam / np.linalg.norm(n_cam)
        ex = np.array([1.0, 0.0, 0.0])
        ey = np.array([0.0, 1.0, 0.0])

        def project_tangent(v, n_):
            return v - np.dot(v, n_) * n_

        tx = project_tangent(ex, n_cam)
        ty = project_tangent(ey, n_cam)

        if np.linalg.norm(tx) < 1e-12:
            ez = np.array([0.0, 0.0, 1.0])
            tx = project_tangent(ez, n_cam)
        if np.linalg.norm(ty) < 1e-12:
            ez = np.array([0.0, 0.0, 1.0])
            ty = project_tangent(ez, n_cam)

        tx = tx / np.linalg.norm(tx)
        ty = ty / np.linalg.norm(ty)

        t_cam = np.cos(theta) * tx + np.sin(theta) * ty
        t_cam = t_cam / np.linalg.norm(t_cam)

        if out_frame == "camera":
            return t_cam
        elif out_frame == "world":
            if R_wc is None:
                raise ValueError(
                    "world_camera must be provided if out_frame is 'world'"
                )
            t_world = R_wc @ t_cam
            t_world = t_world / np.linalg.norm(t_world)
            return t_world
        else:
            raise ValueError(f"Invalid out_frame: {out_frame}")

    def observations_to_comunication_protocol(self, data) -> State:
        """Turn raw observations into State following CMP."""
        obs_3d = data["semantic_3d"]
        sensor_frame_data = data["sensor_frame_data"]
        rgba_feat = data["rgba"]
        depth_feat = data["depth"]
        world_camera = data["world_camera"]

        # Center pixel
        H = int(rgba_feat.shape[0])
        center_row_col = H // 2

        obs_dim = int(np.sqrt(obs_3d.shape[0]))
        half_obs_dim = obs_dim // 2
        center_id = half_obs_dim + obs_dim * half_obs_dim

        # Extract features
        features = {}

        # Object coverage
        if "object_coverage" in self.features:
            # prevent division by zero
            on_obj = obs_3d[:, 3] > 0
            denominator = max(len(on_obj), 1)
            features["object_coverage"] = float(np.count_nonzero(on_obj) / denominator)

        # Center 3D @ semantic
        x, y, z, semantic_id = obs_3d[center_id]
        on_object_center = semantic_id > 0

        # 3D Features
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
        # Defaults
        surface_normal = None
        edge_tangent = None
        edge_perp = None
        pose_fully_defined = False

        # If we are on object, try to compute edge-based pose vectors
        if on_object_center and "pose_vectors" in morphological_features:
            surface_normal = _normalize(morphological_features["pose_vectors"][0])

            # Edge detection at the center
            edge_info = self.extract_2d_edge_features(rgba_feat, center_row_col)
            has_edge = edge_info["has_edge"]

            if has_edge:
                theta = edge_info["edge_orientation"] # radians, image x = right, y=down
                edge_tangent = self.edge_angle_to_3d_tangent(
                    theta,
                    surface_normal,
                    normal_frame="world",
                    world_camera=world_camera,
                    out_frame="world",
                )
                edge_tangent = _normalize(edge_tangent)
                edge_perp = _normalize(np.cross(surface_normal, edge_tangent))

                pose_fully_defined = True

        # Build morphological features payload
        if edge_tangent is not None and edge_perp is not None and surface_normal is not None:
            morphological_features = {
                "pose_vectors": np.vstack(
                    [
                        surface_normal,
                        edge_tangent,
                        edge_perp,
                    ]
                ),
                "pose_fully_defined": pose_fully_defined,
                "pose_from_edge": True,
            }
            use_state = True

        if "pose_from_edge" not in morphological_features:
            morphological_features["pose_from_edge"] = False

        # Extras
        if "on_object" in self.features:
            morphological_features["on_object"] = float(semantic_id > 0)

        if "rgba" in self.features:
            features["rgba"] = rgba_feat[center_row_col, center_row_col]

        if "hsv" in self.features:
            from skimage.color import rgb2hsv

            rgba = rgba_feat[center_row_col, center_row_col]
            hsv = rgb2hsv(rgba[:3])
            features["hsv"] = hsv

        # Assemble state
        observed_state = State(
            location=np.array([x, y, z]),
            morphological_features=morphological_features,
            non_morphological_features=features,
            confidence=1.0,
            use_state=bool(on_object_center and not invalid_signals),
            sender_id=self.sensor_module_id,
            sender_type="SM",
        )

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
        edge_direction=None,
        label_text=None,
        tangent_color=(255, 255, 0),
        normal_color=(0, 255, 255),
        arrow_length=20,
    ):
        """Draw tangent/normal arrows and overlay debug text for a patch.

        Args:
            patch: RGB patch of shape (H, W, 3).
            edge_direction: Edge tangent direction in radians, if available.
            label_text: Text to overlay for debugging (e.g., angle or "No Edge").
            tangent_color: RGB color for tangent arrow (default: yellow).
            normal_color: RGB color for normal arrow (default: cyan).
            arrow_length: Length of arrows in pixels.

        Returns:
            Patch with annotations drawn on it.
        """
        patch_with_pose = patch.copy()
        center_y, center_x = patch.shape[0] // 2, patch.shape[1] // 2

        # Draw pose arrows only if we have an edge direction
        if edge_direction is not None:

            tangent_end_x = int(center_x + arrow_length * np.cos(edge_direction))
            tangent_end_y = int(center_y + arrow_length * np.sin(edge_direction))

            normal_direction = edge_direction + np.pi / 2
            normal_length = arrow_length * 0.7
            normal_end_x = int(center_x + normal_length * np.cos(normal_direction))
            normal_end_y = int(center_y + normal_length * np.sin(normal_direction))

            cv2.arrowedLine(
                patch_with_pose,
                (center_x, center_y),
                (tangent_end_x, tangent_end_y),
                tangent_color,
                thickness=3,
                tipLength=0.3,
            )

            cv2.arrowedLine(
                patch_with_pose,
                (center_x, center_y),
                (normal_end_x, normal_end_y),
                normal_color,
                thickness=3,
                tipLength=0.3,
            )

        # Highlight center pixel source for edge extraction
        cv2.circle(patch_with_pose, (center_x, center_y), 3, (255, 0, 0), -1)

        if label_text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.35
            thickness = 1
            color = (255, 255, 255)
            margin = 3

            (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

            x = patch_with_pose.shape[1] - text_width - margin
            y = text_height + margin

            # Draw background rectangle for readability
            cv2.rectangle(
                patch_with_pose,
                (x - margin, y - text_height - margin),
                (x + text_width + margin, y + margin // 2),
                (0, 0, 0),
                thickness=-1,
            )

            cv2.putText(
                patch_with_pose,
                label_text,
                (x, y),
                font,
                font_scale,
                color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

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

        # Debug visualization: always save annotated patches when enabled
        if self.debug_visualize:
            if rgba_image.shape[2] == 4:
                rgb_patch = rgba_image[:, :, :3]
            else:
                rgb_patch = rgba_image

            if has_edge:
                angle_deg = np.degrees(edge_orientation)
                label_text = f"{angle_deg:.1f}"
                patch_with_debug = self._draw_2d_pose_on_patch(
                    rgb_patch.copy(), edge_orientation, label_text
                )
            else:
                patch_with_debug = self._draw_2d_pose_on_patch(
                    rgb_patch.copy(), None, "No edge"
                )

            filename = (
                f"ep{self.episode_counter:04d}_"
                f"step{self.step_counter:04d}_"
                f"{self.debug_counter:04d}.png"
            )
            filepath = self.debug_save_dir / filename

            plt.imsave(filepath, patch_with_debug)

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
        self._pose_frame_confidence = 0.0

    @staticmethod
    def _normalize(vector):
        """Return unit vector or None if norm is ~0."""
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            return None
        return vector / norm

    def state_dict(self):
        """Return state_dict for logging."""
        return {
            **super().state_dict(),
            "processed_observations": self.processed_obs,
        }
