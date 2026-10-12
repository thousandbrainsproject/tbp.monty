# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from skimage.color import rgb2hsv
from typing_extensions import Self

from tbp.monty.cmp import Message
from tbp.monty.frameworks.utils.sensor_processing import (
    log_sign,
    principal_curvatures,
    scale_clip,
    surface_normal_naive,
    surface_normal_ordinary_least_squares,
    surface_normal_total_least_squares,
)
from tbp.monty.sensor_modules.sensor_module import (
    Payload,
    Transform,
    TransformContext,
)

__all__ = [
    "Default",
    "SurfaceNormalMethod",
    "UnknownFeature",
]

logger = logging.getLogger(__name__)


class SurfaceNormalMethod(Enum):
    TLS = "TLS"
    """Total Least-Squares"""
    OLS = "OLS"
    """Ordinary Least-Squares"""
    NAIVE = "naive"
    """Naive"""


class Default(Transform):
    """Extracts default features from a SensorObservation."""

    CURVATURE_FEATURES: ClassVar[list[str]] = [
        "principal_curvatures",
        "principal_curvatures_log",
        "gaussian_curvature",
        "mean_curvature",
        "gaussian_curvature_sc",
        "mean_curvature_sc",
        "curvature_for_TM",
    ]

    POSSIBLE_FEATURES: ClassVar[list[str]] = [
        "on_object",
        "object_coverage",
        "min_depth",
        "mean_depth",
        "rgba",
        "hsv",
        "pose_vectors",
        "principal_curvatures",
        "principal_curvatures_log",
        "pose_fully_defined",
        "gaussian_curvature",
        "mean_curvature",
        "gaussian_curvature_sc",
        "mean_curvature_sc",
        "curvature_for_TM",
        "coords_for_TM",
        "edge_strength",
        "coherence",
    ]

    _features: list[str]
    _is_surface_sm: bool
    _pc1_is_pc2_threshold: int
    _sensor_module_id: str
    _surface_normal_method: SurfaceNormalMethod
    _weight_curvature: bool

    def __init__(
        self: Self,
        features: list[str],
        sensor_module_id: str,
        pc1_is_pc2_threshold: int = 10,
        surface_normal_method: SurfaceNormalMethod = SurfaceNormalMethod.TLS,
        weight_curvature: bool = True,
        is_surface_sm: bool = False,
    ) -> None:
        """Initializes the Default extractor.

        Args:
            features: List of features to extract. Must be a subset of
                POSSIBLE_FEATURES.
            sensor_module_id: ID of sensor module.
            pc1_is_pc2_threshold: Maximum difference between pc1 and pc2 to be
                classified as being roughly the same (ignore curvature directions).
                Defaults to 10.
            surface_normal_method: Method to use for surface normal extraction. Defaults
              to TLS.
            weight_curvature: Whether to use the weighted implementation for principal
                curvature extraction (True) or unweighted (False). Defaults to True.
            is_surface_sm: Surface SMs do not require that the central pixel is
                "on object" in order to process the observation (i.e., extract
                features). Defaults to False.

        Raises:
            UnknownFeature: If an unknown feature is encountered.
        """
        for feature in features:
            if feature not in self.POSSIBLE_FEATURES:
                raise UnknownFeature(f"{feature} not part of {self.POSSIBLE_FEATURES}")
        self._features = features
        self._is_surface_sm = is_surface_sm
        self._pc1_is_pc2_threshold = pc1_is_pc2_threshold
        self._sensor_module_id = sensor_module_id
        self._surface_normal_method = surface_normal_method
        self._weight_curvature = weight_curvature

    def __call__(
        self: Self,
        ctx: TransformContext,  # noqa: ARG002
        payload: Payload,
    ) -> Payload:
        obs_3d = payload.observation["semantic_3d"]
        sensor_frame_data = payload.observation["sensor_frame_data"]
        cam_to_world = payload.observation["cam_to_world"]
        rgba_feat = payload.observation["rgba"]
        depth_feat = (
            payload.observation["depth"]
            .reshape(payload.observation["depth"].size, 1)
            .astype(np.float64)
        )
        # Assuming squared patches
        center_row_col = rgba_feat.shape[0] // 2
        # Calculate center ID for flat semantic obs
        obs_dim = int(np.sqrt(obs_3d.shape[0]))
        half_obs_dim = obs_dim // 2
        center_id = half_obs_dim + obs_dim * half_obs_dim
        # Extract all specified features
        features = {}
        if "object_coverage" in self._features:
            # Last dimension is semantic ID (integer >0 if on any object)
            features["object_coverage"] = sum(obs_3d[:, 3] > 0) / len(obs_3d[:, 3])
            assert features["object_coverage"] <= 1.0, (
                "Coverage cannot be greater than 100%"
            )

        x, y, z, semantic_id = obs_3d[center_id]
        on_object = semantic_id > 0
        if on_object or (self._is_surface_sm and features["object_coverage"] > 0):
            (
                features,
                morphological_features,
                valid_signals,
            ) = self._extract_and_add_features(
                features,
                obs_3d,
                rgba_feat,
                depth_feat,
                center_id,
                center_row_col,
                sensor_frame_data,
                cam_to_world,
            )
        else:
            valid_signals = False
            morphological_features = {}

        if "on_object" in self._features:
            morphological_features["on_object"] = float(on_object)

        # Sensor module returns features at a location in the form of a Message class.
        # use_state is a bool indicating whether the input is "interesting",
        # which indicates that it merits processing by the learning module; by default
        # it will always be True so long as the surface normal and principal curvature
        # directions were valid; certain SMs and policies used separately can also set
        # it to False under appropriate conditions

        payload.percept = Message(
            location=np.array([x, y, z]),
            morphological_features=morphological_features,
            non_morphological_features=features,
            confidence=1.0,
            use_state=on_object and valid_signals,
            sender_id=self._sensor_module_id,
            sender_type="SM",
        )
        # This is just for logging! Do not use _ attributes for matching
        payload.percept._semantic_id = semantic_id

        return payload

    def _extract_and_add_features(
        self,
        features: dict[str, Any],
        obs_3d: npt.NDArray[np.int_],
        rgba_feat: npt.NDArray[np.uint8],
        depth_feat: npt.NDArray[np.float64],
        center_id: int,
        center_row_col: int,
        sensor_frame_data: npt.NDArray[np.int_],
        cam_to_world: npt.NDArray[np.float64],
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        """Extract features configured for extraction from sensor patch.

        Returns the features in the patch, and True if the surface normal
        and principal curvature directions are well-defined.

        Returns:
            features: The features in the patch.
            morphological_features: ?
            valid_signals: True if the surface normal and principal curvature
                directions are well-defined.
        """
        # ------------ Extract Morphological Features ------------
        # Get surface normal for graph matching with features
        surface_normal, valid_sn = self._get_surface_normals(
            obs_3d, sensor_frame_data, center_id, cam_to_world
        )

        k1, k2, dir1, dir2, valid_pc = principal_curvatures(
            obs_3d, center_id, surface_normal, weighted=self._weight_curvature
        )
        # TODO: test using log curvatures instead
        if np.abs(k1 - k2) < self._pc1_is_pc2_threshold:
            pose_fully_defined = False
        else:
            pose_fully_defined = True

        morphological_features: dict[str, Any] = {
            "pose_vectors": np.vstack(
                [
                    surface_normal,
                    dir1,
                    dir2,
                ]
            ),
            "pose_fully_defined": pose_fully_defined,
        }
        # ---------- Extract Optional, Non-Morphological Features ----------
        if "rgba" in self._features:
            features["rgba"] = rgba_feat[center_row_col, center_row_col]
        if "min_depth" in self._features:
            features["min_depth"] = np.min(depth_feat[obs_3d[:, 3] != 0])
        if "mean_depth" in self._features:
            features["mean_depth"] = np.mean(depth_feat[obs_3d[:, 3] != 0])
        if "hsv" in self._features:
            rgba = rgba_feat[center_row_col, center_row_col]
            hsv = rgb2hsv(rgba[:3])
            features["hsv"] = hsv

        # Note we only determine curvature if we could determine a valid surface normal
        if any(feat in self.CURVATURE_FEATURES for feat in self._features) and valid_sn:
            if valid_pc:
                # Only process the below features if the principal curvature was valid,
                # and therefore we have a defined k1, k2 etc.
                if "principal_curvatures" in self._features:
                    features["principal_curvatures"] = np.array([k1, k2])

                if "principal_curvatures_log" in self._features:
                    features["principal_curvatures_log"] = log_sign(np.array([k1, k2]))

                if "gaussian_curvature" in self._features:
                    features["gaussian_curvature"] = k1 * k2

                if "mean_curvature" in self._features:
                    features["mean_curvature"] = (k1 + k2) / 2

                if "gaussian_curvature_sc" in self._features:
                    gc = k1 * k2
                    gc_scaled_clipped = scale_clip(gc, 4096)
                    features["gaussian_curvature_sc"] = gc_scaled_clipped

                if "mean_curvature_sc" in self._features:
                    mc = (k1 + k2) / 2
                    mc_scaled_clipped = scale_clip(mc, 256)
                    features["mean_curvature_sc"] = mc_scaled_clipped
        else:
            # Flag that PC directions are non-meaningful for e.g. downstream motor
            # policies
            features["pose_fully_defined"] = False

        valid_signals = valid_sn and valid_pc
        if not valid_signals:
            logger.debug("Either the surface-normal or pc-directions were ill-defined")

        return features, morphological_features, valid_signals

    def _get_surface_normals(
        self,
        obs_3d: npt.NDArray[np.int_],
        sensor_frame_data: npt.NDArray[np.int_],
        center_id: int,
        cam_to_world: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], bool]:
        if self._surface_normal_method == SurfaceNormalMethod.TLS:
            surface_normal, valid_sn = surface_normal_total_least_squares(
                obs_3d, center_id, cam_to_world[:3, 2]
            )
        elif self._surface_normal_method == SurfaceNormalMethod.OLS:
            surface_normal, valid_sn = surface_normal_ordinary_least_squares(
                sensor_frame_data, cam_to_world, center_id
            )
        elif self._surface_normal_method == SurfaceNormalMethod.NAIVE:
            surface_normal, valid_sn = surface_normal_naive(
                obs_3d, patch_radius_frac=2.5
            )
        else:
            raise ValueError(
                f"surface_normal_method must be in [{SurfaceNormalMethod.TLS} (default)"
                f", {SurfaceNormalMethod.OLS}, {SurfaceNormalMethod.NAIVE}]."
            )

        return surface_normal, valid_sn


class UnknownFeature(ValueError):
    """Raised when an unknown feature is encountered."""

    pass
