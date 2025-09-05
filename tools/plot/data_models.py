# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Data models and coordinate transformations for hypothesis visualization."""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation


class ObjectModelForVisualization:
    def __init__(
        self,
        points: np.ndarray,
        features: dict[str, np.ndarray],
        model_origin_wrt_world: np.ndarray,
        target_location_wrt_world: np.ndarray,
        target_orientation_wrt_world: np.ndarray,
    ):
        """Initialize ObjectModel and transform from model to world coordinates.

        This class takes the learned object model points and transforms them to be in
        the location and orientation defined by the ground-truth pose of the object in
        the world.

        Args:
            points: Stored points of the object model (n_points, 3).
            features: Stored features of the object model.

            The below arguments are used to transform the model to world coordinates.

            model_origin_wrt_world: Location at which model was learned in world frame.
                Currently in Monty, this is always [0, 1.5, 0] for all objects.
            target_location_wrt_world: Ground truth location of object in inference.
                Currently in Monty, this is always [0, 1.5, 0] for all objects.
            target_orientation_wrt_world: Ground truth orientation of object in
                inference. In Monty, this could be a random test rotation that
                differs from the canonical learning orientation.
        """
        self.object_points_wrt_model = points

        for key, value in features.items():
            setattr(self, key, np.asarray(value))

        self.model_origin_wrt_world = model_origin_wrt_world
        self.target_location_wrt_world = target_location_wrt_world
        self.target_orientation_wrt_world = target_orientation_wrt_world

        self.object_points_wrt_world = transform_locations_model_to_world(
            self.object_points_wrt_model,
            model_origin_wrt_world,
            target_location_wrt_world,
            target_orientation_wrt_world,
        )

        # Reshape pose_vectors from (n_points, 9) to (n_points, 3, 3).
        # The original pose_vectors are stored as flattened row-major matrices
        # [surface_normal, dir1, dir2]. Since we flatten and reshape using the same
        # order ("C"), this preserves the row-major format without changing structure.
        object_feature_orientations_wrt_model = np.reshape(
            features["pose_vectors"], (-1, 3, 3)
        )
        self.object_feature_orientations_wrt_world = (
            transform_orientations_model_to_world(
                object_feature_orientations_wrt_model,
                self.target_orientation_wrt_world,
                orientations_in_row_vector_format=True,
            )
        )

    @property
    def x(self) -> np.ndarray:
        return self.object_points_wrt_model[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.object_points_wrt_model[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self.object_points_wrt_model[:, 2]

    @property
    def kd_tree(self) -> KDTree:
        return KDTree(self.object_points_wrt_world, leafsize=40)

    def __repr__(self) -> str:
        """Return a detailed string representation of the ObjectModel."""
        n_points = len(self.object_points_wrt_model)

        feature_names = [
            attr
            for attr in dir(self)
            if not attr.startswith("_") and not callable(getattr(self, attr))
        ]

        return f"ObjectModel({n_points=!s}, {feature_names=!s})"


def transform_locations_model_to_world(
    locations_wrt_model: np.ndarray,
    model_origin_wrt_world: np.ndarray,
    target_location_wrt_world: np.ndarray,
    target_orientation_wrt_world: np.ndarray | Rotation,
) -> np.ndarray:
    """Transform locations from model to world coordinates.

    To transform from model to world coordinates, we:
    1. Center the locations to [0, 0, 0] by subtracting the model origin ([0, 1.5, 0])
    2. Rotate the locations by the object's ground truth orientation during inference
    3. Translate the locations back to object's ground truth location during inference

    Args:
        locations_wrt_model: Locations to transform (n_points, 3).
            Can be points of object model or hypotheses' locations.
        model_origin_wrt_world: Location at which model was learned in world frame.
            Currently in Monty, this is always [0, 1.5, 0] for all objects.
        target_location_wrt_world: Ground truth location of object in inference.
            Currently in Monty, this is always [0, 1.5, 0] for all objects.
        target_orientation_wrt_world: Ground truth orientation of object in inference.
            Currently in Monty, this could be a random test rotation.

    Returns:
        Transformed locations in world frame.
    """
    if not isinstance(target_orientation_wrt_world, Rotation):
        target_orientation_wrt_world = Rotation.from_euler(
            "xyz", target_orientation_wrt_world, degrees=True
        )

    centered_locations = locations_wrt_model - model_origin_wrt_world
    rotated_locations = target_orientation_wrt_world.apply(centered_locations)
    locations_wrt_world = rotated_locations + target_location_wrt_world

    return locations_wrt_world


def transform_orientations_model_to_world(
    orientations_wrt_model: np.ndarray,
    target_rotation_wrt_world: Rotation,
    orientations_in_row_vector_format: bool = False,
) -> np.ndarray:
    """Transform orientation matrices from model to world coordinates.

    In linear algebra, to rotate matrix (or tensor) A by rotation matrix R:

    1. If A consists of matrices composed of column vectors, then:
        R (shape: (3, 3)) @ A (shape: (N, 3, 3)) = B (shape: (N, 3, 3))
            by broadcasting.
    2. If A consists of matrices composed of row vectors, then:
        A (shape: (N, 3, 3)) @ R.T (shape: (3, 3)) = B (shape: (N, 3, 3))
            by broadcasting.
        Note that this is equivalent to transposing A first, multiplying, then
            transposing back the result, i.e.(R @ A.T).T = A @ R.T

    Args:
        orientations_wrt_model: Orientations to transform, with shape
            (n_points or n_hypotheses, 3, 3).
        target_rotation_wrt_world: Target orientation represented as Euler angles
            (xyz, degrees) using scipy.spatial.transform.Rotation.
        orientations_in_row_vector_format: Whether the orientations_wrt_model
            parameter uses row vector format. This only affects how orientations_wrt_model
            is interpreted, not target_rotation_wrt_world. True for pose_vectors stored
            in ObjectModel, which is a 3x3 matrix of stacked [surface_normal, pc1, pc2]
            row vectors for each point. False for hypothesized object orientations,
            which are regular 3x3 rotation matrices following column-vector convention.

    Returns:
        Transformed orientations in world frame.
    """
    target_rotation_wrt_world = Rotation.from_euler(
        "xyz", target_rotation_wrt_world, degrees=True
    ).as_matrix()

    if orientations_in_row_vector_format:
        return orientations_wrt_model @ target_rotation_wrt_world.T
    else:
        return target_rotation_wrt_world @ orientations_wrt_model
