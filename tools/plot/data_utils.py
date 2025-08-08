# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Data models and loading utilities for hypothesis visualization."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path


import numpy as np
import torch
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R

from tbp.monty.frameworks.utils.graph_matching_utils import get_relevant_curvature

logger = logging.getLogger(__name__)


def apply_world_transform(
    locations: np.ndarray,
    rotation_matrices: np.ndarray,
    learned_position: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray | R,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply world frame transformation to locations and rotation matrices.

    Args:
        locations: Locations to transform (n_points, 3).
        rotation_matrices: Rotation matrices to transform (n_points, 3, 3).
        learned_position: The position where locations were learned.
        target_position: Target position in world frame.
        target_rotation: Target rotation (Euler angles in degrees or Rotation object).

    Returns:
        Tuple of (transformed_locations, transformed_rotations).
    """
    if isinstance(target_rotation, R):
        rot = target_rotation
    else:
        rot = R.from_euler("xyz", target_rotation, degrees=True)

    # Transform locations
    transformed_locations = locations - learned_position
    transformed_locations = rot.apply(transformed_locations)
    transformed_locations = transformed_locations + target_position

    # Transform rotation matrices
    rot_matrix = rot.as_matrix()
    transformed_rotations = np.zeros_like(rotation_matrices)
    for i in range(len(rotation_matrices)):
        transformed_rotations[i] = rot_matrix @ rotation_matrices[i]

    return transformed_locations, transformed_rotations


def deserialize_json_chunks_fast(json_file, start=0, stop=None, episodes=None):
    """Optimized version of deserialize_json_chunks.

    Performance improvements:
    - Uses set for O(1) episode lookups instead of O(n) list checks
    - Avoids redundant string conversions by pre-computing keys
    - Uses float('inf') instead of np.inf to avoid numpy import overhead
    - Direct value extraction without intermediate list creation
    - More efficient line filtering logic

    Args:
        json_file: full path to the json file to load
        start: int, get data starting at this episode
        stop: int, get data ending at this episode, not inclusive
        episodes: iterable of ints with episodes to pull

    Returns:
        detailed_json: dict containing contents of file
    """
    # Pre-process episodes for faster lookup
    if episodes is not None:
        episodes_set = set(episodes)
        str_episodes = [str(i) for i in episodes]
    else:
        episodes_set = None
        str_episodes = None

    detailed_json = {}
    stop = float("inf") if stop is None else stop

    with open(json_file, "r") as f:
        for line_counter, line in enumerate(f):
            # Fast filtering based on episode criteria
            if episodes_set is not None:
                if line_counter not in episodes_set:
                    continue
            elif not (start <= line_counter < stop):
                continue

            # Parse JSON and extract value directly
            tmp_json = json.loads(line)
            # Use next(iter()) to get the first value without creating a list
            detailed_json[str(line_counter)] = next(iter(tmp_json.values()))

    # Validation check if episodes were specified
    if str_episodes is not None:
        if list(detailed_json.keys()) != str_episodes:
            logger.warning(
                "Episode keys did not equal json keys. This can happen if "
                "json file was not appended to in episode order. To manually load the "
                "whole file for debugging, run `deserialize_json_chunks_fast(my_file)` with "
                "no further arguments"
            )

    return detailed_json


class ObjectModel:
    """Mutable wrapper for object models.

    Args:
        pos (ArrayLike): The points of the object model as a sequence of points
          (i.e., has shape (n_points, 3)).
        features (Optional[Mapping]): The features of the object model. For
          convenience, the features become attributes of the ObjectModel instance.
    """

    def __init__(
        self,
        locations: np.ndarray,
        features: dict[str, np.ndarray],
        learned_position: np.ndarray,
        target_position: np.ndarray,
        target_rotation: np.ndarray | R,
    ):
        """Initialize ObjectModel, transforming from learned to world frame.

        Args:
            locations: The points of the object model (n_points, 3).
            features: Features of the object model.
            learned_position: The learned position to transform from to world frame.
            target_position: Target position in world frame for transformation.
            target_rotation: Target rotation (Euler angles in degrees or Rotation object).
        """
        orientation_vectors = features["pose_vectors"] # (n_points, 9)
        n_points = len(orientation_vectors)
        orientation_matrices = orientation_vectors.reshape(n_points, 3, 3)

        transformed_locations, transformed_pose_matrices = apply_world_transform(
            locations,
            orientation_matrices,
            learned_position,
            target_position,
            target_rotation,
        )

        self.locations = np.asarray(transformed_locations, dtype=float)
        features["pose_vectors"] = transformed_pose_matrices.reshape(n_points, 9)

        for key, value in features.items():
            setattr(self, key, np.asarray(value))

    @property
    def x(self) -> np.ndarray:
        return self.pos[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.pos[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self.pos[:, 2]

    @property
    def kd_tree(self) -> KDTree:
        return KDTree(self.locations, leafsize=40)


def get_pretrained_model_path(model_type: str) -> Path:
    """Get the path to the pretrained model for an experiment.

    Args:
        model_type: The type of model to load ("dist" or "surf").

    Returns:
        The path to the pretrained model.

    Raises:
        ValueError: If the experiment name does not contain 'dist' or 'surf'.
    """
    monty_models_dir = Path(os.getenv("MONTY_MODELS"))
    pretrain_dir = monty_models_dir / "pretrained_ycb_v10"

    if model_type == "dist":
        pretrain_dir = pretrain_dir / "supervised_pre_training_base" / "pretrained"
    elif model_type == "surf":
        pretrain_dir = pretrain_dir / "surf_agent_1lm_77obj" / "pretrained"
    else:
        raise ValueError(f"Invalid model type: {model_type}. Must be 'dist' or 'surf'.")
    return pretrain_dir / "model.pt"


def load_object_model(
    model_path: Path,
    object_name: str,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    lm_id: int = 0,
) -> ObjectModel:
    """Load an object model from a pretraining experiment.

    Args:
        model_path: The path to the model.
        object_name: The name of the object.
        target_position: Target position for world frame transformation.
        target_rotation: Target rotation (Euler angles in degrees) for transformation.
        lm_id: The ID of the LM to load the object model from.

    Returns:
        The object model transformed to world frame.
    """
    data = torch.load(model_path, map_location=torch.device("cpu"))
    data = data["lm_dict"][lm_id]["graph_memory"][object_name][
        "patch"
    ]  # GraphObjectModel
    points = np.array(data.pos, dtype=float)

    feature_dict = {}
    # data.feature_mapping contains feature names mapped to column indices in data.x.
    # Available features include: 'node_ids', 'pose_vectors', 'pose_fully_defined',
    # 'on_object', 'object_coverage', 'rgba', 'hsv', 'principal_curvatures',
    # 'principal_curvatures_log', 'gaussian_curvature', 'mean_curvature',
    # 'gaussian_curvature_sc', 'mean_curvature_sc'.
    # Each feature maps to [start_index, end_index] for slicing the feature tensor.

    for feature in data.feature_mapping.keys():
        idx = data.feature_mapping[feature]
        feature_data = np.array(data.x[:, idx[0] : idx[1]])
        feature_dict[feature] = feature_data

    # The pretrained models are stored at learned position [0, 1.5, 0]
    learned_position = np.array([0, 1.5, 0])

    return ObjectModel(
        points,
        features=feature_dict,
        learned_position=learned_position,
        target_position=target_position,
        target_rotation=target_rotation,
    )


class EpisodeDataLoader:
    """Loads and processes episode data from detailed_run_stats.json."""

    def __init__(self, json_path: Path, model_path: Path):
        self.json_path = json_path
        self.model_path = model_path

        self.lm_data = {}
        self.target_data = {}
        self.sm0_data = {}
        self.sm1_data = {}
        self.target_object_name = ""
        self.target_position = np.array([])
        self.target_rotation = np.array([])
        self.num_lm_steps = 0
        self.lm_to_sm_mapping = []

        self.all_hypotheses_locations = []
        self.all_hypotheses_rotations = []
        self.all_mlh_locations = []
        self.all_mlh_rotations = []
        self.all_mlh_graph_ids = []
        self.all_sm0_locations = []
        self.all_sm1_locations = []
        self.all_sm0_rgba = []
        self.all_sm1_rgba = []
        self.sensed_curvatures = []

    def load_episode_data(self, episode_id: int = 0) -> None:
        """Load episode data from JSON file."""
        logger.info(f"Loading episode {episode_id} data from: {self.json_path}")

        episode_data = deserialize_json_chunks_fast(
            self.json_path, episodes=[episode_id]
        )["0"]
        self.lm_data = episode_data["LM_0"]
        self.num_lm_steps = len(self.lm_data["possible_locations"])
        self.sm0_data = episode_data["SM_0"]
        self.sm1_data = episode_data["SM_1"]

        self.target_name = episode_data["target"]["primary_target_object"]
        self.ground_truth_position = episode_data["target"]["primary_target_position"]
        self.ground_truth_rotation = episode_data["target"]["primary_target_rotation_euler"]

        self._initialize_object_model()
        self._initialize_hypotheses_data()
        self._initialize_mlh_data()

        self._find_lm_to_sm_mapping()
        self._extract_max_abs_curvature()
        self._extract_sensor_locations()
        self._extract_sensor_rgba_patches()

    def _initialize_object_model(self) -> None:
        """Initialize target model in world coordinates."""
        # Load object model from pretrained model in world coordinate
        self.object_model = load_object_model(
            model_path=self.model_path,
            object_name=self.target_name,
            target_position=self.ground_truth_position,
            target_rotation=self.ground_truth_rotation,
        )

        logger.info(f"Target object for episode 0: {self.target_name}")
        logger.info(
            f"{self.target_name} is at position: {self.ground_truth_position} "
            f"and rotation: {self.ground_truth_rotation}"
        )

    def _initialize_hypotheses_data(self) -> None:
        """Extract and transform hypotheses' locations and rotations to world frame."""
        learned_position = np.array([0, 1.5, 0])

        for lm_step in range(self.num_lm_steps):
            possible_locations = self.lm_data["possible_locations"][lm_step]
            hypotheses_locations = np.array(possible_locations[self.target_name])
            possible_rotations = self.lm_data["possible_rotations"][0] # not timestep dependent
            hypotheses_rotations = np.array(possible_rotations[self.target_name])

            transformed_locations, transformed_rotations = apply_world_transform(
                hypotheses_locations,
                hypotheses_rotations,
                learned_position,
                self.ground_truth_position,
                self.ground_truth_rotation,
            )

            self.all_hypotheses_locations.append(transformed_locations)
            self.all_hypotheses_rotations.append(transformed_rotations)

    def _initialize_mlh_data(self) -> None:
        """Extract and transform most likely hypothesis (MLH) data to world frame."""
        learned_position = np.array([0, 1.5, 0])

        for lm_step in range(self.num_lm_steps):
            current_mlh = self.lm_data["current_mlh"][lm_step]
            location = np.array(current_mlh["location"])
            rotation_euler = np.array(current_mlh["rotation"])  # Euler angles

            rotation_matrix = R.from_euler("xyz", rotation_euler, degrees=True).as_matrix()

            transformed_location, transformed_rotation = apply_world_transform(
                location.reshape(1, 3),
                rotation_matrix.reshape(1, 3, 3),
                learned_position,
                self.ground_truth_position,
                self.ground_truth_rotation,
            )

            self.all_mlh_locations.append(transformed_location[0])
            self.all_mlh_rotations.append(
                transformed_rotation[0]
            )
            self.all_mlh_graph_ids.append(current_mlh["graph_id"])

    def _find_lm_to_sm_mapping(self) -> None:
        """Find mapping between LM timesteps and SM timesteps using use_state.

        Raises:
            ValueError: If the number of SM timesteps with use_state=True does not
                match the number of LM timesteps.
        """
        logger.info("Finding LM to SM timestep mapping")

        self.lm_to_sm_mapping = []
        processed_obs = self.sm0_data["processed_observations"]

        sm_timesteps_with_use_state_true = []
        for sm_timestep, obs in enumerate(processed_obs):
            if obs["use_state"]:
                sm_timesteps_with_use_state_true.append(sm_timestep)

        if len(sm_timesteps_with_use_state_true) == self.num_lm_steps:
            self.lm_to_sm_mapping = sm_timesteps_with_use_state_true
            logger.info("Successfully mapped LM timesteps to SM timesteps")
        else:
            raise ValueError(
                f"Mismatch: {len(sm_timesteps_with_use_state_true)} SM "
                f"use_state=True vs {self.num_lm_steps} LM timesteps"
            )

    def _extract_max_abs_curvature(self) -> None:
        """Extract sensed curvature values from sensor module data for each timestep."""
        logger.info("Extracting sensed curvatures from sensor module data")

        processed_obs = self.sm0_data["processed_observations"]

        for lm_timestep in range(self.num_lm_steps):
            sm_timestep = self.lm_to_sm_mapping[lm_timestep]
            obs = processed_obs[sm_timestep]

            non_morphological_features = obs[
                "non_morphological_features"
            ]  # hsv and principal_curvatures_log
            max_abs_curvature = get_relevant_curvature(non_morphological_features)
            self.sensed_curvatures.append(max_abs_curvature)

        logger.info(f"Extracted {len(self.sensed_curvatures)} curvature values")
        logger.info(
            f"Curvature range: {min(self.sensed_curvatures):.4f} to "
            f"{max(self.sensed_curvatures):.4f}"
        )

    def _extract_sensor_locations(self) -> None:
        """Extract sensor locations from SM properties for each timestep."""
        logger.info("Extracting sensor locations from SM properties")

        sm0_properties = self.sm0_data["sm_properties"]
        all_sm0_locations = [data["sm_location"] for data in sm0_properties]

        sm1_properties = self.sm1_data["sm_properties"]
        all_sm1_locations = [data["sm_location"] for data in sm1_properties]

        self.all_sm0_locations = []
        self.all_sm1_locations = []

        for lm_timestep in range(self.num_lm_steps):
            sm_timestep = self.lm_to_sm_mapping[lm_timestep]
            self.all_sm0_locations.append(all_sm0_locations[sm_timestep])
            self.all_sm1_locations.append(all_sm1_locations[sm_timestep])

    def _extract_sensor_rgba_patches(self) -> None:
        """Extract RGBA patches from raw observations for each sensor."""
        logger.info("Extracting sensor RGBA patches from raw observations")

        self.all_sm0_rgba = []
        self.all_sm1_rgba = []

        raw_obs = self.sm0_data["raw_observations"]

        for lm_timestep in range(self.num_lm_steps):
            sm_timestep = self.lm_to_sm_mapping[lm_timestep]
            rgba = np.array(raw_obs[sm_timestep]["rgba"])
            self.all_sm0_rgba.append(rgba)

        raw_obs = self.sm1_data["raw_observations"]

        for lm_timestep in range(self.num_lm_steps):
            sm_timestep = self.lm_to_sm_mapping[lm_timestep]
            rgba = np.array(raw_obs[sm_timestep]["rgba"])
            self.all_sm1_rgba.append(rgba)
