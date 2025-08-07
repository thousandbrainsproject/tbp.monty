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
from typing import Optional

import numpy as np
import torch
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R

from tbp.monty.frameworks.utils.graph_matching_utils import get_relevant_curvature

logger = logging.getLogger(__name__)


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
        pos: np.ndarray,
        features: Optional[dict[str, np.ndarray]] = None,
    ):
        self.pos = np.asarray(pos, dtype=float)
        if features:
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
        self._kd_tree = KDTree(self.pos, leafsize=40)
        return self._kd_tree

    def copy(self, deep: bool = True) -> ObjectModel:
        from copy import deepcopy

        return deepcopy(self) if deep else self

    def rotated(
        self,
        rotation: R | np.ndarray,
        degrees: bool = False,
    ) -> ObjectModel:
        """Rotate the object model.

        Args:
            rotation: The rotation to apply to the object model.
            degrees: Whether the rotation is in degrees or radians.

        Returns:
            The rotated object model.

        Raises:
            ValueError: If the rotation argument is invalid.
        """
        if isinstance(rotation, R):
            rot = rotation
        else:
            arr = np.asarray(rotation)
            if arr.shape == (3,):
                rot = R.from_euler("xyz", arr, degrees=degrees)
            elif arr.shape == (3, 3):
                rot = R.from_matrix(arr)
            else:
                raise ValueError(f"Invalid rotation argument: {rotation}")

        pos = rot.apply(self.pos)
        out = self.copy()
        out.pos = pos

        if hasattr(out, "pose_vectors"):
            rot_matrix = rot.as_matrix()
            new_pose_vectors = np.zeros_like(out.pose_vectors)
            for i in range(len(out.pose_vectors)):
                pose_mat = out.pose_vectors[i].reshape(3, 3)
                rotated_pose = rot_matrix @ pose_mat
                new_pose_vectors[i] = rotated_pose.flatten()
            out.pose_vectors = new_pose_vectors

        return out

    def __add__(self, translation: np.ndarray) -> ObjectModel:
        translation = np.asarray(translation)
        out = self.copy(deep=True)
        out.pos += translation
        return out

    def __sub__(self, translation: np.ndarray) -> ObjectModel:
        translation = np.asarray(translation)
        return self + (-translation)


def get_pretrained_model_path(exp_name: str) -> Path:
    """Get the path to the pretrained model for an experiment.

    Args:
        exp_name: The name of the experiment.

    Returns:
        The path to the pretrained model.

    Raises:
        ValueError: If the experiment name does not contain 'dist' or 'surf'.
    """
    monty_models_dir = Path(os.getenv("MONTY_MODELS"))
    pretrain_dir = monty_models_dir / "pretrained_ycb_v10"

    if "dist" in exp_name:
        pretrain_dir = pretrain_dir / "supervised_pre_training_base" / "pretrained"
    elif "surf" in exp_name:
        pretrain_dir = pretrain_dir / "surf_agent_1lm_77obj" / "pretrained"
    else:
        raise ValueError(
            "The experiment name must contain 'dist' or 'surf' in order to load "
            "correct pretrained model for objects."
        )
    return pretrain_dir / "model.pt"


def load_object_model(
    exp_name: str,
    object_name: str,
    lm_id: int = 0,
) -> ObjectModel:
    """Load an object model from a pretraining experiment.

    Args:
        exp_name: The name of the experiment.
        object_name: The name of the object.
        lm_id: The ID of the LM to load the object model from.

    Returns:
        The object model.
    """
    model_path = get_pretrained_model_path(exp_name)
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

    # Load all available features
    for feature in data.feature_mapping.keys():
        idx = data.feature_mapping[feature]
        feature_data = np.array(data.x[:, idx[0] : idx[1]])
        feature_dict[feature] = feature_data

    return ObjectModel(points, features=feature_dict)


class EpisodeDataLoader:
    """Loads and processes episode data from detailed_run_stats.json."""

    def __init__(self, json_path: str):
        self.json_path = json_path
        self.lm_data = None
        self.target_data = None
        self.sm0_data = None
        self.sm1_data = None
        self.target_object_name = None
        self.target_position = None
        self.target_rotation = None
        self.num_lm_steps = 0
        self.lm_to_sm_mapping = []

        self.all_hypotheses_locations = []
        self.all_hypotheses_evidences = []
        self.all_hypotheses_rotations = []
        self.all_mlh_locations = []
        self.all_mlh_rotations = []
        self.all_mlh_graph_ids = []
        self.all_sm0_locations = []
        self.all_sm1_locations = []
        self.all_sm0_rgba = []
        self.all_sm1_rgba = []
        self.sensed_curvatures = []

    def load_episode_data(self) -> None:
        """Load all timesteps data from JSON file.

        This is the main entry point that coordinates loading target/LM data
        and processing through helper functions.
        """
        logger.info(f"Loading episode data from: {self.json_path}")

        episode_data = self._load_and_parse_json()

        self._initialize_target_data(episode_data)
        self._initialize_lm_data(episode_data)
        self._initialize_sensor_module_data(episode_data)

        # Process LM-related data
        self._process_hypothesis_data()
        self._process_mlh_data()

        # Process sensor module data
        self._find_lm_to_sm_mapping()
        self._extract_sensed_curvatures()
        self._extract_sensor_locations()
        self._extract_sensor_rgba_patches()

    def _load_and_parse_json(self) -> dict:
        """Load and parse the JSON file containing episode data.

        Returns:
            The parsed JSON data for the episode.

        Raises:
            ValueError: If episode 0 data is not found in the JSON.
        """
        with open(self.json_path, "r") as f:
            first_line = f.readline().strip()
            data = json.loads(first_line)

        if "0" not in data:
            raise ValueError("Could not find episode 0 data")

        return data["0"]

    def _initialize_target_data(self, episode_data: dict) -> None:
        """Initialize all target-related data from parsed episode data.

        This includes target object name, position, and rotation.

        Args:
            episode_data: The parsed episode data dictionary.
        """
        self.target_data = episode_data["target"]

        if "primary_target_object" not in self.target_data:
            raise ValueError("primary_target_object not found in target data")
        
        self.target_object_name = self.target_data["primary_target_object"]
        logger.info(f"Target object for episode 0: {self.target_object_name}")

        self.target_position = np.array(self.target_data.get("primary_target_position"))
        self.target_rotation = np.array(
            self.target_data.get("primary_target_rotation_euler")
        )
        logger.info(
            f"{self.target_object_name} is at position: {self.target_position} "
            f"and rotation: {self.target_rotation}"
        )

    def _initialize_lm_data(self, episode_data: dict) -> None:
        """Initialize all learning module (LM) data from parsed episode data.

        This includes the raw LM data and the number of LM steps.

        Args:
            episode_data: The parsed episode data dictionary.
        """
        # Extract LM data
        self.lm_data = episode_data["LM_0"]

        # Set number of LM steps
        self.num_lm_steps = len(self.lm_data["possible_locations"])
        logger.info(f"Episode has {self.num_lm_steps} LM steps")

    def _initialize_sensor_module_data(self, episode_data: dict) -> None:
        """Initialize sensor module data from parsed episode data.

        This includes both SM_0 and SM_1 data.

        Args:
            episode_data: The parsed episode data dictionary.
        """
        # Extract sensor module data
        self.sm0_data = episode_data["SM_0"]
        self.sm1_data = episode_data["SM_1"]

    def _process_hypothesis_data(self) -> None:
        """Process hypothesis locations, evidences, and rotations for all timesteps."""
        for lm_step in range(self.num_lm_steps):
            # Process hypothesis locations
            lm_step_data = self.lm_data["possible_locations"][lm_step]
            self.all_hypotheses_locations.append(
                np.array(lm_step_data[self.target_object_name])
            )

            # Process hypothesis evidences
            self.all_hypotheses_evidences.append(
                np.array(self.lm_data["evidences"][lm_step][self.target_object_name])
            )

            # Process hypothesis rotations (same for all timesteps)
            rotation_data = self.lm_data["possible_rotations"][
                0
            ]  # shape: (n_rotations, 3, 3)
            self.all_hypotheses_rotations.append(
                np.array(rotation_data[self.target_object_name])
            )

    def _process_mlh_data(self) -> None:
        """Process most likely hypothesis (MLH) data for all timesteps."""
        for lm_step in range(self.num_lm_steps):
            mlh_data = self.lm_data["current_mlh"][lm_step]
            self.all_mlh_locations.append(np.array(mlh_data["location"]))
            self.all_mlh_rotations.append(np.array(mlh_data["rotation"]))
            self.all_mlh_graph_ids.append(mlh_data["graph_id"])


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

        logger.info(
            f"Found {len(sm_timesteps_with_use_state_true)} SM timesteps with "
            f"use_state=True"
        )
        logger.info(f"LM has {self.num_lm_steps} LM steps")

        if len(sm_timesteps_with_use_state_true) == self.num_lm_steps:
            self.lm_to_sm_mapping = sm_timesteps_with_use_state_true
            logger.info("Successfully mapped LM timesteps to SM timesteps")
        else:
            raise ValueError(
                f"Mismatch: {len(sm_timesteps_with_use_state_true)} SM "
                f"use_state=True vs {self.num_lm_steps} LM timesteps"
            )

    def _extract_sensed_curvatures(self) -> None:
        """Extract sensed curvature values from sensor module data for each timestep."""
        logger.info("Extracting sensed curvatures from sensor module data")

        processed_obs = self.sm0_data["processed_observations"]

        for lm_timestep in range(self.num_lm_steps):
            # Map LM timestep to SM timestep
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
        all_sm0_locations = [property["sm_location"] for property in sm0_properties]
        logger.info(f"Found {len(all_sm0_locations)} total SM_0 locations")

        sm1_properties = self.sm1_data["sm_properties"]
        all_sm1_locations = [property["sm_location"] for property in sm1_properties]
        logger.info(f"Found {len(all_sm1_locations)} total SM_1 locations")

        self.all_sm0_locations = []
        self.all_sm1_locations = []

        for lm_timestep in range(self.num_lm_steps):
            sm_timestep = self.lm_to_sm_mapping[lm_timestep]
            self.all_sm0_locations.append(all_sm0_locations[sm_timestep])
            self.all_sm1_locations.append(all_sm1_locations[sm_timestep])

        logger.info(
            f"Mapped {len(self.all_sm0_locations)} SM_0 locations to LM timesteps"
        )
        logger.info(
            f"Mapped {len(self.all_sm1_locations)} SM_1 locations to LM timesteps"
        )

    def _extract_sensor_rgba_patches(self) -> None:
        """Extract RGBA patches from raw observations for each sensor."""
        logger.info("Extracting sensor RGBA patches from raw observations")

        self.all_sm0_rgba = []
        self.all_sm1_rgba = []

        raw_obs = self.sm0_data["raw_observations"]
        logger.info(f"Found {len(raw_obs)} SM_0 raw observations")

        for lm_timestep in range(self.num_lm_steps):
            sm_timestep = self.lm_to_sm_mapping[lm_timestep]
            rgba = np.array(raw_obs[sm_timestep]["rgba"])
            self.all_sm0_rgba.append(rgba)

        raw_obs = self.sm1_data["raw_observations"]
        logger.info(f"Found {len(raw_obs)} SM_1 raw observations")

        for lm_timestep in range(self.num_lm_steps):
            sm_timestep = self.lm_to_sm_mapping[lm_timestep]
            rgba = np.array(raw_obs[sm_timestep]["rgba"])
            self.all_sm1_rgba.append(rgba)