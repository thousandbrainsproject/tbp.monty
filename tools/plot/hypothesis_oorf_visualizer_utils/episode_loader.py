# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Episode data loading utilities for hypothesis visualization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

from tbp.monty.frameworks.utils.graph_matching_utils import get_relevant_curvature
from tbp.monty.frameworks.utils.logging_utils import deserialize_json_chunks

from .data_models import (
    ObjectModelForVisualization,
    transform_locations_model_to_world,
    transform_orientations_model_to_world,
)


class TimestepMappingError(ValueError):
    """Raised when LM and SM timesteps cannot be properly mapped."""


if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def get_model_path(experiment_log_dir: Path) -> Path:
    """Get the path to the model for an experiment from the experiment log directory.

    Args:
        experiment_log_dir: The experiment log directory containing the model.

    Returns:
        The path to the model.pt file.

    Raises:
        FileNotFoundError: If the model.pt file is not found in the expected location.
    """
    model_path = experiment_log_dir / "0" / "model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    return model_path


def load_object_model(
    model_path: Path,
    object_name: str,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    lm_id: int = 0,
) -> ObjectModelForVisualization:
    """Load an object model from a pretraining experiment.

    The model file should contain a GraphObjectModel with feature_mapping that includes
    available features: 'node_ids', 'pose_vectors', 'pose_fully_defined', 'on_object',
    'object_coverage', 'rgba', 'hsv', 'principal_curvatures', 'principal_curvatures_log'
    'gaussian_curvature', 'mean_curvature', 'gaussian_curvature_sc', and
    'mean_curvature_sc'. Each feature maps to [start_index, end_index] for slicing the
    feature tensor.

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

    for feature in data.feature_mapping.keys():
        idx = data.feature_mapping[feature]
        feature_data = np.array(data.x[:, idx[0] : idx[1]])
        feature_dict[feature] = feature_data

    return ObjectModelForVisualization(
        points,
        features=feature_dict,
        model_origin_wrt_world=np.array([0, 1.5, 0]),
        target_location_wrt_world=target_position,
        target_orientation_wrt_world=target_rotation,
    )


class EpisodeDataLoader:
    """Loads and processes episode data from detailed_run_stats.json."""

    def __init__(self, json_path: Path, model_path: Path, episode_id: int = 0):
        self.json_path = json_path
        self.model_path = model_path
        self.episode_id = episode_id

        self.lm_data = {}
        self.target_data = {}
        self.sm0_data = {}
        self.sm1_data = {}
        self.target_object_name = ""
        self.target_position = np.array([])
        self.target_rotation = np.array([])
        self.num_lm_steps = 0
        self.lm_to_sm_mapping = []

        self.all_hyp_locations = []
        self.all_hyp_object_orientations = []
        self.highest_evidence_location = []
        self.highest_evidence_object_orientation = []
        self.highest_evidence_indices = []
        self.all_mlh_locations = []
        self.all_mlh_orientations = []
        self.all_mlh_graph_ids = []
        self.all_sm0_locations = []
        self.all_sm1_locations = []
        self.all_sm0_rgba = []
        self.all_sm1_rgba = []
        self.max_abs_curvatures = []
        self.sensed_orientations = []

    def load_episode_data(self) -> None:
        """Load episode data from JSON file."""
        logger.info(f"Loading episode {self.episode_id} data from: {self.json_path}")

        episode_data = deserialize_json_chunks(
            self.json_path, episodes=[self.episode_id]
        )[str(self.episode_id)]
        self.lm_data = episode_data["LM_0"]
        self.num_lm_steps = len(self.lm_data["possible_locations"])
        self.sm0_data = episode_data["SM_0"]
        self.sm1_data = episode_data["SM_1"]

        self.target_name = episode_data["target"]["primary_target_object"]
        self.ground_truth_position = episode_data["target"]["primary_target_position"]
        self.ground_truth_rotation = episode_data["target"][
            "primary_target_rotation_euler"
        ]

        self._initialize_object_model()
        self._initialize_hypotheses_data()
        self._initialize_mlh_data()

        self._find_lm_to_sm_mapping()
        self._extract_max_abs_curvature()
        self._extract_sensed_orientations()
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
        """Extract and transform hypotheses for visualization.

        The hypotheses are visualized as hypothesized locations superimposed in the
        internal reference frame, which itself is superimposed in world space. This
        enables us to both:
        1. Visualize where the actual object is in the world and where the sensor is
           actually present relative to the object
        2. Show what the hypothesis space looks like in the internal reference frame

        Note that hypothesized rotations are NOT reflected by rotating multiple copies
        of the model and showing these in world coordinates. Instead, the movement in
        the internal reference frame that is visualized is influenced by the
        hypothesized rotation of the object.

        Additionally identify hypotheses with highest evidence for the target object.
        This may be different from MLH, which considers hypotheses across all objects.
        """
        for lm_step in range(self.num_lm_steps):
            hyp_locations_wrt_model = np.array(
                self.lm_data["possible_locations"][lm_step][self.target_name]
            )
            hyp_object_orientations_wrt_model = np.array(
                self.lm_data["possible_rotations"][0][self.target_name]
            )  # not timestep dependent
            hyp_evidences = self.lm_data["evidences"][lm_step][self.target_name]

            hyp_locations = transform_locations_model_to_world(
                hyp_locations_wrt_model,
                self.object_model.model_origin_wrt_world,
                self.ground_truth_position,
                self.ground_truth_rotation,
            )
            hyp_object_orientations = transform_orientations_model_to_world(
                hyp_object_orientations_wrt_model,
                self.ground_truth_rotation,
                orientations_in_row_vector_format=False,
            )
            self.all_hyp_locations.append(hyp_locations)
            self.all_hyp_object_orientations.append(hyp_object_orientations)

            highest_evidence_index = np.argmax(hyp_evidences)
            highest_evidence_location = hyp_locations[highest_evidence_index]
            highest_evidence_orientation = hyp_object_orientations[
                highest_evidence_index
            ]

            self.highest_evidence_location.append(highest_evidence_location)
            self.highest_evidence_object_orientation.append(
                highest_evidence_orientation
            )
            self.highest_evidence_indices.append(highest_evidence_index)

    def _initialize_mlh_data(self) -> None:
        """Extract target object name for MLH."""
        for lm_step in range(self.num_lm_steps):
            current_mlh = self.lm_data["current_mlh"][lm_step]
            self.all_mlh_graph_ids.append(current_mlh["graph_id"])

    def _find_lm_to_sm_mapping(self) -> None:
        """Find mapping between LM timesteps and SM timesteps using use_state.

        This accounts for the fact that not all sensed observations are sent
        to the LM, hence sensor module may be in a higher timestep than the LM.

        For visualization, we want to find the corresponding SM timestep for each LM
        timestep. Note that this may mean that the SM may seem to "jump" in location
        as it jump from SM Timestep 1 to SM Timestep 6 (but only one LM step).

        Raises:
            TimestepMappingError: If the number of SM timesteps with use_state=True does
                not match the number of LM timesteps.
        """
        logger.info("Finding LM to SM timestep mapping")

        processed_obs = self.sm0_data["processed_observations"]

        sm_timesteps_with_use_state_true = []
        for sm_timestep, obs in enumerate(processed_obs):
            if obs["use_state"]:
                sm_timesteps_with_use_state_true.append(sm_timestep)

        if len(sm_timesteps_with_use_state_true) == self.num_lm_steps:
            self.lm_to_sm_mapping = sm_timesteps_with_use_state_true
            logger.info("Successfully mapped LM timesteps to SM timesteps")
        else:
            raise TimestepMappingError(
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
            self.max_abs_curvatures.append(max_abs_curvature)

        logger.info(f"Extracted {len(self.max_abs_curvatures)} curvature values")
        logger.info(
            f"Curvature range: {min(self.max_abs_curvatures):.4f} to "
            f"{max(self.max_abs_curvatures):.4f}"
        )

    def _extract_sensed_orientations(self) -> None:
        """Extract sensed pose vectors from sensor module data for each timestep."""
        logger.info("Extracting sensed pose vectors from sensor module data")

        processed_obs = self.sm0_data["processed_observations"]

        for lm_timestep in range(self.num_lm_steps):
            sm_timestep = self.lm_to_sm_mapping[lm_timestep]
            obs = processed_obs[sm_timestep]

            morphological_features = obs["morphological_features"]
            pose_vectors = np.array(morphological_features["pose_vectors"])
            self.sensed_orientations.append(pose_vectors)

        logger.info(f"Extracted {len(self.sensed_orientations)} orientation matrices")
        logger.info(f"Orientation matrices shape: {self.sensed_orientations[0].shape}")

    def _extract_sensor_locations(self) -> None:
        """Extract sensor locations from SM properties for each timestep.

        Note that sensor locations are in world frame.
        """
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

        raw_obs = None
        if self.sm0_data["raw_observations"] is not None:
            raw_obs = self.sm0_data["raw_observations"]

        for lm_timestep in range(self.num_lm_steps):
            sm_timestep = self.lm_to_sm_mapping[lm_timestep]
            rgba = np.array(raw_obs[sm_timestep]["rgba"])
            self.all_sm0_rgba.append(rgba)

        if self.sm1_data["raw_observations"] is not None:
            raw_obs = self.sm1_data["raw_observations"]

        for lm_timestep in range(self.num_lm_steps):
            sm_timestep = self.lm_to_sm_mapping[lm_timestep]
            rgba = np.array(raw_obs[sm_timestep]["rgba"])
            self.all_sm1_rgba.append(rgba)
