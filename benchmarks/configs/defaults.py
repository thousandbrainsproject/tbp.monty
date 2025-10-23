# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os

import numpy as np

from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.goal_state_generation import EvidenceGoalStateGenerator
from tbp.monty.frameworks.models.sensor_modules import HabitatSM

default_all_noise_params = {
    "features": {
        "pose_vectors": 2,  # rotate by random degrees along xyz
        "hsv": 0.1,  # add gaussian noise with 0.1 std
        "principal_curvatures_log": 0.1,
        "pose_fully_defined": 0.01,  # flip bool in 1% of cases
    },
    "location": 0.002,  # add gaussian noise with 0.002 std
}

default_sensor_features = [
    "pose_vectors",
    "pose_fully_defined",
    "on_object",
    "hsv",
    "principal_curvatures_log",
]

default_all_noisy_sensor_module = dict(
    sensor_module_class=HabitatSM,
    sensor_module_args=dict(
        sensor_module_id="patch",
        features=default_sensor_features,
        save_raw_obs=False,
        delta_thresholds={
            "on_object": 0,
            "distance": 0.01,
        },
        noise_params=default_all_noise_params,
    ),
)

# Everything is weighted 1, except for saturation and value which are not used.
default_feature_weights = {
    "patch": {
        # Weighting saturation and value less since these might change under different
        # lighting conditions. In the future we can extract better features in the SM
        # such as relative value changes.
        "hsv": np.array([1, 0.5, 0.5]),
    }
}

default_tolerance_values = {
    "hsv": np.array([0.1, 0.2, 0.2]),
    "principal_curvatures_log": np.ones(2),
}

default_tolerances = {
    "patch": default_tolerance_values
}  # features where weight is not specified default weight to 1

default_evidence_lm_config = dict(
    learning_module_class=EvidenceGraphLM,
    learning_module_args=dict(
        # mmd of 0.015 get higher performance but slower run time
        max_match_distance=0.01,  # =1cm
        tolerances=default_tolerances,
        feature_weights=default_feature_weights,
        # smaller threshold reduces runtime but also performance
        x_percent_threshold=20,
        # Use this to update all hypotheses at every step as previously
        # evidence_threshold_config="all",
        # Use this to update all hypotheses with evidence > 80% of max evidence (faster)
        evidence_threshold_config="80%",
        # use_multithreading=False,
        # NOTE: Currently not used when loading pretrained graphs.
        max_graph_size=0.3,  # 30cm
        num_model_voxels_per_dim=100,
        gsg_class=EvidenceGoalStateGenerator,
        gsg_args=dict(
            goal_tolerances=dict(
                location=0.015,  # distance in meters
            ),  # Tolerance(s) when determining goal-state success
            elapsed_steps_factor=10,  # Factor that considers the number of elapsed
            # steps as a possible condition for initiating a hypothesis-testing goal
            # state; should be set to an integer reflecting a number of steps
            min_post_goal_success_steps=5,  # Number of necessary steps for a hypothesis
            # goal-state to be considered
            x_percent_scale_factor=0.75,  # Scale x-percent threshold to decide
            # when we should focus on pose rather than determining object ID; should
            # be bounded between 0:1.0; "mod" for modifier
            desired_object_distance=0.03,  # Distance from the object to the
            # agent that is considered "close enough" to the object
        ),
        hypotheses_updater_args=dict(
            # Using a smaller max_nneighbors (5 instead of 10) makes runtime faster,
            # but reduces performance a bit
            max_nneighbors=10
        ),
    ),
)

default_evidence_1lm_config = dict(learning_module_0=default_evidence_lm_config)

# NOTE: maybe lower once we have better policies
# Is not really nescessary for good performance but makes sure we don't just overfit
# on the first few points.
min_eval_steps = 20

monty_models_dir = os.getenv("MONTY_MODELS")

# v6 : Using TLS for surface normal estimation
# v7 : Updated for State class support + using new feature names like pose_vectors
# v8 : Using separate graph per input channel
# v9 : Using models trained on 14 unique rotations
# v10 : Using models trained without the semantic sensor
pretrained_dir = os.path.expanduser(
    os.path.join(monty_models_dir, "pretrained_ycb_v10")
)
