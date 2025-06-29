# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""
Experiment configurations for testing unsupervised object ID association learning.

This module provides experiment configurations that demonstrate and test the
unsupervised association learning capabilities between different learning modules.
"""
from tbp.monty.frameworks.actions.action_samplers import ConstantSampler
from tbp.monty.frameworks.config_utils.config_args import (
    MontyArgs,
)

from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    RandomRotationObjectInitializer,
    get_object_names_by_idx,
)
from tbp.monty.frameworks.config_utils.policy_setup_utils import (
    make_informed_policy_config,
)
from tbp.monty.frameworks.config_utils.unsupervised_association_configs import (
    create_cross_modal_lm_configs,
    create_unsupervised_association_monty_config,
    get_association_params_preset,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.models.evidence_matching.unsupervised_evidence_lm import (
    UnsupervisedEvidenceGraphLM,
)
from tbp.monty.frameworks.models.evidence_matching.unsupervised_model import (
    MontyForUnsupervisedAssociation,
)
from tbp.monty.frameworks.models.goal_state_generation import EvidenceGoalStateGenerator
from tbp.monty.frameworks.models.motor_system import MotorSystem
from tbp.monty.frameworks.models.sensor_modules import (
    FeatureChangeSM,
)
from tbp.monty.simulators.habitat.configs import (
    PatchViewFinderMontyWorldMountHabitatDatasetArgs,
)

# Default objects for testing
test_objects = get_object_names_by_idx(0, 5)  # First 5 YCB objects

# Base sensor module configuration
base_sensor_module = {
    "sensor_module_class": FeatureChangeSM,
    "sensor_module_args": {
        "sensor_module_id": "patch",
        "features": [
            "on_object",
            "object_coverage",
            "rgba",
            "hsv",
            "point_normal",
            "principal_curvatures_log",
        ],
        "save_raw_obs": False,
        "delta_thresholds": {
            "on_object": 0,
            "distance": 0.01,
        },
    },
}

# Base learning module configuration
base_learning_module = {
    "learning_module_class": UnsupervisedEvidenceGraphLM,
    "learning_module_args": {
        "max_match_distance": 0.01,
        "tolerances": {
            "hsv": [0.1, 0.2, 0.2],
            "principal_curvatures_log": [1.0, 1.0],
        },
        "feature_weights": {
            "patch": {
                "hsv": [1, 0.5, 0.5],
                "principal_curvatures_log": [1, 1],
            }
        },
        "object_evidence_threshold": 1.0,
        "x_percent_threshold": 10,
        "gsg_class": EvidenceGoalStateGenerator,
        "gsg_args": {},
    },
}

# Base motor system configuration
base_motor_system = {
    "motor_system_class": MotorSystem,
    "motor_system_args": {
        "policy": make_informed_policy_config(
            action_space_type="distant_agent",
            action_sampler_class=ConstantSampler,
            action_sampler_args={"action": [0, 0, 0, 0, 1]},
        )
    },
}

# Base dataset configuration
base_dataset_args = PatchViewFinderMontyWorldMountHabitatDatasetArgs()

# Base dataloader configuration
base_dataloader_args = EnvironmentDataloaderPerObjectArgs(
    object_names=test_objects,
    object_init_sampler=RandomRotationObjectInitializer(),
)


def create_experiment_config(monty_config, logging_config=None):
    """Create a standard experiment configuration with the given monty_config and logging_config."""
    if logging_config is None:
        logging_config = {
            "python_log_level": "INFO",
            "python_log_to_file": True,
            "python_log_to_stdout": True,
            "python_log_to_stderr": False,
            "run_name": "test_run",
            "wandb_handlers": [],
        }

    return {
        "experiment_class": MontyObjectRecognitionExperiment,
        "experiment_args": EvalExperimentArgs(
            n_eval_epochs=1,
            max_eval_steps=len(test_objects),
        ),
        "logging_config": logging_config,
        "monty_config": monty_config,
        "dataset_class": ED.EnvironmentDataset,
        "dataset_args": base_dataset_args,
        "train_dataloader_class": ED.InformedEnvironmentDataLoader,
        "train_dataloader_args": base_dataloader_args,
        "eval_dataloader_class": ED.InformedEnvironmentDataLoader,
        "eval_dataloader_args": base_dataloader_args,
    }


def create_simple_cross_modal_config():
    """Create a simple cross-modal experiment configuration."""
    # Create two learning modules with different modality IDs
    lm_configs = create_cross_modal_lm_configs(
        base_learning_module,
        num_lms=2,
        modality_names=['visual', 'touch'],
        association_params=get_association_params_preset('balanced'),
    )

    # Create Monty configuration with association capabilities
    monty_config = create_unsupervised_association_monty_config(
        base_monty_config={
            "monty_class": MontyForUnsupervisedAssociation,
            "monty_args": MontyArgs(
                min_eval_steps=200,
                min_train_steps=200,
                num_exploratory_steps=1000,
                max_total_steps=2000,
            ),
            "sensor_module_configs": [base_sensor_module, base_sensor_module],
            "learning_module_configs": lm_configs,
            "motor_system_config": base_motor_system,
            "sm_to_agent_dict": {0: 0, 1: 0},  # Both sensors on same agent
            "sm_to_lm_matrix": [[0], [1]],  # Each sensor to different LM
            "lm_to_lm_matrix": [[], []],  # No hierarchical connections
            "lm_to_lm_vote_matrix": [[1], [0]],  # Cross-voting between LMs
        },
        lm_configs=lm_configs,
        enable_association_analysis=True,
        log_association_details=True,
    )

    return create_experiment_config(monty_config)


def create_multi_modal_config():
    """Create a multi-modal experiment configuration with 3 learning modules."""
    # Create three learning modules representing different modalities
    lm_configs = create_cross_modal_lm_configs(
        base_learning_module,
        num_lms=3,
        modality_names=['visual', 'touch', 'proprioceptive'],
        association_params=get_association_params_preset('aggressive'),
    )

    # Create Monty configuration with full cross-voting
    monty_config = create_unsupervised_association_monty_config(
        base_monty_config={
            "monty_class": MontyForUnsupervisedAssociation,
            "monty_args": MontyArgs(
                min_eval_steps=300,
                min_train_steps=300,
                num_exploratory_steps=1500,
                max_total_steps=3000,
            ),
            "sensor_module_configs": [base_sensor_module] * 3,
            "learning_module_configs": lm_configs,
            "motor_system_config": base_motor_system,
            "sm_to_agent_dict": {0: 0, 1: 0, 2: 0},  # All sensors on same agent
            "sm_to_lm_matrix": [[0], [1], [2]],  # Each sensor to different LM
            "lm_to_lm_matrix": [[], [], []],  # No hierarchical connections
            "lm_to_lm_vote_matrix": [[1, 2], [0, 2], [0, 1]],  # Full cross-voting
        },
        lm_configs=lm_configs,
        enable_association_analysis=True,
        log_association_details=True,
    )

    return create_experiment_config(monty_config)


def create_conservative_association_config():
    """Create a configuration with conservative association parameters."""
    lm_configs = create_cross_modal_lm_configs(
        base_learning_module,
        num_lms=2,
        modality_names=['visual', 'touch'],
        association_params=get_association_params_preset('conservative'),
    )

    monty_config = create_unsupervised_association_monty_config(
        base_monty_config={
            "monty_class": MontyForUnsupervisedAssociation,
            "monty_args": MontyArgs(
                min_eval_steps=400,  # Longer to build stronger associations
                min_train_steps=400,
                num_exploratory_steps=2000,
                max_total_steps=4000,
            ),
            "sensor_module_configs": [base_sensor_module, base_sensor_module],
            "learning_module_configs": lm_configs,
            "motor_system_config": base_motor_system,
            "sm_to_agent_dict": {0: 0, 1: 0},
            "sm_to_lm_matrix": [[0], [1]],
            "lm_to_lm_matrix": [[], []],
            "lm_to_lm_vote_matrix": [[1], [0]],
        },
        lm_configs=lm_configs,
        enable_association_analysis=True,
        log_association_details=False,  # Less verbose logging
    )

    # Custom logging config for conservative approach
    conservative_logging_config = {
        "python_log_level": "WARNING",  # Less verbose
        "python_log_to_file": True,
        "python_log_to_stdout": False,
        "wandb_handlers": [],
    }

    return create_experiment_config(monty_config, conservative_logging_config)


def create_association_strategy_comparison_config():
    """Create configuration for comparing different association strategies."""
    # Define base components needed for the experiment
    base_sensor_module_local = {
        "sensor_module_class": "HabitatDistantPatchSM",  # Placeholder
        "sensor_module_args": {"sensor_module_id": "patch"},
    }

    base_motor_system_local = {
        "motor_system_class": MotorSystem,
        "motor_system_args": make_informed_policy_config(
            action_space_type="distant_agent",
            action_sampler_class=ConstantSampler,
        ),
    }

    # Create multiple LM configs with different association parameters
    lm_configs = []

    # Strategy 1: Balanced weights
    balanced_params = get_association_params_preset("balanced")
    lm_configs.append(create_cross_modal_lm_configs(
        base_learning_module,
        num_lms=2,
        association_params=balanced_params
    ))

    # Strategy 2: Conservative (spatial-focused)
    conservative_params = get_association_params_preset("conservative")
    lm_configs.append(create_cross_modal_lm_configs(
        base_learning_module,
        num_lms=2,
        association_params=conservative_params
    ))

    # Strategy 3: Aggressive (co-occurrence-focused)
    aggressive_params = get_association_params_preset("aggressive")
    lm_configs.append(create_cross_modal_lm_configs(
        base_learning_module,
        num_lms=2,
        association_params=aggressive_params
    ))

    # Create base monty config for comparison
    base_monty_config = {
        "monty_class": MontyForUnsupervisedAssociation,
        "monty_args": MontyArgs(
            min_eval_steps=400,
            min_train_steps=400,
            num_exploratory_steps=2000,
            max_total_steps=4000,
        ),
        "sensor_module_configs": [base_sensor_module_local, base_sensor_module_local],
        "learning_module_configs": lm_configs[0],  # Use balanced as base
        "motor_system_config": base_motor_system_local,
        "sm_to_agent_dict": {0: 0, 1: 0},
        "sm_to_lm_matrix": [[0], [1]],
        "lm_to_lm_matrix": [[], []],
        "lm_to_lm_vote_matrix": [[1], [0]],
    }

    monty_config = create_unsupervised_association_monty_config(
        base_monty_config=base_monty_config,
        lm_configs=lm_configs[0],
        enable_association_analysis=True,
        log_association_details=True,
    )

    # Comparison logging config with detailed metrics
    comparison_logging_config = {
        "python_log_level": "INFO",
        "python_log_to_file": True,
        "python_log_to_stdout": True,
        "wandb_handlers": [],
        "detailed_association_metrics": True,
    }

    return create_experiment_config(monty_config, comparison_logging_config)


# Export experiment configurations following Monty patterns
from benchmarks.configs.names import UnsupervisedAssociationExperiments
from dataclasses import asdict

# Create the experiments dataclass instance
experiments = UnsupervisedAssociationExperiments(
    simple_cross_modal_association=create_simple_cross_modal_config(),
    multi_modal_association=create_multi_modal_config(),
    association_strategy_comparison=create_association_strategy_comparison_config(),
)

# Convert to dictionary format expected by the benchmarks system
CONFIGS = asdict(experiments)
