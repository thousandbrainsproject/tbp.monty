# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Experiment configurations for testing unsupervised object ID association learning.

This module provides experiment configurations that demonstrate and test the
unsupervised association learning capabilities between different learning modules.
"""

import os
from dataclasses import asdict

from benchmarks.configs.names import UnsupervisedAssociationExperiments
from tbp.monty.frameworks.actions.action_samplers import ConstantSampler
from tbp.monty.frameworks.config_utils.config_args import MontyArgs
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    ExperimentArgs,
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
from tbp.monty.frameworks.loggers.monty_handlers import (
    BasicCSVStatsHandler,
    ReproduceEpisodeHandler,
)
from tbp.monty.frameworks.models.evidence_matching.unsupervised_evidence_lm import (
    UnsupervisedEvidenceGraphLM,
)
from tbp.monty.frameworks.models.evidence_matching.unsupervised_model import (
    MontyForUnsupervisedAssociation,
)
from tbp.monty.frameworks.models.goal_state_generation import EvidenceGoalStateGenerator
from tbp.monty.frameworks.models.motor_policies import InformedPolicy
from tbp.monty.frameworks.models.motor_system import MotorSystem
from tbp.monty.frameworks.models.sensor_modules import FeatureChangeSM
from tbp.monty.simulators.habitat.configs import (
    TwoLMStackedDistantMountHabitatDatasetArgs,
)

# Default objects for testing
test_objects = get_object_names_by_idx(0, 5)  # First 5 YCB objects

# Sensor module configuration for first patch sensor (patch_0)
patch_0_sensor_module = {
    "sensor_module_class": FeatureChangeSM,
    "sensor_module_args": {
        "sensor_module_id": "patch_0",
        "features": [
            "on_object",
            "object_coverage",
            "rgba",
            "hsv",
            "pose_vectors",
            "principal_curvatures_log",
        ],
        "save_raw_obs": False,
        "delta_thresholds": {
            "on_object": 0,
            "distance": 0.01,
        },
    },
}

# Sensor module configuration for second patch sensor (patch_1)
patch_1_sensor_module = {
    "sensor_module_class": FeatureChangeSM,
    "sensor_module_args": {
        "sensor_module_id": "patch_1",
        "features": [
            "on_object",
            "object_coverage",
            "rgba",
            "hsv",
            "pose_vectors",
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
            # Single-modality (two patches)
            "patch_0": {
                "hsv": [0.1, 0.2, 0.2],
                "principal_curvatures_log": [1.0, 1.0],
                "pose_vectors": [0.1, 0.1, 0.1],
            },
            "patch_1": {
                "hsv": [0.1, 0.2, 0.2],
                "principal_curvatures_log": [1.0, 1.0],
                "pose_vectors": [0.1, 0.1, 0.1],
            },
        },
        "feature_weights": {
            "patch_0": {
                "hsv": [1, 0.5, 0.5],
                "principal_curvatures_log": [1, 1],
                "pose_vectors": [1, 1, 1],
            },
            "patch_1": {
                "hsv": [1, 0.5, 0.5],
                "principal_curvatures_log": [1, 1],
                "pose_vectors": [1, 1, 1],
            },
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
        "policy_class": InformedPolicy,
        "policy_args": make_informed_policy_config(
            action_space_type="distant_agent",
            action_sampler_class=ConstantSampler,
            action_sampler_args={"action": [0, 0, 0, 0, 1]},
        ),
    },
}

# Base dataset configuration - two patch sensors on distant agent mount
base_dataset_args = TwoLMStackedDistantMountHabitatDatasetArgs()

# Base dataloader configuration
base_dataloader_args = EnvironmentDataloaderPerObjectArgs(
    object_names=test_objects,
    object_init_sampler=RandomRotationObjectInitializer(),
)


def create_experiment_config(monty_config, logging_config=None):
    """Create a standard experiment configuration.

    Returns:
        dict: Experiment configuration dictionary.
    """
    if logging_config is None:
        logging_config = {
            "python_log_level": "INFO",
            "monty_log_level": "BASIC",
            "python_log_to_file": True,
            "python_log_to_stdout": True,
            "python_log_to_stderr": False,
            "run_name": "test_run",
            "output_dir": os.getenv("MONTY_LOGS", "~/tbp/results/monty/"),
            "wandb_handlers": [],
            "monty_handlers": [BasicCSVStatsHandler],
        }

    return {
        "experiment_class": MontyObjectRecognitionExperiment,
        "experiment_args": ExperimentArgs(
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


def create_single_modality_balanced_config():
    """Create a simple single-modality (two patches) experiment with balanced params.

    Returns:
        dict: Experiment configuration dictionary.
    """
    lm_configs = create_cross_modal_lm_configs(
        base_learning_module,
        num_lms=2,
        modality_names=["visual_0", "visual_1"],
        association_params=get_association_params_preset("balanced"),
    )

    monty_config = create_unsupervised_association_monty_config(
        base_monty_config={
            "monty_class": MontyForUnsupervisedAssociation,
            "monty_args": MontyArgs(
                min_eval_steps=200,
                min_train_steps=200,
                num_exploratory_steps=1000,
                max_total_steps=2000,
            ),
            "sensor_module_configs": {
                # patch_0
                "sensor_module_0": patch_0_sensor_module,
                # patch_1
                "sensor_module_1": patch_1_sensor_module,
            },
            "learning_module_configs": lm_configs,
            "motor_system_config": base_motor_system,
            "sm_to_agent_dict": {"patch_0": "agent_id_0", "patch_1": "agent_id_0"},
            "sm_to_lm_matrix": [[0], [1]],  # Each patch to different LM
            "lm_to_lm_matrix": [[], []],  # No hierarchical connections
            "lm_to_lm_vote_matrix": [[1], [0]],  # Cross-voting between LMs
        },
        lm_configs=lm_configs,
        enable_association_analysis=True,
        log_association_details=True,
    )

    return create_experiment_config(monty_config)


def create_single_modality_aggressive_config():
    """Create a single-modality (two patches) experiment with aggressive params.

    Returns:
        dict: Experiment configuration dictionary.
    """
    # Create two learning modules representing the same modality (two patches)
    lm_configs = create_cross_modal_lm_configs(
        base_learning_module,
        num_lms=2,
        modality_names=["visual_0", "visual_1"],
        association_params=get_association_params_preset("aggressive"),
    )

    # Create Monty configuration with full cross-voting
    monty_config = create_unsupervised_association_monty_config(
        base_monty_config={
            "monty_class": MontyForUnsupervisedAssociation,
            "monty_args": MontyArgs(
                min_eval_steps=400,  # Longer to build stronger associations
                min_train_steps=400,
                num_exploratory_steps=2000,
                max_total_steps=4000,
            ),
            "sensor_module_configs": {
                # patch_0
                "sensor_module_0": patch_0_sensor_module,
                # patch_1
                "sensor_module_1": patch_1_sensor_module,
            },
            "learning_module_configs": lm_configs,
            "motor_system_config": base_motor_system,
            # Both sensors on same agent
            "sm_to_agent_dict": {"patch_0": "agent_id_0", "patch_1": "agent_id_0"},
            "sm_to_lm_matrix": [[0], [1]],  # Each patch sensor to different LM
            "lm_to_lm_matrix": [[], []],  # No hierarchical connections
            "lm_to_lm_vote_matrix": [[1], [0]],  # Cross-voting between LMs
        },
        lm_configs=lm_configs,
        enable_association_analysis=True,
        log_association_details=True,
    )

    exp = create_experiment_config(monty_config)
    # Override dataset to a mount that provides patch_0 and patch_1
    exp["dataset_args"] = TwoLMStackedDistantMountHabitatDatasetArgs()
    return exp


def create_single_modality_conservative_config():
    """Create a single-modality (two patches) experiment with conservative params.

    Returns:
        dict: Experiment configuration dictionary.
    """
    lm_configs = create_cross_modal_lm_configs(
        base_learning_module,
        num_lms=2,
        modality_names=["visual_0", "visual_1"],
        association_params=get_association_params_preset("conservative"),
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
            "sensor_module_configs": {
                "sensor_module_0": patch_0_sensor_module,
                "sensor_module_1": patch_1_sensor_module,
            },
            "learning_module_configs": lm_configs,
            "motor_system_config": base_motor_system,
            "sm_to_agent_dict": {"patch_0": "agent_id_0", "patch_1": "agent_id_0"},
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
        "monty_log_level": "BASIC",
        "python_log_to_file": True,
        "python_log_to_stdout": False,
        "output_dir": os.getenv("MONTY_LOGS", "~/tbp/results/monty/"),
        "run_name": "conservative_association",
        "wandb_handlers": [],
        "monty_handlers": [],
    }

    return create_experiment_config(monty_config, conservative_logging_config)


def create_association_strategy_comparison_config():
    """Create configuration for comparing different association strategies.

    Returns:
        dict: Experiment configuration dictionary.
    """
    # Reuse top-level base configs to avoid duplicated code

    # Create multiple LM configs with different association parameters
    lm_configs = []

    # Strategy 1: Balanced weights
    balanced_params = get_association_params_preset("balanced")
    lm_configs.append(
        create_cross_modal_lm_configs(
            base_learning_module,
            num_lms=2,
            association_params=balanced_params,
        )
    )

    # Strategy 2: Conservative (spatial-focused)
    conservative_params = get_association_params_preset("conservative")
    lm_configs.append(
        create_cross_modal_lm_configs(
            base_learning_module,
            num_lms=2,
            association_params=conservative_params,
        )
    )

    # Strategy 3: Aggressive (co-occurrence-focused)
    aggressive_params = get_association_params_preset("aggressive")
    lm_configs.append(
        create_cross_modal_lm_configs(
            base_learning_module,
            num_lms=2,
            association_params=aggressive_params,
        )
    )

    # Create base monty config for comparison
    base_monty_config = {
        "monty_class": MontyForUnsupervisedAssociation,
        "monty_args": MontyArgs(
            min_eval_steps=400,
            min_train_steps=400,
            num_exploratory_steps=2000,
            max_total_steps=4000,
        ),
        "sensor_module_configs": {
            "sensor_module_0": patch_0_sensor_module,
            "sensor_module_1": patch_1_sensor_module,
        },
        # Use balanced as base
        "learning_module_configs": lm_configs[0],
        "motor_system_config": base_motor_system,
        "sm_to_agent_dict": {"patch_0": "agent_id_0", "patch_1": "agent_id_0"},
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
        "monty_log_level": "BASIC",
        "python_log_to_file": True,
        "python_log_to_stdout": True,
        "python_log_to_stderr": False,  # Add missing key
        "log_parallel_wandb": False,  # Add missing key
        "output_dir": (
            os.getenv("MONTY_LOGS", "~/tbp/results/monty/") + "association_comparison"
        ),
        "run_name": "association_comparison",
        "wandb_handlers": [],
        "monty_handlers": [],
        "detailed_association_metrics": True,
    }

    return create_experiment_config(monty_config, comparison_logging_config)


def create_5lm_77obj_benchmark_config():
    """Create a 5-LM unsupervised association learning experiment.

    This experiment trains 5 learning modules with unsupervised association
    capabilities from scratch (no pretrained models) to learn object
    representations and associations through cross-modal coordination. After
    training, it evaluates on 77 YCB objects.

    Returns:
        Dict[str, Any]: Complete experiment configuration for the 5-LM/77-obj
        benchmark.
    """
    import numpy as np

    from benchmarks.configs.ycb_experiments import (
        default_5sm_config,
        min_eval_steps,
    )
    from tbp.monty.frameworks.config_utils.config_args import (
        MontyArgs,
    )
    from tbp.monty.frameworks.config_utils.make_dataset_configs import (
        EnvironmentDataloaderPerObjectArgs,
        EvalExperimentArgs,
        RandomRotationObjectInitializer,
        get_object_names_by_idx,
    )
    from tbp.monty.frameworks.environments.embodied_data import (
        EnvironmentDataset,
        InformedEnvironmentDataLoader,
    )
    from tbp.monty.simulators.habitat.configs import FiveLMMountHabitatDatasetArgs

    # Create 5 LMs with unsupervised association capabilities
    # These will learn object representations and associations from scratch
    balanced_params = get_association_params_preset("balanced")

    # Ensure association learning is enabled for training from scratch
    balanced_params["association_learning_enabled"] = True

    # Create individual LM configs
    lm_configs = {}
    for i in range(5):
        lm_configs[f"learning_module_{i}"] = {
            "learning_module_class": UnsupervisedEvidenceGraphLM,
            "learning_module_args": {
                "learning_module_id": f"lm_{i}",
                "max_match_distance": 0.01,
                "tolerances": {
                    f"patch_{i}": {
                        "hsv": np.array([0.1, 0.2, 0.2]),
                        "principal_curvatures_log": np.array([1.0, 1.0]),
                    },
                },
                "feature_weights": {
                    f"patch_{i}": {
                        "hsv": np.array([1.0, 0.5, 0.5]),
                    },
                },
                "graph_delta_thresholds": {
                    f"patch_{i}": {
                        "distance": 0.01,
                        "pose": 30,
                    },
                },
                **balanced_params,
            },
        }

    # Create the experiment config for unsupervised association learning from scratch
    experiment_config = {
        "experiment_class": MontyObjectRecognitionExperiment,
        "experiment_args": EvalExperimentArgs(
            model_name_or_path=None,  # Start from scratch - no pretrained labels
            n_train_epochs=3,  # Enable training for association learning
            n_eval_epochs=1,  # Evaluate after training
            min_lms_match=3,  # Same as baseline
            do_train=True,  # Enable training for unsupervised association learning
            do_eval=True,  # Evaluate the learned associations
            max_train_steps=1000,  # Allow sufficient training for association learning
            max_eval_steps=500,  # Same as baseline for fair comparison
        ),
        "dataset_class": EnvironmentDataset,
        "train_dataloader_class": InformedEnvironmentDataLoader,
        "eval_dataloader_class": InformedEnvironmentDataLoader,
        "logging_config": {
            "python_log_level": "INFO",
            "monty_log_level": "BASIC",
            "python_log_to_file": True,
            "python_log_to_stdout": True,
            "python_log_to_stderr": False,
            "log_parallel_wandb": False,
            "output_dir": (
                os.getenv("MONTY_LOGS", "~/tbp/results/monty/")
                + "unsupervised_5lm_from_scratch"
            ),
            "run_name": "unsupervised_5lm_from_scratch_benchmark",
            "wandb_handlers": [],
            "monty_handlers": [
                BasicCSVStatsHandler,
                ReproduceEpisodeHandler,
            ],
        },
        "monty_config": {
            "monty_class": MontyForUnsupervisedAssociation,
            "monty_args": MontyArgs(
                min_eval_steps=min_eval_steps,
            ),
            "sensor_module_configs": default_5sm_config,
            "learning_module_configs": lm_configs,
            "motor_system_config": {
                "motor_system_class": MotorSystem,
                "motor_system_args": {
                    "policy_class": InformedPolicy,
                    "policy_args": make_informed_policy_config(
                        action_space_type="distant_agent",
                        action_sampler_class=ConstantSampler,
                    ),
                },
            },
            "sm_to_agent_dict": {
                "patch_0": "agent_id_0",
                "patch_1": "agent_id_0",
                "patch_2": "agent_id_0",
                "patch_3": "agent_id_0",
                "patch_4": "agent_id_0",
            },
            "sm_to_lm_matrix": [[0], [1], [2], [3], [4]],
            "lm_to_lm_matrix": [[], [], [], [], []],
            "lm_to_lm_vote_matrix": [
                [1, 2, 3, 4],
                [0, 2, 3, 4],
                [0, 1, 3, 4],
                [0, 1, 2, 4],
                [0, 1, 2, 3],
            ],
        },
        "dataset_args": FiveLMMountHabitatDatasetArgs(),
        "train_dataloader_args": EnvironmentDataloaderPerObjectArgs(
            object_names=get_object_names_by_idx(
                0,
                30,
            ),  # Use 30 objects for association learning
            object_init_sampler=RandomRotationObjectInitializer(),
        ),
        "eval_dataloader_args": EnvironmentDataloaderPerObjectArgs(
            object_names=get_object_names_by_idx(
                0,
                77,
            ),  # Use all 77 objects like baseline
            object_init_sampler=RandomRotationObjectInitializer(),
        ),
    }

    return experiment_config


# Export experiment configurations following Monty patterns

# Create the experiments dataclass instance
experiments = UnsupervisedAssociationExperiments(
    simple_single_modality_association=create_single_modality_balanced_config(),
    single_modality_aggressive=create_single_modality_aggressive_config(),
    single_modality_conservative=create_single_modality_conservative_config(),
    association_strategy_comparison=create_association_strategy_comparison_config(),
    unsupervised_5lm_77obj_benchmark=create_5lm_77obj_benchmark_config(),
)

# Convert to dictionary format expected by the benchmarks system
CONFIGS = asdict(experiments)
