# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import copy
import os
from dataclasses import asdict

from benchmarks.configs.defaults import pretrained_dir
from benchmarks.configs.names import UnsupervisedInferenceExperiments
from benchmarks.configs.ycb_experiments import (
    randrot_noise_10distinctobj_dist_agent,
    randrot_noise_10distinctobj_surf_agent,
)
from tbp.monty.frameworks.config_utils.config_args import (
    DetailedEvidenceLMLoggingConfig,
    get_cube_face_and_corner_views_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.loggers.monty_handlers import (
    BasicCSVStatsHandler,
    DetailedJSONHandler,
)
from tbp.monty.frameworks.models.evidence_unsupervised_inference_matching import (
    MontyForUnsupervisedEvidenceGraphMatching,
)

"""
The configs provided in this file are used for unsupervised inference experiments.

These experiments are simpler versions of the benchmark experiments designed for fast
prototyping and testing. More specifically, they are configured for less number of
rotations, less number of objects, and no noise. These configs call
`MontyForUnsupervisedEvidenceGraphMatching`, which removes explicit reset logic.
"""

# surface agent benchmark configs
unsupervised_inference_distinctobj_surf_agent = copy.deepcopy(
    randrot_noise_10distinctobj_surf_agent
)
unsupervised_inference_distinctobj_surf_agent["logging_config"].wandb_handlers = []

# distant agent benchmarks configs
unsupervised_inference_distinctobj_dist_agent = copy.deepcopy(
    randrot_noise_10distinctobj_dist_agent
)
unsupervised_inference_distinctobj_dist_agent["logging_config"].wandb_handlers = []


# === Benchmark Configs === #

# Monty Class to use
MONTY_CLASS = MontyForUnsupervisedEvidenceGraphMatching

# Number of Eval steps
EVAL_STEPS = 30

# define surface agent monty configs
surf_monty_config = copy.deepcopy(
    unsupervised_inference_distinctobj_surf_agent["monty_config"]
)
surf_monty_config.monty_class = MONTY_CLASS
surf_monty_config.monty_args.min_eval_steps = EVAL_STEPS
unsupervised_inference_distinctobj_surf_agent.update(
    {"monty_config": surf_monty_config}
)
unsupervised_inference_distinctobj_surf_agent[
    "experiment_args"
].max_eval_steps = EVAL_STEPS


# define distant agent monty configs
dist_monty_config = copy.deepcopy(
    unsupervised_inference_distinctobj_dist_agent["monty_config"]
)
dist_monty_config.monty_class = MONTY_CLASS
dist_monty_config.monty_args.min_eval_steps = EVAL_STEPS
unsupervised_inference_distinctobj_dist_agent.update(
    {"monty_config": dist_monty_config}
)
unsupervised_inference_distinctobj_dist_agent[
    "experiment_args"
].max_eval_steps = EVAL_STEPS

# === End Benchmark Configs === #


# === Rapid Prototyping Configs === #

# This enables or disables rapid prototyping configs
APPLY_RAPID_CONFIGS = False

# Changes the number of rotations per object
NUM_ROTATIONS = 1

# Changes the types of YCB objects evaluated
OBJECTS_LIST = ["strawberry", "banana"]

# Use False during debugging (with breakpoints) of evidence updates
USE_MULTITHREADING = True

# Controls the number of evaluation steps for each object
EVAL_STEPS = 50

# Controls whether to output detailed logs
DETAILED_LOG = False

# === End Rapid Prototyping Configs === #

# define rotations
test_rotations = get_cube_face_and_corner_views_rotations()[:NUM_ROTATIONS]

# define monty loggers
monty_handlers = [
    BasicCSVStatsHandler,
]
if DETAILED_LOG:
    monty_handlers.append(DetailedJSONHandler)


# define data path for supervised graph models
model_path_10distinctobj = os.path.join(
    pretrained_dir,
    "surf_agent_1lm_10distinctobj/pretrained/",
)


# define surface agent monty configs
surf_monty_config = copy.deepcopy(
    unsupervised_inference_distinctobj_surf_agent["monty_config"]
)
surf_monty_config.learning_module_configs["learning_module_0"]["learning_module_args"][
    "use_multithreading"
] = USE_MULTITHREADING


# define distant agent monty configs
dist_monty_config = copy.deepcopy(
    unsupervised_inference_distinctobj_dist_agent["monty_config"]
)
dist_monty_config.learning_module_configs["learning_module_0"]["learning_module_args"][
    "use_multithreading"
] = USE_MULTITHREADING


# Apply prototyping configs
if APPLY_RAPID_CONFIGS:
    unsupervised_inference_distinctobj_surf_agent.update(
        dict(
            experiment_args=EvalExperimentArgs(
                model_name_or_path=model_path_10distinctobj,
                n_eval_epochs=NUM_ROTATIONS,
                max_eval_steps=EVAL_STEPS,
            ),
            logging_config=DetailedEvidenceLMLoggingConfig(
                monty_handlers=monty_handlers,
                wandb_handlers=[],
                # python_log_level="WARNING",
            ),
            monty_config=surf_monty_config,
            eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
                object_names=OBJECTS_LIST,
                object_init_sampler=PredefinedObjectInitializer(
                    rotations=test_rotations
                ),
            ),
        )
    )

    unsupervised_inference_distinctobj_dist_agent.update(
        dict(
            experiment_args=EvalExperimentArgs(
                model_name_or_path=model_path_10distinctobj,
                n_eval_epochs=NUM_ROTATIONS,
                max_eval_steps=EVAL_STEPS,
            ),
            logging_config=DetailedEvidenceLMLoggingConfig(
                monty_handlers=monty_handlers,
                wandb_handlers=[],
                # python_log_level="WARNING",
            ),
            monty_config=dist_monty_config,
            eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
                object_names=OBJECTS_LIST,
                object_init_sampler=PredefinedObjectInitializer(
                    rotations=test_rotations
                ),
            ),
        )
    )

experiments = UnsupervisedInferenceExperiments(
    unsupervised_inference_distinctobj_surf_agent=unsupervised_inference_distinctobj_surf_agent,
    unsupervised_inference_distinctobj_dist_agent=unsupervised_inference_distinctobj_dist_agent,
)
CONFIGS = asdict(experiments)
