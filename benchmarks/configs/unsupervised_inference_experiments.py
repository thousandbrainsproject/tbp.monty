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
from tbp.monty.frameworks.models.no_reset_evidence_matching import (
    MontyForNoResetEvidenceGraphMatching,
    NoResetEvidenceGraphLM,
)

"""
These configurations define the experimental setup for testing unsupervised inference
in dynamic environments, where objects are swapped without resetting Monty's internal
state. The goal of these experiments is to evaluate Monty's ability to dynamically
adapt its hypotheses when the underlying object changes — without receiving any
external signal or supervisory reset.

At a high level, these configs extend existing benchmark experiments
(`randrot_noise_10distinctobj_{surf,dist}_agent`) but replace the core Monty
and LM classes with variants that explicitly disable episode-based reset
logic (`MontyForNoResetEvidenceGraphMatching` and `NoResetEvidenceGraphLM`).
This ensures that Monty's internal state and evidence accumulation mechanisms
persist across objects.

In standard experiments, Monty's internal state is reinitialized at the start of
each episode via a reset signal. This includes resetting evidence scores, hypothesis
space, and internal counters. In this unsupervised inference setup, that reset signal
is removed — allowing us to simulate real-world dynamics where object boundaries are
not clearly marked.

Here are some key characteristics of the available configs:
    - **Evaluation-only**: No learning or graph updates occur during these runs.
        Pre-trained object models are loaded from model_path_10distinctobj before
        the experiment begins.
    - **Controlled number of steps**: Each object is shown for a fixed number of steps
        i.e., EVAL_STEPS, after which the object is swapped.
    - **Distant and surface agents**: We provide configs for both distant and surface
        agents, with 10 random rotations and random noise added to observations.
    - **Rapid prototyping**: By toggling `APPLY_RAPID_CONFIGS`, users can have more
        control over the number of objects and rotations for quicker iteration and
        debugging. This is intended to be removed after RFC 9 is implemented.
"""

# surface agent benchmark configs
unsupervised_inference_distinctobj_surf_agent = copy.deepcopy(
    randrot_noise_10distinctobj_surf_agent
)

# distant agent benchmarks configs
unsupervised_inference_distinctobj_dist_agent = copy.deepcopy(
    randrot_noise_10distinctobj_dist_agent
)


# === Benchmark Configs === #

# Monty Class to use
MONTY_CLASS = MontyForNoResetEvidenceGraphMatching

# LM Class to use
LM_CLASS = NoResetEvidenceGraphLM

# Number of Eval steps
# This will be used for min_eval_steps and max_eval_steps
# because we want to run the evaluation for exactly EVAL_STEPS
EVAL_STEPS = 100

# define surface agent monty configs to set the classes and eval steps.
surf_monty_config = copy.deepcopy(
    unsupervised_inference_distinctobj_surf_agent["monty_config"]
)
surf_monty_config.learning_module_configs["learning_module_0"][
    "learning_module_class"
] = LM_CLASS
surf_monty_config.monty_class = MONTY_CLASS
surf_monty_config.monty_args.min_eval_steps = EVAL_STEPS
unsupervised_inference_distinctobj_surf_agent.update(
    {"monty_config": surf_monty_config}
)
unsupervised_inference_distinctobj_surf_agent[
    "experiment_args"
].max_eval_steps = EVAL_STEPS


# define distant agent monty configs to set the classes and eval steps.
dist_monty_config = copy.deepcopy(
    unsupervised_inference_distinctobj_dist_agent["monty_config"]
)
dist_monty_config.learning_module_configs["learning_module_0"][
    "learning_module_class"
] = LM_CLASS
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
