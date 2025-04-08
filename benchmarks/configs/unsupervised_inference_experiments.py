# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import copy
from dataclasses import asdict

from benchmarks.configs.names import UnsupervisedInferenceExperiments
from benchmarks.configs.ycb_experiments import (
    randrot_noise_10distinctobj_dist_agent,
    randrot_noise_10distinctobj_surf_agent,
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

experiments = UnsupervisedInferenceExperiments(
    unsupervised_inference_distinctobj_surf_agent=unsupervised_inference_distinctobj_surf_agent,
    unsupervised_inference_distinctobj_dist_agent=unsupervised_inference_distinctobj_dist_agent,
)
CONFIGS = asdict(experiments)
