# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy
import os
from dataclasses import dataclass, field
from itertools import product
from numbers import Number
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Union,
)

import numpy as np
import wandb
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.actions.action_samplers import (
    ConstantSampler,
    UniformlyDistributedSampler,
)
from tbp.monty.frameworks.config_utils.policy_setup_utils import (
    make_base_policy_config,
    make_curv_surface_policy_config,
    make_informed_policy_config,
    make_naive_scan_policy_config,
    make_surface_policy_config,
)
from tbp.monty.frameworks.loggers.monty_handlers import (
    BasicCSVStatsHandler,
    DetailedJSONHandler,
    ReproduceEpisodeHandler,
)
from tbp.monty.frameworks.loggers.wandb_handlers import (
    BasicWandbChartStatsHandler,
    BasicWandbTableStatsHandler,
    DetailedWandbMarkedObsHandler,
)
from tbp.monty.frameworks.models.abstract_monty_classes import Monty
from tbp.monty.frameworks.models.displacement_matching import DisplacementGraphLM
from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.graph_matching import MontyForGraphMatching
from tbp.monty.frameworks.models.motor_policies import (
    BasePolicy,
    InformedPolicy,
    NaiveScanPolicy,
    SurfacePolicy,
    SurfacePolicyCurvatureInformed,
)
from tbp.monty.frameworks.models.motor_system import MotorSystem
from tbp.monty.frameworks.models.sensor_modules import (
    HabitatSM,
    Probe,
)
from tbp.monty.frameworks.utils.dataclass_utils import Dataclass

# -- Table of contents --
# -----------------------
# Logging Configurations
# Motor System Configurations
# Monty Configurations
# -----------------------

monty_logs_dir = os.getenv("MONTY_LOGS")


@dataclass
class LoggingConfig:
    monty_log_level: str = "DETAILED"
    monty_handlers: List = field(
        default_factory=lambda: [
            BasicCSVStatsHandler,
            DetailedJSONHandler,
            ReproduceEpisodeHandler,
        ]
    )
    wandb_handlers: list = field(default_factory=list)
    python_log_level: str = "INFO"
    python_log_to_file: bool = True
    python_log_to_stderr: bool = True
    output_dir: str = os.path.expanduser(
        os.path.join(monty_logs_dir, "projects/monty_runs/")
    )
    run_name: str = ""
    resume_wandb_run: Union[bool, str] = False
    wandb_id: str = field(default_factory=wandb.util.generate_id)
    wandb_group: str = "debugging"
    log_parallel_wandb: bool = False


@dataclass
class WandbLoggingConfig(LoggingConfig):
    monty_log_level: str = "BASIC"
    monty_handlers: List = field(
        default_factory=lambda: [
            BasicCSVStatsHandler,
            ReproduceEpisodeHandler,
        ]
    )
    wandb_handlers: List = field(
        default_factory=lambda: [
            BasicWandbTableStatsHandler,
            BasicWandbChartStatsHandler,
        ]
    )
    wandb_group: str = "debugging"


@dataclass
class CSVLoggingConfig(LoggingConfig):
    monty_handlers: List = field(
        default_factory=lambda: [
            BasicCSVStatsHandler,
        ]
    )
    wandb_handlers: List = field(default_factory=lambda: [])


@dataclass
class DetailedWandbLoggingConfig(LoggingConfig):
    monty_handlers: List = field(
        default_factory=lambda: [
            BasicCSVStatsHandler,
            DetailedJSONHandler,
            ReproduceEpisodeHandler,
        ]
    )
    wandb_handlers: List = field(
        default_factory=lambda: [
            BasicWandbTableStatsHandler,
            BasicWandbChartStatsHandler,
            DetailedWandbMarkedObsHandler,
        ]
    )
    wandb_group: str = "debugging"


@dataclass
class EvalLoggingConfig(LoggingConfig):
    output_dir: str = os.path.expanduser(
        os.path.join(monty_logs_dir, "projects/feature_eval_runs/logs")
    )
    monty_handlers: List = field(
        default_factory=lambda: [
            BasicCSVStatsHandler,
            # DetailedJSONHandler,
            ReproduceEpisodeHandler,
        ]
    )
    wandb_handlers: List = field(
        default_factory=lambda: [
            BasicWandbTableStatsHandler,
            BasicWandbChartStatsHandler,
            # DetailedWandbMarkedObsHandler,
        ]
    )
    wandb_group: str = "gm_eval_runs"


@dataclass
class EvalEvidenceLMLoggingConfig(LoggingConfig):
    output_dir: str = os.path.expanduser(
        os.path.join(monty_logs_dir, "projects/evidence_eval_runs/logs")
    )
    monty_handlers: List = field(
        default_factory=lambda: [
            BasicCSVStatsHandler,
            ReproduceEpisodeHandler,
        ]
    )
    wandb_handlers: List = field(
        default_factory=lambda: [
            BasicWandbTableStatsHandler,
            BasicWandbChartStatsHandler,
            # DetailedWandbMarkedObsHandler,
        ]
    )
    wandb_group: str = "evidence_eval_runs"
    monty_log_level: str = "BASIC"


@dataclass
class ParallelEvidenceLMLoggingConfig(LoggingConfig):
    # Config useful for running parallel experiments
    # on lambda-node, i.e. has appropriate wandb flags
    # and parsimonious Python logging
    output_dir: str = os.path.expanduser(
        os.path.join(monty_logs_dir, "projects/evidence_eval_runs/logs")
    )
    monty_handlers: List = field(
        default_factory=lambda: [
            BasicCSVStatsHandler,
            ReproduceEpisodeHandler,
        ]
    )
    wandb_handlers: List = field(
        default_factory=lambda: [
            BasicWandbTableStatsHandler,
            # Note that parallel runs will log a table to wandb no matter if
            # this logger is specified or not
            BasicWandbChartStatsHandler,
        ]
    )
    wandb_group: str = "parallel_eval_runs"  # User to set appropriately
    monty_log_level: str = "BASIC"

    python_log_level: str = "WARNING"
    log_parallel_wandb: bool = True


@dataclass
class DetailedEvidenceLMLoggingConfig(EvalEvidenceLMLoggingConfig):
    monty_handlers: List = field(
        default_factory=lambda: [
            BasicCSVStatsHandler,
            DetailedJSONHandler,
            ReproduceEpisodeHandler,
        ]
    )
    wandb_handlers: list = field(default_factory=list)
    monty_log_level: str = "DETAILED"
    detailed_episodes_to_save: str = "all"
    detailed_save_per_episode: bool = False


@dataclass
class PretrainLoggingConfig(LoggingConfig):
    monty_log_level: str = "SILENT"
    python_log_level: str = "WARNING"
    monty_handlers: List = field(default_factory=list)


# -----
# Motor System Configs
# ----


@dataclass
class MotorSystemConfig:
    motor_system_class: MotorSystem = MotorSystem
    motor_system_args: Union[Dict, Dataclass] = field(
        default_factory=lambda: dict(
            policy_class=BasePolicy,
            policy_args=make_base_policy_config(
                action_space_type="distant_agent",
                action_sampler_class=UniformlyDistributedSampler,
            ),
        )
    )


@dataclass
class MotorSystemConfigRelNoTrans:
    motor_system_class: MotorSystem = MotorSystem
    motor_system_args: Union[Dict, Dataclass] = field(
        default_factory=lambda: dict(
            policy_class=BasePolicy,
            policy_args=make_base_policy_config(
                action_space_type="distant_agent_no_translation",
                action_sampler_class=UniformlyDistributedSampler,
            ),
        )
    )


@dataclass
class MotorSystemConfigInformedNoTrans:
    motor_system_class: MotorSystem = MotorSystem
    motor_system_args: Union[Dict, Dataclass] = field(
        default_factory=lambda: dict(
            policy_class=InformedPolicy,
            policy_args=make_informed_policy_config(
                action_space_type="distant_agent_no_translation",
                action_sampler_class=ConstantSampler,
                rotation_degrees=5.0,
                use_goal_state_driven_actions=False,
            ),
        )
    )


@dataclass
class MotorSystemConfigInformedNoTransStepS3:
    motor_system_class: MotorSystem = MotorSystem
    motor_system_args: Union[Dict, Dataclass] = field(
        default_factory=lambda: dict(
            policy_class=InformedPolicy,
            policy_args=make_informed_policy_config(
                action_space_type="distant_agent_no_translation",
                action_sampler_class=ConstantSampler,
                rotation_degrees=3.0,
                use_goal_state_driven_actions=False,
            ),
        )
    )


@dataclass
class MotorSystemConfigInformedNoTransStepS1:
    motor_system_class: MotorSystem = MotorSystem
    motor_system_args: Union[Dict, Dataclass] = field(
        default_factory=lambda: dict(
            policy_class=InformedPolicy,
            policy_args=make_informed_policy_config(
                action_space_type="distant_agent_no_translation",
                action_sampler_class=ConstantSampler,
                rotation_degrees=1.0,
                use_goal_state_driven_actions=False,
            ),
        )
    )


@dataclass
class MotorSystemConfigInformedNoTransStepS6:
    motor_system_class: MotorSystem = MotorSystem
    motor_system_args: Union[Dict, Dataclass] = field(
        default_factory=lambda: dict(
            policy_class=InformedPolicy,
            policy_args=make_informed_policy_config(
                action_space_type="distant_agent_no_translation",
                action_sampler_class=ConstantSampler,
                rotation_degrees=6.0,
                use_goal_state_driven_actions=False,
            ),
        )
    )


@dataclass
class MotorSystemConfigInformedNoTransStepS20:
    motor_system_class: MotorSystem = MotorSystem
    motor_system_args: Union[Dict, Dataclass] = field(
        default_factory=lambda: dict(
            policy_class=InformedPolicy,
            policy_args=make_informed_policy_config(
                action_space_type="distant_agent_no_translation",
                action_sampler_class=ConstantSampler,
                rotation_degrees=20.0,
                use_goal_state_driven_actions=False,
            ),
        )
    )


@dataclass
class MotorSystemConfigInformedNoTransCloser:
    motor_system_class: MotorSystem = MotorSystem
    motor_system_args: Union[Dict, Dataclass] = field(
        default_factory=lambda: dict(
            policy_class=InformedPolicy,
            policy_args=make_informed_policy_config(
                action_space_type="distant_agent_no_translation",
                action_sampler_class=ConstantSampler,
                rotation_degrees=5.0,
                good_view_percentage=0.7,
                use_goal_state_driven_actions=False,
            ),
        )
    )


@dataclass
class MotorSystemConfigInformedNoTransFurtherAway:
    motor_system_class: MotorSystem = MotorSystem
    motor_system_args: Union[Dict, Dataclass] = field(
        default_factory=lambda: dict(
            policy_class=InformedPolicy,
            policy_args=make_informed_policy_config(
                action_space_type="distant_agent_no_translation",
                action_sampler_class=ConstantSampler,
                rotation_degrees=5.0,
                good_view_percentage=0.3,
                use_goal_state_driven_actions=False,
            ),
        )
    )


@dataclass
class MotorSystemConfigNaiveScanSpiral:
    motor_system_class: MotorSystem = MotorSystem
    motor_system_args: Union[Dict, Dataclass] = field(
        default_factory=lambda: dict(
            policy_class=NaiveScanPolicy,
            policy_args=make_naive_scan_policy_config(step_size=5),
        )
    )


@dataclass
class MotorSystemConfigSurface:
    motor_system_class: MotorSystem = MotorSystem
    motor_system_args: Union[Dict, Dataclass] = field(
        default_factory=lambda: dict(
            policy_class=SurfacePolicy,
            policy_args=make_surface_policy_config(
                desired_object_distance=0.025,  # 2.5 cm desired distance
                alpha=0.1,  # alpha 0.1 means we mostly maintain our heading
                use_goal_state_driven_actions=False,
            ),
        )
    )


@dataclass
class MotorSystemConfigCurvatureInformedSurface:
    motor_system_class: MotorSystem = MotorSystem
    motor_system_args: Union[Dict, Dataclass] = field(
        default_factory=lambda: dict(
            policy_class=SurfacePolicyCurvatureInformed,
            policy_args=make_curv_surface_policy_config(
                desired_object_distance=0.025,
                alpha=0.1,
                pc_alpha=0.5,
                # For a description of the below step parameters, see the class
                # SurfacePolicyCurvatureInformed
                max_pc_bias_steps=32,
                min_general_steps=8,
                min_heading_steps=12,
                use_goal_state_driven_actions=False,
            ),
        )
    )


# Distant-agent ("eye") policy that also performs hypothesis-testing jumps
@dataclass
class MotorSystemConfigInformedGoalStateDriven:
    motor_system_class: MotorSystem = MotorSystem
    motor_system_args: Union[Dict, Dataclass] = field(
        default_factory=lambda: dict(
            policy_class=InformedPolicy,
            policy_args=make_informed_policy_config(
                action_space_type="distant_agent_no_translation",
                action_sampler_class=ConstantSampler,
                rotation_degrees=5.0,
                use_goal_state_driven_actions=True,
            ),
        )
    )


@dataclass
class MotorSystemConfigInformedGoalStateDrivenFartherAway:
    motor_system_class: MotorSystem = MotorSystem
    motor_system_args: Union[Dict, Dataclass] = field(
        default_factory=lambda: dict(
            policy_class=InformedPolicy,
            policy_args=make_informed_policy_config(
                action_space_type="distant_agent_no_translation",
                action_sampler_class=ConstantSampler,
                rotation_degrees=10.0,  # Relatively large step-size
                good_view_percentage=0.5,  # Relatively far from the object
                use_goal_state_driven_actions=True,
            ),
        )
    )


# Curvature-informed, surface-agent ("finger") policy where we also use
# hypothesis-testing jumps
@dataclass
class MotorSystemConfigCurInformedSurfaceGoalStateDriven:
    motor_system_class: MotorSystem = MotorSystem
    motor_system_args: Union[Dict, Dataclass] = field(
        default_factory=lambda: dict(
            policy_class=SurfacePolicyCurvatureInformed,
            policy_args=make_curv_surface_policy_config(
                desired_object_distance=0.025,
                alpha=0.1,
                pc_alpha=0.5,
                max_pc_bias_steps=32,
                min_general_steps=8,
                min_heading_steps=12,
                use_goal_state_driven_actions=True,
            ),
        )
    )


# -------------
# Monty Configurations
# -------------


@dataclass
class MontyArgs:
    """Step-based parameters for Monty configuration.

    Attributes:
        num_exploratory_steps: Number of steps allowed for exploration.  Defaults to
            1000.
        min_eval_steps: Minimum number of evaluation steps. Defaults to 3.
        min_train_steps: Minimum number of training steps. Defaults to 3.
        max_total_steps: Maximum total episode steps before timeout, regardless of
            whether LMs receive sensory information and perform a true matching step.
            Defaults to 2500.
    """

    num_exploratory_steps: int = 1_000
    min_eval_steps: int = 3
    min_train_steps: int = 3
    max_total_steps: int = 2_500


@dataclass
class MontyFeatureGraphArgs(MontyArgs):
    num_exploratory_steps: int = 1_000
    min_eval_steps: int = 3
    min_train_steps: int = 3
    max_total_steps: int = 2_500


@dataclass
class MontyConfig:
    """Use this config to specify a monty architecture in an experiment config.

    The monty_parser code will convert the configs for learning modules etc. into
    instances, and call MontyArgs to instantiate a Monty instance.
    """

    monty_class: Monty
    learning_module_configs: Dict
    sensor_module_configs: Dict
    motor_system_config: Dict
    sm_to_agent_dict: Dict
    sm_to_lm_matrix: Dict
    lm_to_lm_matrix: Dict
    lm_to_lm_vote_matrix: Dict
    monty_args: Union[Dict, MontyArgs]


@dataclass
class PatchAndViewMontyConfig(MontyConfig):
    monty_class: Callable = MontyForGraphMatching
    learning_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict(
            learning_module_0=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            )
        )
    )
    sensor_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    # TODO: would be nicer to just use lm.tolerances.keys() here
                    # but not sure how to easily do this.
                    features=[
                        # morphological features (nescessarry)
                        "pose_vectors",
                        "pose_fully_defined",
                        "on_object",
                        # non-morphological features (optional)
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    save_raw_obs=True,
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=Probe,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=True,
                ),
            ),
        )
    )
    motor_system_config: Union[dataclass, Dict] = field(
        default_factory=MotorSystemConfigInformedNoTrans
    )
    sm_to_agent_dict: Dict = field(
        default_factory=lambda: dict(
            patch="agent_id_0",
            view_finder="agent_id_0",
        )
    )
    sm_to_lm_matrix: List = field(
        default_factory=lambda: [[0]],  # View finder (sm1) not connected to lm
    )
    lm_to_lm_matrix: Optional[List] = None
    lm_to_lm_vote_matrix: Optional[List] = None
    monty_args: Union[Dict, dataclass] = field(default_factory=MontyArgs)


@dataclass
class PatchAndViewSOTAMontyConfig(PatchAndViewMontyConfig):
    """The best existing combination of sensor module and policy attributes.

    Uses the best existing combination of sensor module and policy attributes,
    including the feature-change sensor module, and the hypothesis-testing action
    policy.
    """

    monty_class: Callable = MontyForEvidenceGraphMatching
    sensor_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        # morphological features (nescessarry)
                        "pose_vectors",
                        "pose_fully_defined",
                        "on_object",
                        # non-morphological features (optional)
                        "object_coverage",
                        "hsv",
                        "principal_curvatures_log",
                    ],
                    delta_thresholds={
                        "on_object": 0,
                        "n_steps": 20,
                        "hsv": [0.1, 0.1, 0.1],
                        "pose_vectors": [np.pi / 4, np.pi * 2, np.pi * 2],
                        "principal_curvatures_log": [2, 2],
                        "distance": 0.01,
                    },
                    save_raw_obs=False,
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=Probe,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        )
    )
    motor_system_config: Union[dataclass, Dict] = field(
        default_factory=MotorSystemConfigInformedGoalStateDriven
    )


@dataclass
class PatchAndViewFartherAwaySOTAMontyConfig(PatchAndViewSOTAMontyConfig):
    """PatchAndViewSOTAMontyConfig with a farther away target object and "saccades".

    Uses the best existing combination of sensor module and policy attributes,
    including the feature-change sensor module, and the hypothesis-testing action
    policy, but while maintaining a larger distance to the target object, and performing
    larger "saccades".
    Useful for testing how the policy deals with multiple objects
    """

    motor_system_config: Union[dataclass, Dict] = field(
        default_factory=MotorSystemConfigInformedGoalStateDrivenFartherAway
    )


@dataclass
class SurfaceAndViewMontyConfig(PatchAndViewMontyConfig):
    sensor_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatSM,
                sensor_module_args=dict(
                    is_surface_sm=True,
                    sensor_module_id="patch",
                    features=[
                        # morphological features (nescessarry)
                        "pose_vectors",
                        "pose_fully_defined",
                        "on_object",
                        # non-morphological features (optional)
                        "object_coverage",
                        "min_depth",
                        "mean_depth",
                        "hsv",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    save_raw_obs=True,
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=Probe,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=True,
                ),
            ),
        )
    )
    motor_system_config: Union[dataclass, Dict] = field(
        default_factory=MotorSystemConfigSurface
    )
    sm_to_agent_dict: Dict = field(
        default_factory=lambda: dict(
            patch="agent_id_0",
            view_finder="agent_id_0",
        )
    )
    sm_to_lm_matrix: List = field(
        default_factory=lambda: [[0]],  # View finder (sm1) not connected to lm
    )
    lm_to_lm_matrix: Optional[List] = None
    lm_to_lm_vote_matrix: Optional[List] = None
    monty_args: Union[Dict, dataclass] = field(default_factory=MontyArgs)


@dataclass
class SurfaceAndViewSOTAMontyConfig(SurfaceAndViewMontyConfig):
    """The best existing combination of sensor module and policy attributes.

    Uses the best existing combination of sensor module and policy attributes,
    including the feature-change sensor module, the curvature-informed surface policy,
    and the hypothesis-testing action policy.
    """

    monty_class: Callable = MontyForEvidenceGraphMatching
    sensor_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        # morphological features (nescessarry)
                        "pose_vectors",
                        "pose_fully_defined",
                        "on_object",
                        # non-morphological features (optional)
                        "object_coverage",
                        "min_depth",
                        "mean_depth",
                        "hsv",
                        "principal_curvatures",
                        "principal_curvatures_log",
                    ],
                    delta_thresholds={
                        "on_object": 0,
                        "n_steps": 20,
                        "hsv": [0.1, 0.1, 0.1],
                        "pose_vectors": [np.pi / 4, np.pi * 2, np.pi * 2],
                        "principal_curvatures_log": [2, 2],
                        "distance": 0.01,
                    },
                    is_surface_sm=True,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=Probe,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        )
    )
    motor_system_config: Union[dataclass, Dict] = field(
        default_factory=MotorSystemConfigCurInformedSurfaceGoalStateDriven
    )


@dataclass
class PatchAndViewFeatureChangeConfig(PatchAndViewMontyConfig):
    sensor_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatSM,
                sensor_module_args=dict(
                    sensor_module_id="patch",
                    features=[
                        # morphological features (nescessarry)
                        "pose_vectors",
                        "pose_fully_defined",
                        "on_object",
                        # non-morphological features (optional)
                        "object_coverage",
                        "rgba",
                        "hsv",
                        "principal_curvatures",
                        "principal_curvatures_log",
                        "gaussian_curvature",
                        "mean_curvature",
                        "gaussian_curvature_sc",
                        "mean_curvature_sc",
                    ],
                    # here we don't have +- the th like with the tolerances
                    # but just the distance shouldn't be > th. Maybe we should
                    # make this the same.
                    delta_thresholds={
                        "on_object": 0,
                        "hsv": [0.2, 1, 1],
                        "principal_curvatures_log": [2, 2],
                        "distance": 0.05,
                    },
                    save_raw_obs=True,
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=Probe,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=True,
                ),
            ),
        )
    )
    motor_system_config: Union[dataclass, Dict] = field(
        default_factory=MotorSystemConfigNaiveScanSpiral
    )


# For voting
features = [
    "on_object",
    "rgba",
    "hsv",
    "pose_vectors",
    "principal_curvatures",
    "principal_curvatures_log",
    "gaussian_curvature",
    "mean_curvature",
    "gaussian_curvature_sc",
    "mean_curvature_sc",
]


@dataclass
class TwoLMMontyConfig(MontyConfig):
    monty_class: Callable = MontyForGraphMatching
    learning_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict(
            learning_module_0=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
            learning_module_1=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
        )
    )
    sensor_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_0",
                    features=features,
                    save_raw_obs=True,
                ),
            ),
            sensor_module_1=dict(
                sensor_module_class=HabitatSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_1",
                    features=features,
                    save_raw_obs=True,
                ),
            ),
            sensor_module_2=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=Probe,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=True,
                ),
            ),
        )
    )
    motor_system_config: Union[dataclass, Dict] = field(
        default_factory=MotorSystemConfigInformedNoTrans
    )
    sm_to_agent_dict: Dict = field(
        default_factory=lambda: dict(
            patch_0="agent_id_0",
            patch_1="agent_id_0",
            view_finder="agent_id_0",
        )
    )
    sm_to_lm_matrix: List = field(
        default_factory=lambda: [[0], [1]],  # View finder (sm2) not connected to lm
    )
    lm_to_lm_matrix: Optional[List] = None
    lm_to_lm_vote_matrix: List = field(default_factory=lambda: [[1], [0]])
    monty_args: Union[Dict, dataclass] = field(default_factory=MontyArgs)


@dataclass
class TwoLMStackedMontyConfig(TwoLMMontyConfig):
    monty_class: Callable = MontyForEvidenceGraphMatching
    learning_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict(
            learning_module_0=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
            learning_module_1=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
        )
    )
    sensor_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_0",
                    features=[
                        # morphological features (nescessarry)
                        "pose_vectors",
                        "pose_fully_defined",
                        "on_object",
                        # non-morphological features (optional)
                        "object_coverage",
                        "min_depth",
                        "mean_depth",
                        "hsv",
                        "principal_curvatures_log",
                    ],
                    delta_thresholds={
                        "on_object": 0,
                        "n_steps": 20,
                        "hsv": [0.1, 0.1, 0.1],
                        "pose_vectors": [np.pi / 4, np.pi * 2, np.pi * 2],
                        "principal_curvatures_log": [2, 2],
                        "distance": 0.01,
                    },
                    save_raw_obs=True,
                ),
            ),
            sensor_module_1=dict(
                sensor_module_class=HabitatSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_1",
                    features=[
                        # morphological features (nescessarry)
                        "pose_vectors",
                        "pose_fully_defined",
                        "on_object",
                        # non-morphological features (optional)
                        "object_coverage",
                        "hsv",
                        "principal_curvatures_log",
                    ],
                    delta_thresholds={
                        "on_object": 0,
                        "n_steps": 100,
                        "hsv": [0.2, 0.2, 0.2],
                        "pose_vectors": [np.pi / 4, np.pi * 2, np.pi * 2],
                        "principal_curvatures_log": [4, 4],
                        "distance": 0.05,
                    },
                    save_raw_obs=True,
                ),
            ),
            sensor_module_2=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=Probe,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=True,
                ),
            ),
        )
    )
    sm_to_lm_matrix: List = field(
        default_factory=lambda: [
            [0],
            [1],
        ],  # View finder (sm2) not connected to lm
    )
    # First LM only gets sensory input, second gets input from first + sensor
    lm_to_lm_matrix: Optional[List] = field(default_factory=lambda: [[], [0]])
    lm_to_lm_vote_matrix: Optional[List] = None


@dataclass
class FiveLMMontyConfig(MontyConfig):
    monty_class: Callable = MontyForGraphMatching
    learning_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict(
            learning_module_0=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
            learning_module_1=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
            learning_module_2=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
            learning_module_3=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
            learning_module_4=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(k=5, match_attribute="displacement"),
            ),
        )
    )
    sensor_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict(
            sensor_module_0=dict(
                sensor_module_class=HabitatSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_0",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_1=dict(
                sensor_module_class=HabitatSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_1",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_2=dict(
                sensor_module_class=HabitatSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_2",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_3=dict(
                sensor_module_class=HabitatSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_3",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_4=dict(
                sensor_module_class=HabitatSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_4",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_5=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=Probe,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        )
    )
    motor_system_config: Union[dataclass, Dict] = field(
        default_factory=MotorSystemConfigInformedNoTrans
    )
    sm_to_agent_dict: Dict = field(
        default_factory=lambda: dict(
            patch_0="agent_id_0",
            patch_1="agent_id_0",
            patch_2="agent_id_0",
            patch_3="agent_id_0",
            patch_4="agent_id_0",
            view_finder="agent_id_0",
        )
    )
    sm_to_lm_matrix: List = field(
        default_factory=lambda: [
            [0],
            [1],
            [2],
            [3],
            [4],
        ],  # View finder (sm2) not connected to lm
    )
    lm_to_lm_matrix: Optional[List] = None
    # lm_to_lm_vote_matrix: Optional[List] = None
    # All LMs connect to each other
    lm_to_lm_vote_matrix: List = field(
        default_factory=lambda: [
            [1, 2, 3, 4],
            [0, 2, 3, 4],
            [0, 1, 3, 4],
            [0, 1, 2, 4],
            [0, 1, 2, 3],
        ]
    )
    monty_args: Union[Dict, dataclass] = field(default_factory=MontyArgs)


@dataclass
class FiveLMMontySOTAConfig(FiveLMMontyConfig):
    motor_system_config: Union[dataclass, Dict] = field(
        default_factory=MotorSystemConfigInformedGoalStateDriven
    )


"""
Multi-LM Config Utils.
"""


def make_multi_lm_flat_dense_connectivity(n_lms: int) -> Dict:
    """Create flat, dense connectivity matrices for a multi-LM experiment.

    Generates connectivity matrices for a `MontyConfig` with multiple sensor
    and learning modules where learning modules are not hierarchically connected
    ('flat'), and voting is all-to-all ('dense'). This assumes each LM is connected
    to a single upstream sensor module in 1:1 fashion, and all sensor modules are
    mounted on a single agent.

    Args:
        n_lms: Number of LMs. It is assumed that the number of sensor modules (not
            including a view finder) is equal to the number of LMs.

    Returns:
        A dictionary with keys "sm_to_agent_dict", "sm_to_lm_matrix",
            "lm_to_lm_matrix", and "lm_to_lm_vote_matrix".
    """
    # Create default sm_to_lm_matrix: all sensors are on 'agent_id_0'.
    sm_to_agent_dict = {f"patch_{i}": f"agent_id_0" for i in range(n_lms)}
    sm_to_agent_dict["view_finder"] = "agent_id_0"

    # Create default sm_to_lm_matrix: each sensor connects to one LM.
    sm_to_lm_matrix = [[i] for i in range(n_lms)]

    # Create default lm_to_lm_matrix: no LM hierarchy.
    lm_to_lm_matrix = None

    # Create default lm_to_lm_vote_matrix: all-to-all voting.
    lm_to_lm_vote_matrix = []
    for i in range(n_lms):
        lst = list(range(n_lms))
        lst.remove(i)
        lm_to_lm_vote_matrix.append(lst)

    return {
        "sm_to_agent_dict": sm_to_agent_dict,
        "sm_to_lm_matrix": sm_to_lm_matrix,
        "lm_to_lm_matrix": lm_to_lm_matrix,
        "lm_to_lm_vote_matrix": lm_to_lm_vote_matrix,
    }


def make_multi_lm_monty_config(
    n_lms: int,
    *,
    monty_class: type,
    learning_module_class: type,
    learning_module_args: Optional[Mapping],
    sensor_module_class: type,
    sensor_module_args: Optional[Mapping],
    motor_system_class: type,
    motor_system_args: Optional[Mapping],
    monty_args: Optional[Union[Mapping, MontyArgs]],
    connectivity_func: Callable[[int], Mapping] = make_multi_lm_flat_dense_connectivity,
    view_finder_config: Optional[Mapping] = None,
) -> MontyConfig:
    """Create a monty config for multi-LM experiments.

    Creates a complete monty config for a multi-LM experiment.

    This function primarily duplicates learning and sensor module configs and connects
    them, and it uses the following conventions:
        - A `sensor_module_id` is of the form `"patch_{i}"` except for the view
          finder which always has the ID `"view_finder"`.
        - IMPORTANT: A reference to a sensor module with ID "patch" is a placeholder,
          and it will be replaced by some `"patch_{i}"`. For example, learning
          module args may contain parameters that reference a sensor module like so:
          ```python
            learning_module_args = dict(
                tolerances={
                    "patch": {
                        "hsv": np.array([0.1, 0.2, 0.2]),
                        "principal_curvatures_log": np.ones(2),
                    }
                },
                feature_weights={
                    "patch": {
                        "hsv": np.array([1, 0.5, 0.5]),
                    }
                }
            }
            ```
          When we are constructing the config for learning module **i**, that entry
          for `"patch"` will be replaced with `"patch_{i}"`. This is currently
          done for three possible items in `learning_module_args`:
          `"graph_delta_thresholds"`, `"tolerances"`, and `"feature_weights"`.


    Args:
        n_lms: Number of learning modules.
        monty_class: Monty class.
        learning_module_class: Learning module class.
        learning_module_args: Arguments for learning modules.
        sensor_module_class: Sensor module class.
        sensor_module_args: Arguments for sensor modules.
        motor_system_class: Motor system class.
        motor_system_args: Arguments for motor system.
        monty_args: Arguments for monty.
        connectivity_func: Function that returns a
            dictionary of connectivity matrices given a number of learning modules.
            In particular, it must return a dictionary with keys "sm_to_agent_dict",
            "sm_to_lm_matrix", "lm_to_lm_matrix", and "lm_to_lm_vote_matrix". Defaults
            to `make_multi_lm_flat_dense_connectivity`.
        view_finder_config: A mapping which contains the items
            `"sensor_module_class"` and `"sensor_module_args"`. If not specified,
            a config is added using the class `Probe` with  `"view_finder"`
            as the `sensor_module_id`. `"save_raw_obs"` will default to match the
            value in `sensor_module_args` and `False` if none was provided.

    Returns:
        A complete monty config for multi-LM experiment.
    """
    # Make learning module configs.
    if learning_module_args is None:
        learning_module_args = {}
    learning_module_configs = {}
    for i in range(n_lms):
        lm_args_i = copy.deepcopy(learning_module_args)
        # Rename specs keyed with "patch" to "patch_{i}".
        for name in ["graph_delta_thresholds", "tolerances", "feature_weights"]:
            if name in lm_args_i:
                spec = lm_args_i[name]
                if "patch" in spec:
                    spec[f"patch_{i}"] = spec.pop("patch")
        learning_module_configs[f"learning_module_{i}"] = {
            "learning_module_class": learning_module_class,
            "learning_module_args": lm_args_i,
        }

    # Make sensor module configs.
    if sensor_module_args is None:
        sensor_module_args = {}
    sensor_module_configs = {}
    for i in range(n_lms):
        sm_args_i = copy.deepcopy(sensor_module_args)
        sm_args_i["sensor_module_id"] = f"patch_{i}"
        sensor_module_configs[f"sensor_module_{i}"] = {
            "sensor_module_class": sensor_module_class,
            "sensor_module_args": sm_args_i,
        }
    if view_finder_config is None:
        sensor_module_configs["view_finder"] = {
            "sensor_module_class": Probe,
            "sensor_module_args": {
                "sensor_module_id": "view_finder",
                "save_raw_obs": sensor_module_args.get("save_raw_obs", False),
            },
        }
    else:
        sensor_module_configs["view_finder"] = copy.deepcopy(view_finder_config)

    # Make motor system config.
    if motor_system_args is None:
        motor_system_args = {}
    else:
        motor_system_args = copy.deepcopy(motor_system_args)
    motor_system_config = {
        "motor_system_class": motor_system_class,
        "motor_system_args": motor_system_args,
    }

    connectivity = connectivity_func(n_lms)

    if monty_args is None:
        monty_args = MontyArgs()
    elif isinstance(monty_args, MontyArgs):
        monty_args = copy.deepcopy(monty_args)
    else:
        monty_args = MontyArgs(**monty_args)

    monty_config = MontyConfig(
        monty_class=monty_class,
        learning_module_configs=learning_module_configs,
        sensor_module_configs=sensor_module_configs,
        motor_system_config=motor_system_config,
        sm_to_agent_dict=connectivity["sm_to_agent_dict"],
        sm_to_lm_matrix=connectivity["sm_to_lm_matrix"],
        lm_to_lm_matrix=connectivity["lm_to_lm_matrix"],
        lm_to_lm_vote_matrix=connectivity["lm_to_lm_vote_matrix"],
        monty_args=monty_args,
    )
    return monty_config


def get_possible_3d_rotations(
    degrees: Iterable[Number],
    displacement: Number = 0,
) -> List[np.ndarray]:
    """Get list of 24 unique 3d rotations that tile the space. Used for configs.

    Args:
        degrees: Sequence of degrees to sample from.
        displacement: Additional offset (in degrees) to apply to all rotations;
            useful if want to e.g. tile a similar space at training and evaluation, but
            slightly offset between these settings.

    Returns:
        List of unique 3D rotations in euler angles (degrees).

    """
    # Generate all possible 3D rotations (non-unique)
    all_poses = [np.array(p) for p in product(degrees, degrees, degrees)]

    # Apply displacement, and get poses modulo 360.
    all_poses = [(p + displacement) % 360 for p in all_poses]

    # Remove equivalent rotations
    unique_poses, dual_quats = [], []
    for pose in all_poses:
        quat = Rotation.from_euler("xyz", pose, degrees=True).as_quat()
        if not any(np.allclose(quat, q) for q in dual_quats):
            # Store unique pose and the two equivalent quaternions.
            unique_poses.append(pose)
            dual_quats.append(quat)
            dual_quats.append(-quat)

    return unique_poses


def get_cube_face_and_corner_views_rotations() -> List[np.ndarray]:
    """Get 14 rotations that correspond to the 6 cube faces and 8 cube corners.

    If we imagine an object enclosed in an invisible cube, then we can form 6 unique
    views of the object by looking through the cube faces. To get even better coverage
    of the object, we can also look at the object from each corners of the cube. The
    rotations returned here rotate the object 14 ways to obtain such views.

    Returns:
        List of 3d rotations.
    """
    return [
        np.array([0, 0, 0]),
        np.array([0, 90, 0]),
        np.array([0, 180, 0]),
        np.array([0, 270, 0]),
        np.array([90, 0, 0]),
        np.array([90, 180, 0]),
        np.array([35, 45, 0]),
        np.array([325, 45, 0]),
        np.array([35, 315, 0]),
        np.array([325, 315, 0]),
        np.array([35, 135, 0]),
        np.array([325, 135, 0]),
        np.array([35, 225, 0]),
        np.array([325, 225, 0]),
    ]
