# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
from dataclasses import dataclass, field
from itertools import permutations
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import wandb

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
from tbp.monty.frameworks.models.evidence_matching import (
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.graph_matching import MontyForGraphMatching
from tbp.monty.frameworks.models.monty_base import (
    LearningModuleBase,
    MontyBase,
    SensorModuleBase,
)
from tbp.monty.frameworks.models.motor_policies import (
    BasePolicy,
    InformedPolicy,
    MotorSystem,
    NaiveScanPolicy,
    SurfacePolicy,
    SurfacePolicyCurvatureInformed,
)
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    FeatureChangeSM,
    HabitatDistantPatchSM,
    HabitatSurfacePatchSM,
)

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
    python_log_to_stdout: bool = True
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
    motor_system_class: MotorSystem = BasePolicy
    motor_system_args: Union[Dict, dataclass] = field(
        default_factory=lambda: make_base_policy_config(
            action_space_type="distant_agent",
            action_sampler_class=UniformlyDistributedSampler,
        )
    )


@dataclass
class MotorSystemConfigRelNoTrans:
    motor_system_class: MotorSystem = BasePolicy
    motor_system_args: Union[Dict, dataclass] = field(
        default_factory=lambda: make_base_policy_config(
            action_space_type="distant_agent_no_translation",
            action_sampler_class=UniformlyDistributedSampler,
        )
    )


@dataclass
class MotorSystemConfigInformedNoTrans:
    motor_system_class: MotorSystem = InformedPolicy
    motor_system_args: Union[Dict, dataclass] = field(
        default_factory=lambda: make_informed_policy_config(
            action_space_type="distant_agent_no_translation",
            action_sampler_class=ConstantSampler,
            rotation_degrees=5.0,
            use_goal_state_driven_actions=False,
        )
    )


@dataclass
class MotorSystemConfigInformedNoTransStepS3:
    motor_system_class: MotorSystem = InformedPolicy
    motor_system_args: Union[Dict, dataclass] = field(
        default_factory=lambda: make_informed_policy_config(
            action_space_type="distant_agent_no_translation",
            action_sampler_class=ConstantSampler,
            rotation_degrees=3.0,
            use_goal_state_driven_actions=False,
        )
    )


@dataclass
class MotorSystemConfigInformedNoTransStepS1:
    motor_system_class: MotorSystem = InformedPolicy
    motor_system_args: Union[Dict, dataclass] = field(
        default_factory=lambda: make_informed_policy_config(
            action_space_type="distant_agent_no_translation",
            action_sampler_class=ConstantSampler,
            rotation_degrees=1.0,
            use_goal_state_driven_actions=False,
        )
    )


@dataclass
class MotorSystemConfigInformedNoTransStepS6:
    motor_system_class: MotorSystem = InformedPolicy
    motor_system_args: Union[Dict, dataclass] = field(
        default_factory=lambda: make_informed_policy_config(
            action_space_type="distant_agent_no_translation",
            action_sampler_class=ConstantSampler,
            rotation_degrees=6.0,
            use_goal_state_driven_actions=False,
        )
    )


@dataclass
class MotorSystemConfigInformedNoTransStepS20:
    motor_system_class: MotorSystem = InformedPolicy
    motor_system_args: Union[Dict, dataclass] = field(
        default_factory=lambda: make_informed_policy_config(
            action_space_type="distant_agent_no_translation",
            action_sampler_class=ConstantSampler,
            rotation_degrees=20.0,
            use_goal_state_driven_actions=False,
        )
    )


@dataclass
class MotorSystemConfigInformedNoTransCloser:
    motor_system_class: MotorSystem = InformedPolicy
    motor_system_args: Union[Dict, dataclass] = field(
        default_factory=lambda: make_informed_policy_config(
            action_space_type="distant_agent_no_translation",
            action_sampler_class=ConstantSampler,
            rotation_degrees=5.0,
            good_view_percentage=0.7,
            use_goal_state_driven_actions=False,
        )
    )


@dataclass
class MotorSystemConfigInformedNoTransFurtherAway:
    motor_system_class: MotorSystem = InformedPolicy
    motor_system_args: Union[Dict, dataclass] = field(
        default_factory=lambda: make_informed_policy_config(
            action_space_type="distant_agent_no_translation",
            action_sampler_class=ConstantSampler,
            rotation_degrees=5.0,
            good_view_percentage=0.3,
            use_goal_state_driven_actions=False,
        )
    )


@dataclass
class MotorSystemConfigNaiveScanSpiral:
    motor_system_class: MotorSystem = NaiveScanPolicy
    motor_system_args: Union[Dict, dataclass] = field(
        default_factory=lambda: make_naive_scan_policy_config(step_size=5)
    )


@dataclass
class MotorSystemConfigSurface:
    motor_system_class: MotorSystem = SurfacePolicy
    motor_system_args: Union[Dict, dataclass] = field(
        default_factory=lambda: make_surface_policy_config(
            desired_object_distance=0.025,  # 2.5 cm desired distance
            alpha=0.1,  # alpha 0.1 means we mostly maintain our heading
            use_goal_state_driven_actions=False,
        )
    )


@dataclass
class MotorSystemConfigCurvatureInformedSurface:
    motor_system_class: MotorSystem = SurfacePolicyCurvatureInformed
    motor_system_args: Union[Dict, dataclass] = field(
        default_factory=lambda: make_curv_surface_policy_config(
            desired_object_distance=0.025,
            alpha=0.1,
            pc_alpha=0.5,
            # For a description of the below step parameters, see the class
            # SurfacePolicyCurvatureInformed
            max_pc_bias_steps=32,
            min_general_steps=8,
            min_heading_steps=12,
            use_goal_state_driven_actions=False,
        )
    )


# Distant-agent ("eye") policy that also performs hypothesis-testing jumps
@dataclass
class MotorSystemConfigInformedGoalStateDriven:
    motor_system_class: MotorSystem = InformedPolicy
    motor_system_args: Union[Dict, dataclass] = field(
        default_factory=lambda: make_informed_policy_config(
            action_space_type="distant_agent_no_translation",
            action_sampler_class=ConstantSampler,
            rotation_degrees=5.0,
            use_goal_state_driven_actions=True,
        )
    )


@dataclass
class MotorSystemConfigInformedGoalStateDrivenFartherAway:
    motor_system_class: MotorSystem = InformedPolicy
    motor_system_args: Union[Dict, dataclass] = field(
        default_factory=lambda: make_informed_policy_config(
            action_space_type="distant_agent_no_translation",
            action_sampler_class=ConstantSampler,
            rotation_degrees=10.0,  # Relatively large step-size
            good_view_percentage=0.5,  # Relatively far from the object
            use_goal_state_driven_actions=True,
        )
    )


# Curvature-informed, surface-agent ("finger") policy where we also use
# hypothesis-testing jumps
@dataclass
class MotorSystemConfigCurInformedSurfaceGoalStateDriven:
    motor_system_class: MotorSystem = SurfacePolicyCurvatureInformed
    motor_system_args: Union[Dict, dataclass] = field(
        default_factory=lambda: make_curv_surface_policy_config(
            desired_object_distance=0.025,
            alpha=0.1,
            pc_alpha=0.5,
            max_pc_bias_steps=32,
            min_general_steps=8,
            min_heading_steps=12,
            use_goal_state_driven_actions=True,
        )
    )


# -------------
# Monty Configurations
# -------------


@dataclass
class MontyArgs:
    # Step based parameters
    num_exploratory_steps: int = 1_000
    min_eval_steps: int = 3
    min_train_steps: int = 3
    max_total_steps: int = (
        2_500  # Total number of episode steps that can be taken before
    )
    # timing out, regardless of e.g. whether LMs receive sensory information and
    # therefore perform a true matching step


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
class SingleCameraMontyConfig(MontyConfig):
    monty_class: Callable = MontyBase
    learning_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict(
            learning_module_1=dict(
                learning_module_class=LearningModuleBase,
                learning_module_args=dict(),
            )
        )
    )
    sensor_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict(
            sensor_module_0=dict(
                sensor_module_class=SensorModuleBase,
                sensor_module_args=dict(sensor_module_id="sensor_id_0"),
            ),
        )
    )
    motor_system_config: Union[dataclass, Dict] = field(
        default_factory=MotorSystemConfig
    )
    sm_to_agent_dict: Dict = field(
        default_factory=lambda: dict(sensor_id_0="agent_id_0")
    )
    sm_to_lm_matrix: List = field(default_factory=lambda: [[0]])
    lm_to_lm_matrix: Optional[List] = None
    lm_to_lm_vote_matrix: Optional[List] = None
    monty_args: Union[Dict, MontyArgs] = field(default_factory=MontyArgs)


@dataclass
class BaseMountMontyConfig(MontyConfig):
    monty_class: Callable = MontyBase
    learning_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict(
            learning_module_0=dict(
                learning_module_class=LearningModuleBase,
                learning_module_args=dict(),
            ),
            learning_module_1=dict(
                learning_module_class=LearningModuleBase,
                learning_module_args=dict(),
            ),
        )
    )
    sensor_module_configs: Union[dataclass, Dict] = field(
        default_factory=lambda: dict(
            sensor_module_0=dict(
                sensor_module_class=SensorModuleBase,
                sensor_module_args=dict(sensor_module_id="sensor_id_0"),
            ),
            sensor_module_1=dict(
                sensor_module_class=SensorModuleBase,
                sensor_module_args=dict(sensor_module_id="sensor_id_1"),
            ),
        )
    )
    motor_system_config: Union[dataclass, Dict] = field(
        default_factory=MotorSystemConfig
    )
    sm_to_agent_dict: Dict = field(
        # TODO: move SAM to config args?
        default_factory=lambda: dict(
            sensor_id_0="agent_id_0",
            sensor_id_1="agent_id_0",
        )
    )
    sm_to_lm_matrix: List = field(default_factory=lambda: [[0], [1]])
    lm_to_lm_matrix: Optional[List] = None
    lm_to_lm_vote_matrix: Optional[List] = None
    monty_args: Union[Dict, MontyArgs] = field(default_factory=MontyArgs)


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
                sensor_module_class=HabitatDistantPatchSM,
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
                sensor_module_class=DetailedLoggingSM,
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
                sensor_module_class=FeatureChangeSM,
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
                sensor_module_class=DetailedLoggingSM,
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
                sensor_module_class=HabitatSurfacePatchSM,
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
                sensor_module_class=DetailedLoggingSM,
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
                sensor_module_class=FeatureChangeSM,
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
                    surf_agent_sm=True,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_1=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=DetailedLoggingSM,
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
                sensor_module_class=FeatureChangeSM,
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
                sensor_module_class=DetailedLoggingSM,
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
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_0",
                    features=features,
                    save_raw_obs=True,
                ),
            ),
            sensor_module_1=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_1",
                    features=features,
                    save_raw_obs=True,
                ),
            ),
            sensor_module_2=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=DetailedLoggingSM,
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
                sensor_module_class=FeatureChangeSM,
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
                sensor_module_class=FeatureChangeSM,
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
                sensor_module_class=DetailedLoggingSM,
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
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_0",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_1=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_1",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_2=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_2",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_3=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_3",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_4=dict(
                sensor_module_class=HabitatDistantPatchSM,
                sensor_module_args=dict(
                    sensor_module_id="patch_4",
                    features=features,
                    save_raw_obs=False,
                ),
            ),
            sensor_module_5=dict(
                # No need to extract features from the view finder since it is not
                # connected to a learning module (just used at beginning of episode)
                sensor_module_class=DetailedLoggingSM,
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


def get_possible_3d_rotations(degrees, displacement=0):
    """Get list of 3d rotations that tile the space. Used for configs.

    Args:
        degrees: List of degrees to sample from.
        displacement: Additional offset (in degrees) to apply to all rotations;
            useful if want to e.g. tile a similar space at training and evaluation,
            but slightly offset between these settings

    Returns:
        List of 3d rotations that tile the space.
    """
    all_degrees = np.hstack([degrees, degrees, degrees])
    all_poses = list(permutations(all_degrees, 3))
    all_poses = np.unique(all_poses, axis=0)
    all_poses = (all_poses + displacement) % 360

    # Make sure we remove poses that are equivalent (in euler angles (a, b, c) ==
    # (a + 180, -b + 180, c + 180) -> permutations used above will generate duplicates)
    unique_poses = []
    dual_poses = []
    for pose in all_poses:
        dual_pose = np.array(
            [
                (pose[0] + 180) % 360,
                (-pose[1] + 180) % 360,
                (pose[2] + 180) % 360,
            ]
        )

        if list(pose) not in dual_poses:
            unique_poses.append(pose)
            dual_poses.append(list(dual_pose))

    return unique_poses
