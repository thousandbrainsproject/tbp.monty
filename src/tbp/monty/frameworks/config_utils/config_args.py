# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from dataclasses import dataclass, field

from tbp.monty.frameworks.config_utils.policy_setup_utils import (
    make_curv_surface_policy_config,
)
from tbp.monty.frameworks.models.motor_policies import SurfacePolicyCurvatureInformed
from tbp.monty.frameworks.models.motor_system import MotorSystem
from tbp.monty.frameworks.utils.dataclass_utils import Dataclass


@dataclass
class MotorSystemConfigCurvatureInformedSurface:
    motor_system_class: MotorSystem = MotorSystem
    motor_system_args: dict | Dataclass = field(
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
