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

from dataclasses import dataclass

from tbp.monty.frameworks.actions.action_samplers import (
    ActionSampler,
    ConstantSampler,
)
from tbp.monty.frameworks.actions.actions import (
    Action,
    LookDown,
    LookUp,
    MoveForward,
    MoveTangentially,
    OrientHorizontal,
    OrientVertical,
    SetAgentPitch,
    SetAgentPose,
    SetSensorPitch,
    SetSensorPose,
    SetSensorRotation,
    SetYaw,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.agents import AgentID


@dataclass
class BasePolicyConfig:
    """Config for BasePolicy."""

    # conf/experiment/config/monty/motor_system/policy/base.yaml

    action_sampler_args: dict
    action_sampler_class: type[ActionSampler]
    agent_id: AgentID
    file_name: str | None = None
    switch_frequency: float = 0.05


@dataclass
class InformedPolicyConfig:
    # conf/experiment/config/monty/motor_system/policy/informed.yaml
    action_sampler_args: dict
    action_sampler_class: type[ActionSampler]
    agent_id: AgentID
    file_name: str | None = None
    good_view_percentage: float = 0.5
    desired_object_distance: float = 0.03
    use_goal_state_driven_actions: bool = False
    switch_frequency: float = 1.0
    min_perc_on_obj: float = 0.25


@dataclass
class SurfacePolicyConfig(InformedPolicyConfig):
    # conf/experiment/config/monty/motor_system/policy/surface.yaml
    desired_object_distance: float = 0.025
    alpha: float = 0.1


@dataclass
class SurfaceCurveInformedPolicyConfig(SurfacePolicyConfig):
    # conf/experiment/config/monty/motor_system/policy/surface_curve_informed.yaml
    desired_object_distance: float = 0.025
    pc_alpha: float = 0.5
    max_pc_bias_steps: int = 32
    min_general_steps: int = 8
    min_heading_steps: int = 12


def generate_action_list(action_space_type) -> list[Action]:
    """Generate an action list based on a given action space type.

    Args:
        action_space_type: name of action space, one of `"distant_agent"`,
            `"distant_agent_no_translation"`, `"absolute_only"`, or `"surface_agent"`

    Returns:
        Action list to use for the given action space type
    """
    assert action_space_type in [
        "distant_agent",
        "distant_agent_no_translation",
        "absolute_only",
        "surface_agent",
    ]

    actions = []
    if action_space_type == "distant_agent":
        actions = [
            LookUp,
            LookDown,
            TurnLeft,
            TurnRight,
            MoveForward,
            SetAgentPose,
            SetSensorRotation,
        ]
    elif action_space_type == "distant_agent_no_translation":
        actions = [
            LookUp,
            LookDown,
            TurnLeft,
            TurnRight,
            SetAgentPose,
            SetSensorRotation,
        ]
    elif action_space_type == "absolute_only":
        actions = [
            SetAgentPitch,
            SetSensorPitch,
            SetYaw,
            SetAgentPose,
            SetSensorRotation,
            SetSensorPose,
        ]
    elif action_space_type == "surface_agent":
        actions = [
            MoveForward,
            MoveTangentially,
            OrientHorizontal,
            OrientVertical,
            SetAgentPose,
            SetSensorRotation,
        ]
    return actions


def make_base_policy_config(
    # conf/experiment/config/monty/motor_system/policy/base.yaml
    action_space_type: str,
    action_sampler_class: type[ActionSampler],
    agent_id: AgentID = AgentID("agent_id_0"),
):
    """Generates a config that will apply for the BasePolicy class.

    Args:
        action_space_type: name of action space, one of `"distant_agent"`,
            `"distant_agent_no_translation"`, `"absolute_only"`, or `"surface_agent"`
        action_sampler_class: ActionSampler class to use
        agent_id: Agent name. Defaults to "agent_id_0".

    Returns:
        BasePolicyConfig instance
    """
    actions = generate_action_list(action_space_type)

    return BasePolicyConfig(
        action_sampler_args=dict(actions=actions),
        action_sampler_class=action_sampler_class,
        agent_id=agent_id,
    )


def make_curv_surface_policy_config(
    # conf/experiment/config/monty/motor_system/policy/surface_curve_informed.yaml
    desired_object_distance,
    alpha,
    pc_alpha,
    max_pc_bias_steps,
    min_general_steps,
    min_heading_steps,
    use_goal_state_driven_actions=False,
    action_sampler_class: type[ActionSampler] = ConstantSampler,
    action_space_type="surface_agent",
    file_name=None,
    agent_id: AgentID = AgentID("agent_id_0"),
    **kwargs,
):
    """For the SurfacePolicyCurvatureInformed policy.

    Args:
        desired_object_distance: ?
        alpha: ?
        pc_alpha: ?
        max_pc_bias_steps: ?
        min_general_steps: ?
        min_heading_steps: ?
        use_goal_state_driven_actions: Defaults to False
        action_sampler_class: Defaults to ConstantSampler
        action_space_type: Defaults to "surface_agent"
        file_name: Defaults to None
        agent_id: Agent name. Defaults to "agent_id_0".
        **kwargs: Any additional keyword arguments. These may include parameters for
            ActionSampler configuration:
                absolute_degrees,
                max_absolute_degrees,
                min_absolute_degrees,
                direction,
                location,
                rotation_degrees,
                rotation_quat,
                max_rotation_degrees,
                min_rotation_degrees,
                translation_distance,
                max_translation,
                min_translation,

    Returns:
        SurfaceCurveInformedPolicyConfig instance
    """
    actions = generate_action_list(action_space_type)

    return SurfaceCurveInformedPolicyConfig(
        action_sampler_args=dict(**kwargs, actions=actions),
        action_sampler_class=action_sampler_class,
        agent_id=agent_id,
        desired_object_distance=desired_object_distance,
        alpha=alpha,
        pc_alpha=pc_alpha,
        max_pc_bias_steps=max_pc_bias_steps,
        min_general_steps=min_general_steps,
        min_heading_steps=min_heading_steps,
        use_goal_state_driven_actions=use_goal_state_driven_actions,
        file_name=file_name,
    )
