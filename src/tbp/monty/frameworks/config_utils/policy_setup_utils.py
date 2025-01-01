# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from dataclasses import dataclass
from typing import Dict, List, Optional, Type

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


@dataclass
class BasePolicyConfig:
    """Config for BasePolicy."""

    action_sampler_args: Dict
    action_sampler_class: Type[ActionSampler]
    agent_id: str
    file_name: Optional[str] = None
    switch_frequency: float = 0.05


@dataclass
class InformedPolicyConfig:
    action_sampler_args: Dict
    action_sampler_class: Type[ActionSampler]
    agent_id: str
    file_name: Optional[str] = None
    good_view_percentage: float = 0.5
    desired_object_distance: float = 0.03
    use_goal_state_driven_actions: bool = False
    switch_frequency: float = 1.0
    min_perc_on_obj: float = 0.25


@dataclass
class NaiveScanPolicyConfig(InformedPolicyConfig):
    use_goal_state_driven_actions: bool = False
    fixed_amount: float = 3.0


@dataclass
class SurfacePolicyConfig(InformedPolicyConfig):
    desired_object_distance: float = 0.025
    alpha: float = 0.1


@dataclass
class SurfaceCurveInformedPolicyConfig(SurfacePolicyConfig):
    desired_object_distance: float = 0.025
    pc_alpha: float = 0.5
    max_pc_bias_steps: int = 32
    min_general_steps: int = 8
    min_heading_steps: int = 12


def generate_action_list(action_space_type) -> List[Action]:
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
    action_space_type: str,
    action_sampler_class: Type[ActionSampler],
    agent_id: str = "agent_id_0",
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


def make_informed_policy_config(
    action_space_type: str,
    action_sampler_class: Type[ActionSampler],
    good_view_percentage: float = 0.5,
    use_goal_state_driven_actions: bool = False,
    file_name: str = None,
    agent_id: str = "agent_id_0",
    switch_frequency: float = 1.0,
    **kwargs,
):
    """Similar to BasePolicyConfigGenerator, but for InformedPolicy class.

    Args:
        action_space_type: name of action space, one of `"distant_agent"`,
            `"distant_agent_no_translation"`, `"absolute_only"`, or `"surface_agent"`
        action_sampler_class: ActionSampler class to use
        good_view_percentage: Defaults to 0.5
        use_goal_state_driven_actions: Defaults to False
        file_name: Defaults to None
        agent_id: Agent name. Defaults to "agent_id_0".
        switch_frequency: Defaults to 1.0
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
        InformedPolicyConfig instance
    """
    actions = generate_action_list(action_space_type)

    return InformedPolicyConfig(
        action_sampler_args=dict(**kwargs, actions=actions),
        action_sampler_class=action_sampler_class,
        agent_id=agent_id,
        good_view_percentage=good_view_percentage,
        use_goal_state_driven_actions=use_goal_state_driven_actions,
        file_name=file_name,
        switch_frequency=switch_frequency,
    )


def make_naive_scan_policy_config(step_size: float, agent_id="agent_id_0"):
    """Simliar to InformedPolicyConfigGenerator, but for NaiveScanPolicyConfig.

    Currently less flexible than the other two classes above, because this is currently
    only used with one set of parameters

    Args:
        step_size: Fixed amount to move the agent
        agent_id: Agent name. Defaults to "agent_id_0".

    Returns:
        NaiveScanPolicyConfig instance
    """
    actions = generate_action_list(action_space_type="distant_agent_no_translation")

    return NaiveScanPolicyConfig(
        action_sampler_args=dict(actions=actions),
        action_sampler_class=ConstantSampler,
        agent_id=agent_id,
        switch_frequency=1,
        fixed_amount=step_size,
    )


def make_surface_policy_config(
    desired_object_distance: float,
    alpha: float,
    use_goal_state_driven_actions: bool = False,
    action_sampler_class: Type[ActionSampler] = ConstantSampler,
    action_space_type: str = "surface_agent",
    file_name: str = None,
    agent_id: str = "agent_id_0",
    **kwargs,
):
    """Similar to BasePolicyConfigGenerator, but for InformedPolicy class.

    Args:
        desired_object_distance: ?
        alpha: ?
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
        SurfacePolicyConfig instance
    """
    actions = generate_action_list(action_space_type)

    return SurfacePolicyConfig(
        action_sampler_args=dict(**kwargs, actions=actions),
        action_sampler_class=action_sampler_class,
        agent_id=agent_id,
        desired_object_distance=desired_object_distance,
        alpha=alpha,
        use_goal_state_driven_actions=use_goal_state_driven_actions,
        file_name=file_name,
    )


def make_curv_surface_policy_config(
    desired_object_distance,
    alpha,
    pc_alpha,
    max_pc_bias_steps,
    min_general_steps,
    min_heading_steps,
    use_goal_state_driven_actions=False,
    action_sampler_class: Type[ActionSampler] = ConstantSampler,
    action_space_type="surface_agent",
    file_name=None,
    agent_id="agent_id_0",
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
