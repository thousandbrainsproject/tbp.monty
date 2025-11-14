# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.config_utils.make_env_interface_configs import (
    FiveLMMountConfig,
    MultiLMMountConfig,
    PatchAndViewFinderMountConfig,
    PatchAndViewFinderMultiObjectMountConfig,
    SurfaceAndViewFinderMountConfig,
    TwoLMStackedDistantMountConfig,
)
from tbp.monty.frameworks.environment_utils.transforms import (
    DepthTo3DLocations,
    MissingToMaxDepth,
)
from tbp.monty.simulators.habitat import MultiSensorAgent, SingleSensorAgent
from tbp.monty.simulators.habitat.environment import (
    AgentConfig,
    HabitatEnvironment,
    ObjectConfig,
)

__all__ = [
    "EnvInitArgs",
    "EnvInitArgsFiveLMMount",
    "EnvInitArgsMultiLMMount",
    "EnvInitArgsPatchViewFinderMultiObjectMount",
    "EnvInitArgsPatchViewMount",
    "EnvInitArgsSinglePTZ",
    "EnvInitArgsSurfaceViewMount",
    "EnvInitArgsTwoLMDistantStackedMount",
    "FiveLMMountHabitatEnvInterfaceConfig",
    "MultiLMMountHabitatEnvInterfaceConfig",
    "ObjectConfig",
    "PatchViewFinderMountHabitatEnvInterfaceConfig",
    "PatchViewFinderMultiObjectMountHabitatEnvInterfaceConfig",
    "SinglePTZHabitatEnvInterfaceConfig",
    "SurfaceViewFinderMountHabitatEnvInterfaceConfig",
    "TwoLMStackedDistantMountHabitatEnvInterfaceConfig",
]


@dataclass
class EnvInitArgs:
    """Args for :class:`HabitatEnvironment`."""

    agents: list[AgentConfig]
    objects: list[ObjectConfig] = field(
        default_factory=lambda: [ObjectConfig("coneSolid", position=(0.0, 1.5, -0.1))]
    )
    scene_id: int | None = field(default=None)
    seed: int = field(default=42)
    data_path: str = os.path.join(os.environ["MONTY_DATA"], "habitat/objects/ycb")


@dataclass
class EnvInitArgsSinglePTZ(EnvInitArgs):
    """Use this to make a sim with a cone and a single PTZCameraAgent."""

    agents: list[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(
                SingleSensorAgent,
                dict(
                    agent_id=AgentID("agent_id_0"),
                    sensor_id="sensor_id_0",
                    resolution=(320, 240),
                ),
            )
        ]
    )


@dataclass
class EnvInitArgsPatchViewMount(EnvInitArgs):
    # conf/experiment/config/environment/init_args/patch_view_mount.yaml
    agents: list[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, PatchAndViewFinderMountConfig().__dict__)
        ]
    )


@dataclass
class EnvInitArgsSurfaceViewMount(EnvInitArgs):
    # conf/experiment/config/environment/init_args/surface_view_mount.yaml
    agents: list[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, SurfaceAndViewFinderMountConfig().__dict__)
        ]
    )


@dataclass
class SinglePTZHabitatEnvInterfaceConfig:
    """Define environment interface config with a single cone & single PTZCameraAgent.

    Use this to make a :class:`EnvironmentInterface` with an env with a single cone and
    a single PTZCameraAgent.
    """

    env_init_func: Callable = field(default=HabitatEnvironment)
    env_init_args: dict | dataclass = field(
        default_factory=lambda: EnvInitArgsSinglePTZ().__dict__
    )
    transform: Callable | list | None = field(default=None)


@dataclass
class PatchViewFinderMountHabitatEnvInterfaceConfig:
    env_init_func: Callable = field(default=HabitatEnvironment)
    env_init_args: dict = field(
        default_factory=lambda: EnvInitArgsPatchViewMount().__dict__
    )
    transform: Callable | list | None = None
    rng: Callable | None = None

    def __post_init__(self):
        agent_args = self.env_init_args["agents"][0].agent_args
        self.transform = [
            MissingToMaxDepth(agent_id=AgentID(agent_args["agent_id"]), max_depth=1),
            DepthTo3DLocations(
                agent_id=AgentID(agent_args["agent_id"]),
                sensor_ids=agent_args["sensor_ids"],
                resolutions=agent_args["resolutions"],
                world_coord=True,
                zooms=agent_args["zooms"],
                get_all_points=True,
                use_semantic_sensor=False,
            ),
        ]


@dataclass
class SurfaceViewFinderMountHabitatEnvInterfaceConfig(
    PatchViewFinderMountHabitatEnvInterfaceConfig
):
    env_init_args: dict = field(
        default_factory=lambda: EnvInitArgsSurfaceViewMount().__dict__
    )

    def __post_init__(self):
        agent_args = self.env_init_args["agents"][0].agent_args
        self.transform = [
            MissingToMaxDepth(agent_id=AgentID(agent_args["agent_id"]), max_depth=1),
            DepthTo3DLocations(
                agent_id=AgentID(agent_args["agent_id"]),
                sensor_ids=agent_args["sensor_ids"],
                resolutions=agent_args["resolutions"],
                world_coord=True,
                zooms=agent_args["zooms"],
                get_all_points=True,
                use_semantic_sensor=False,
                depth_clip_sensors=[0],
                clip_value=0.05,
            ),
        ]


@dataclass
class EnvInitArgsMultiLMMount(EnvInitArgs):
    # conf/experiment/config/environment/init_args/multi_lm_mount.yaml
    agents: list[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, MultiLMMountConfig().__dict__)
        ]
    )


@dataclass
class MultiLMMountHabitatEnvInterfaceConfig:
    # conf/experiment/config/environment/multi_lm_mount_habitat.yaml
    env_init_func: Callable = field(default=HabitatEnvironment)
    env_init_args: dict = field(
        default_factory=lambda: EnvInitArgsMultiLMMount().__dict__
    )
    transform: Callable | list | None = None

    def __post_init__(self):
        agent_args = self.env_init_args["agents"][0].agent_args
        self.transform = [
            MissingToMaxDepth(agent_id=AgentID(agent_args["agent_id"]), max_depth=1),
            DepthTo3DLocations(
                agent_id=AgentID(agent_args["agent_id"]),
                sensor_ids=agent_args["sensor_ids"],
                resolutions=agent_args["resolutions"],
                world_coord=True,
                zooms=agent_args["zooms"],
                get_all_points=True,
                use_semantic_sensor=False,
            ),
        ]


@dataclass
class EnvInitArgsTwoLMDistantStackedMount(EnvInitArgs):
    agents: list[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, TwoLMStackedDistantMountConfig().__dict__)
        ]
    )


@dataclass
class TwoLMStackedDistantMountHabitatEnvInterfaceConfig(
    MultiLMMountHabitatEnvInterfaceConfig
):
    env_init_args: dict = field(
        default_factory=lambda: EnvInitArgsTwoLMDistantStackedMount().__dict__
    )


@dataclass
class EnvInitArgsFiveLMMount(EnvInitArgs):
    # conf/experiment/config/environment/init_args/five_lm_mount.yaml
    agents: list[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, FiveLMMountConfig().__dict__)
        ]
    )


@dataclass
class FiveLMMountHabitatEnvInterfaceConfig(MultiLMMountHabitatEnvInterfaceConfig):
    # conf/experiment/config/environment/five_lm_mount_habitat.yaml
    env_init_args: dict = field(
        default_factory=lambda: EnvInitArgsFiveLMMount().__dict__
    )


@dataclass
class EnvInitArgsPatchViewFinderMultiObjectMount(EnvInitArgs):
    agents: list[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(
                MultiSensorAgent, PatchAndViewFinderMultiObjectMountConfig().__dict__
            )
        ]
    )


@dataclass
class PatchViewFinderMultiObjectMountHabitatEnvInterfaceConfig:
    env_init_func: Callable = field(default=HabitatEnvironment)
    env_init_args: dict = field(
        default_factory=lambda: EnvInitArgsPatchViewFinderMultiObjectMount().__dict__
    )
    transform: Callable | list | None = None
    rng: Callable | None = None

    def __post_init__(self):
        agent_args = self.env_init_args["agents"][0].agent_args
        self.transform = [
            MissingToMaxDepth(agent_id=AgentID(agent_args["agent_id"]), max_depth=1),
            DepthTo3DLocations(
                agent_id=AgentID(agent_args["agent_id"]),
                sensor_ids=agent_args["sensor_ids"],
                resolutions=agent_args["resolutions"],
                world_coord=True,
                zooms=agent_args["zooms"],
                get_all_points=True,
                use_semantic_sensor=True,
            ),
        ]
