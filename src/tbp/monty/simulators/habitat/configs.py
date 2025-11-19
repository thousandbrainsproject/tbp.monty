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
from typing import Callable, Mapping

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.config_utils.make_env_interface_configs import (
    FiveLMMountConfig,
    MultiLMMountConfig,
    PatchAndViewFinderMountConfig,
    PatchAndViewFinderMountLowResConfig,
    PatchAndViewFinderMultiObjectMountConfig,
    SurfaceAndViewFinderMountConfig,
    TwoCameraMountConfig,
    TwoLMStackedDistantMountConfig,
    TwoLMStackedSurfaceMountConfig,
    make_multi_sensor_mount_config,
)
from tbp.monty.frameworks.environment_utils.transforms import (
    AddNoiseToRawDepthImage,
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
    "EnvInitArgsMontyWorldPatchViewMount",
    "EnvInitArgsMontyWorldSurfaceViewMount",
    "EnvInitArgsMultiLMMount",
    "EnvInitArgsPatchViewFinderMultiObjectMount",
    "EnvInitArgsPatchViewMount",
    "EnvInitArgsPatchViewMountLowRes",
    "EnvInitArgsShapenetPatchViewMount",
    "EnvInitArgsSimpleMount",
    "EnvInitArgsSinglePTZ",
    "EnvInitArgsSurfaceViewMount",
    "EnvInitArgsTwoLMDistantStackedMount",
    "EnvInitArgsTwoLMSurfaceStackedMount",
    "FiveLMMountHabitatEnvInterfaceConfig",
    "MultiLMMountHabitatEnvInterfaceConfig",
    "NoisyPatchViewFinderMountHabitatEnvInterfaceConfig",
    "NoisySurfaceViewFinderMountHabitatEnvInterfaceConfig",
    "ObjectConfig",
    "PatchViewFinderLowResMountHabitatEnvInterfaceConfig",
    "PatchViewFinderMontyWorldMountHabitatEnvInterfaceConfig",
    "PatchViewFinderMountHabitatEnvInterfaceConfig",
    "PatchViewFinderMultiObjectMountHabitatEnvInterfaceConfig",
    "PatchViewFinderShapenetMountHabitatEnvInterfaceConfig",
    "SimpleMountHabitatEnvInterfaceConfig",
    "SinglePTZHabitatEnvInterfaceConfig",
    "SurfaceViewFinderMontyWorldMountHabitatEnvInterfaceConfig",
    "SurfaceViewFinderMountHabitatEnvInterfaceConfig",
    "TwoLMStackedDistantMountHabitatEnvInterfaceConfig",
    "TwoLMStackedSurfaceMountHabitatEnvInterfaceConfig",
    "make_multi_sensor_habitat_env_interface_config",
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
class EnvInitArgsSimpleMount(EnvInitArgs):
    """Use this to make a sim with a cone and a single mount agent with two cameras."""

    agents: list[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, TwoCameraMountConfig().__dict__)
        ]
    )


@dataclass
class EnvInitArgsPatchViewMount(EnvInitArgs):
    agents: list[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, PatchAndViewFinderMountConfig().__dict__)
        ]
    )


@dataclass
class EnvInitArgsSurfaceViewMount(EnvInitArgs):
    agents: list[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, SurfaceAndViewFinderMountConfig().__dict__)
        ]
    )


@dataclass
class EnvInitArgsMontyWorldPatchViewMount(EnvInitArgsPatchViewMount):
    data_path: str = os.path.join(os.environ["MONTY_DATA"], "numenta_lab")


@dataclass
class EnvInitArgsMontyWorldSurfaceViewMount(EnvInitArgsMontyWorldPatchViewMount):
    agents: list[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, SurfaceAndViewFinderMountConfig().__dict__)
        ]
    )


@dataclass
class EnvInitArgsPatchViewMountLowRes(EnvInitArgs):
    agents: list[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(
                MultiSensorAgent, PatchAndViewFinderMountLowResConfig().__dict__
            )
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
class SimpleMountHabitatEnvInterfaceConfig:
    """Define single cone, two camera single mount agent environment interface config.

    Use this to make a :class:`EnvironmentInterface` with an env with a single cone and
    a single mount agent with two cameras.
    """

    env_init_func: Callable = field(default=HabitatEnvironment)
    env_init_args: dict = field(
        default_factory=lambda: EnvInitArgsSimpleMount().__dict__
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
class NoisyPatchViewFinderMountHabitatEnvInterfaceConfig:
    env_init_func: Callable = field(default=HabitatEnvironment)
    env_init_args: dict = field(
        default_factory=lambda: EnvInitArgsPatchViewMount().__dict__
    )
    transform: Callable | list | None = None

    def __post_init__(self):
        agent_args = self.env_init_args["agents"][0].agent_args
        self.transform = [
            MissingToMaxDepth(agent_id=AgentID(agent_args["agent_id"]), max_depth=1),
            AddNoiseToRawDepthImage(
                agent_id=AgentID(agent_args["agent_id"]),
                sigma=0.001,
            ),  # add gaussian noise with 0.001 std to depth image
            # Uncomment line below to enable smoothing of sensor patch depth
            # GaussianSmoothing(agent_id=AgentID(agent_args["agent_id"])),
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
class EnvInitArgsShapenetPatchViewMount(EnvInitArgsPatchViewMount):
    data_path: str = os.path.join(os.environ["MONTY_DATA"], "shapenet")


@dataclass
class PatchViewFinderLowResMountHabitatEnvInterfaceConfig(
    PatchViewFinderMountHabitatEnvInterfaceConfig
):
    env_init_args: dict = field(
        default_factory=lambda: EnvInitArgsPatchViewMountLowRes().__dict__
    )


@dataclass
class PatchViewFinderShapenetMountHabitatEnvInterfaceConfig(
    PatchViewFinderMountHabitatEnvInterfaceConfig
):
    env_init_args: dict = field(
        default_factory=lambda: EnvInitArgsShapenetPatchViewMount().__dict__
    )


@dataclass
class PatchViewFinderMontyWorldMountHabitatEnvInterfaceConfig(
    PatchViewFinderMountHabitatEnvInterfaceConfig
):
    env_init_args: dict = field(
        default_factory=lambda: EnvInitArgsMontyWorldPatchViewMount().__dict__
    )


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
                depth_clip_sensors=(0,),  # comma needed to make it a tuple
                clip_value=0.05,
            ),
        ]


@dataclass
class SurfaceViewFinderMontyWorldMountHabitatEnvInterfaceConfig(
    SurfaceViewFinderMountHabitatEnvInterfaceConfig
):
    env_init_args: dict = field(
        default_factory=lambda: EnvInitArgsMontyWorldSurfaceViewMount().__dict__
    )


@dataclass
class NoisySurfaceViewFinderMountHabitatEnvInterfaceConfig(
    PatchViewFinderMountHabitatEnvInterfaceConfig
):
    env_init_args: dict = field(
        default_factory=lambda: EnvInitArgsSurfaceViewMount().__dict__
    )

    def __post_init__(self):
        agent_args = self.env_init_args["agents"][0].agent_args
        self.transform = [
            MissingToMaxDepth(agent_id=AgentID(agent_args["agent_id"]), max_depth=1),
            AddNoiseToRawDepthImage(
                agent_id=AgentID(agent_args["agent_id"]),
                sigma=0.001,
            ),  # add gaussian noise with 0.001 std to depth image
            # Uncomment line below to enable smoothing of sensor patch depth
            # GaussianSmoothing(agent_id=AgentID(agent_args["agent_id"])),
            DepthTo3DLocations(
                agent_id=AgentID(agent_args["agent_id"]),
                sensor_ids=agent_args["sensor_ids"],
                resolutions=agent_args["resolutions"],
                world_coord=True,
                zooms=agent_args["zooms"],
                get_all_points=True,
                use_semantic_sensor=False,
                depth_clip_sensors=(0,),  # comma needed to make it a tuple
                clip_value=0.05,
            ),
        ]


@dataclass
class EnvInitArgsMultiLMMount(EnvInitArgs):
    agents: list[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, MultiLMMountConfig().__dict__)
        ]
    )


@dataclass
class MultiLMMountHabitatEnvInterfaceConfig:
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
class EnvInitArgsTwoLMSurfaceStackedMount(EnvInitArgs):
    agents: list[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, TwoLMStackedSurfaceMountConfig().__dict__)
        ]
    )


@dataclass
class TwoLMStackedSurfaceMountHabitatEnvInterfaceConfig(
    MultiLMMountHabitatEnvInterfaceConfig
):
    env_init_args: dict = field(
        default_factory=lambda: EnvInitArgsTwoLMSurfaceStackedMount().__dict__
    )


@dataclass
class EnvInitArgsFiveLMMount(EnvInitArgs):
    agents: list[AgentConfig] = field(
        default_factory=lambda: [
            AgentConfig(MultiSensorAgent, FiveLMMountConfig().__dict__)
        ]
    )


@dataclass
class FiveLMMountHabitatEnvInterfaceConfig(MultiLMMountHabitatEnvInterfaceConfig):
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


def make_multi_sensor_habitat_env_interface_config(
    n_sensors: int,
    **mount_kwargs: Mapping,
) -> MultiLMMountHabitatEnvInterfaceConfig:
    """Generate environment interface configs for a multi-LM experiment config.

    This function is useful for creating habitat environment interface configs for
    multi-LM experiments. The default arguments will place sensors on a grid, with
    sensors spreading out from the center and with 1 cm spacing between sensors,
    64 x 64 resolution, and 10x zoom (except for the view finder which has a zoom of
    1.0). See `make_multi_sensor_mount_config` and `make_sensor_positions_on_grid` for
    more details.

    Any keyword arguments are passed to `make_multi_sensor_mount_config`. You can, for
    example, build non-default sensor positions (perhaps using
    `make_sensor_positions_on_grid`) and supply them to this function. All other
    attributes will be generated according to default behavior.

    Args:
        n_sensors: Number of sensors, not including the view finder.
        **mount_kwargs: Arguments forwarded to `make_multi_sensor_mount_config`. See
            `make_multi_sensor_mount_config` for details.

    Returns:
        Config ready for use in an experiment config.
    """
    mount_config = make_multi_sensor_mount_config(n_sensors, **mount_kwargs)

    env_init_args = EnvInitArgsMultiLMMount()
    env_init_args.agents = [AgentConfig(MultiSensorAgent, mount_config)]
    env_init_args = env_init_args.__dict__
    return MultiLMMountHabitatEnvInterfaceConfig(env_init_args=env_init_args)
