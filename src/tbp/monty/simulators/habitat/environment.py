# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import TYPE_CHECKING, Sequence

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.environments.environment import (
    ObjectID,
    QuaternionWXYZ,
    SemanticID,
    SimulatedObjectEnvironment,
    VectorXYZ,
)
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_system_state import ProprioceptiveState
from tbp.monty.frameworks.utils.dataclass_utils import (
    create_dataclass_args,
)
from tbp.monty.simulators.habitat import (
    HabitatAgent,
    HabitatSim,
    MultiSensorAgent,
    SingleSensorAgent,
)

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "AgentConfig",
    "HabitatEnvironment",
    "MultiSensorAgentArgs",
    "ObjectConfig",
    "SingleSensorAgentArgs",
]


# Create agent and object configuration helper dataclasses

# ObjectConfig dataclass based on the arguments of `HabitatSim.add_object` method
ObjectConfig = create_dataclass_args("ObjectConfig", HabitatSim.add_object)
ObjectConfig.__module__ = __name__


# FIXME: Using HabitatAgent constructor as base class will cause the `make_dataclass`
#        function to throw the following exception:
#        `TypeError: non-default argument 'sensor_id' follows default argument`
#        For now, we will just use plain empty class for HabitaAgentArgs
#
# HabitatAgentArgs = create_dataclass_args("HabitatAgentArgs", HabitatAgent.__init__)
class HabitatAgentArgs:
    pass


# SingleSensorAgentArgs dataclass based on constructor args
SingleSensorAgentArgs = create_dataclass_args(
    "SingleSensorAgentArgs", SingleSensorAgent.__init__, HabitatAgentArgs
)
SingleSensorAgentArgs.__module__ = __name__

# MultiSensorAgentArgs dataclass based on constructor args
MultiSensorAgentArgs = create_dataclass_args(
    "MultiSensorAgentArgs", MultiSensorAgent.__init__, HabitatAgentArgs
)
MultiSensorAgentArgs.__module__ = __name__


@dataclass
class AgentConfig:
    """Agent configuration used by :class:`HabitatEnvironment`."""

    agent_type: type[HabitatAgent]
    agent_args: dict | type[HabitatAgentArgs]


class HabitatEnvironment(SimulatedObjectEnvironment):
    """habitat-sim environment compatible with Monty.

    Attributes:
        agents: List of :class:`AgentConfig` to place in the scene.
        objects: Optional list of :class:`ObjectConfig` to place in the scene.
        scene_id: Scene to use or None for empty environment.
        seed: Simulator seed to use
        data_path: Path to the dataset.
    """

    def __init__(
        self,
        agents: dict | AgentConfig,
        objects: list[dict | ObjectConfig] | None = None,
        scene_id: str | None = None,
        seed: int = 42,
        data_path: str | Path | None = None,
    ):
        super().__init__()
        # TODO: Change the configuration to configure multiple agents
        agents = [agents]
        self._agents = []
        for config in agents:
            cfg_dict = asdict(config) if is_dataclass(config) else config
            agent_type = cfg_dict["agent_type"]
            args = cfg_dict["agent_args"]
            if is_dataclass(args):
                args = asdict(args)
            agent = agent_type(**args)
            self._agents.append(agent)

        self._env = HabitatSim(
            agents=self._agents,
            scene_id=scene_id,
            seed=seed,
            data_path=data_path,
        )

        if objects is not None:
            for obj in objects:
                obj_dict = asdict(obj) if is_dataclass(obj) else obj
                self._env.add_object(**obj_dict)

    def add_object(
        self,
        name: str,
        position: VectorXYZ = (0.0, 0.0, 0.0),
        rotation: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0),
        scale: VectorXYZ = (1.0, 1.0, 1.0),
        semantic_id: SemanticID | None = None,
        primary_target_object: ObjectID | None = None,
    ) -> ObjectID:
        return self._env.add_object(
            name,
            position,
            rotation,
            scale,
            semantic_id,
            primary_target_object,
        ).object_id

    def step(
        self, actions: Sequence[Action]
    ) -> tuple[Observations, ProprioceptiveState]:
        return self._env.step(actions)

    def remove_all_objects(self) -> None:
        return self._env.remove_all_objects()

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        return self._env.reset()

    def close(self) -> None:
        _env = getattr(self, "_env", None)
        if _env is not None:
            _env.close()
            self._env = None
