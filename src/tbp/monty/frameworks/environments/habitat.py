# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from dataclasses import asdict, dataclass, is_dataclass
from typing import Dict, List, Optional, Type, Union

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.environments.embodied_environment import (
    ActionSpace,
    EmbodiedEnvironment,
)
from tbp.monty.frameworks.utils.dataclass_utils import create_dataclass_args
from tbp.monty.simulators.habitat import (
    HabitatAgent,
    HabitatSim,
    MultiSensorAgent,
    SingleSensorAgent,
)

__all__ = [
    "AgentConfig",
    "HabitatActionSpace",
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

    agent_type: Type[HabitatAgent]
    agent_args: Union[dict, Type[HabitatAgentArgs]]


class HabitatActionSpace(tuple, ActionSpace):
    """`ActionSpace` wrapper for Habitat's `AgentConfiguration`.

    Wraps :class:`habitat_sim.agent.AgentConfiguration` action space as monty
    :class:`.ActionSpace`.
    """

    def sample(self):
        return self.rng.choice(self)


class HabitatEnvironment(EmbodiedEnvironment):
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
        agents: List[Union[dict, AgentConfig]],
        objects: Optional[List[Union[dict, ObjectConfig]]] = None,
        scene_id: Optional[str] = None,
        seed: int = 42,
        data_path: Optional[str] = None,
    ):
        super().__init__()
        self._agents = []
        for config in agents:
            if is_dataclass(config):
                config = asdict(config)
            agent_type = config["agent_type"]
            args = config["agent_args"]
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
                if is_dataclass(obj):
                    obj = asdict(obj)
                self._env.add_object(**obj)

    @property
    def action_space(self):
        return HabitatActionSpace(self._env.get_action_space())

    def step(self, action: Action) -> Dict[str, Dict]:
        return self._env.apply_action(action)

    def reset(self):
        return self._env.reset()

    def close(self):
        _env = getattr(self, "_env", None)
        if _env is not None:
            _env.close()
            self._env = None

    def get_state(self):
        return self._env.get_states()
