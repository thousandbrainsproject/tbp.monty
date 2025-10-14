# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import gym

from tbp.monty.frameworks.environments.embodied_environment import (
    ActionSpace,
    EmbodiedEnvironment,
)

__all__ = ["RealRobotsEnvironment"]


class GymActionSpace(ActionSpace):
    """Wraps gym space as Monty action space."""

    def __init__(self, action_space):
        assert isinstance(action_space, gym.Space)
        self.action_space = action_space

    def __contains__(self, item):
        return self.action_space.contains(item)

    def sample(self):
        return self.action_space.sample()


class RealRobotsEnvironment(EmbodiedEnvironment):
    """Real Robots environment compatible with Monty.

    Note:
        `real_robots` dependencies are not installed by default.
        Install `real_robot` extra dependencies if you want to use `real_robot`
        environment
    """

    def __init__(self, id):  # noqa: A002
        super().__init__()
        self._env = gym.make(id=id)
        self._env.reset()

    @property
    def action_space(self):
        return GymActionSpace(self._env.action_space)

    def add_object(self, *args, **kwargs):
        # TODO The NotImplementedError highlights an issue with the EmbodiedEnvironment
        #      interface and how the class hierarchy is defined and used.
        raise NotImplementedError(
            "RealRobotsEnvironment does not support adding objects"
        )

    def step(self, actions):
        observation, reward, done, info = self._env.step(actions)
        return dict(**observation, reward=reward, done=done, info=info)

    def remove_all_objects(self):
        # TODO The NotImplementedError highlights an issue with the EmbodiedEnvironment
        #      interface and how the class hierarchy is defined and used.
        raise NotImplementedError(
            "RealRobotsEnvironment does not support removing all objects"
        )

    def reset(self):
        observation = self._env.reset()
        return dict(**observation, reward=0, done=False, info=None)

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
