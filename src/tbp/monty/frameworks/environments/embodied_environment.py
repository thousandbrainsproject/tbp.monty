# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import abc
import collections.abc
from typing import Any, Dict

from tbp.monty.frameworks.actions.actions import Action

__all__ = ["EmbodiedEnvironment", "ActionSpace"]


class ActionSpace(collections.abc.Container):
    """Represents the environment action space."""

    @abc.abstractmethod
    def sample(self):
        """Sample the action space returning a random action."""
        pass


class EmbodiedEnvironment(abc.ABC):
    @property
    @abc.abstractmethod
    def action_space(self):
        """Returns list of all possible actions available in the environment."""
        pass

    @abc.abstractmethod
    def step(self, action: Action) -> Dict[Any, Dict]:
        """Apply the given action to the environment.

        Return the current observations and other environment information (i.e. sensor
        pose) after the action is applied.
        """
        pass

    @abc.abstractmethod
    def get_state(self):
        """Return the state of the environment (and agent)."""
        pass

    @abc.abstractmethod
    def reset(self):
        """Reset enviroment to its initial state.

        Return the environment's initial observations.
        """
        pass

    @abc.abstractmethod
    def close(self):
        """Close the environmnt releasing all resources.

        Any call to any other environment method may raise an exception
        """
        pass

    def __del__(self):
        self.close()
