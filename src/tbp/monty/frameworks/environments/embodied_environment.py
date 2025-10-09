# Copyright 2025 Thousand Brains Project
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
from typing import Any, Dict, Optional, Sequence, Tuple

from tbp.monty.frameworks.actions.actions import Action

__all__ = ["EmbodiedEnvironment", "ActionSpace", "VectorXYZ", "QuaternionWXYZ"]

VectorXYZ = Tuple[float, float, float]
QuaternionWXYZ = Tuple[float, float, float, float]


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
    def add_object(
        self,
        name: str,
        position: VectorXYZ = (0.0, 0.0, 0.0),
        rotation: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0),
        scale: VectorXYZ = (1.0, 1.0, 1.0),
        semantic_id: Optional[str] = None,
        enable_physics: Optional[bool] = False,
        object_to_avoid=False,
        primary_target_object=None,
    ):
        """Add an object to the environment.

        Args:
            name: The name of the object to add.
            position: The initial absolute position of the object.
            rotation: The initial rotation WXYZ quaternion of the object. Defaults to
                (1,0,0,0).
            scale: The scale of the object to add. Defaults to (1,1,1).
            semantic_id: Optional override for the object semantic ID.
            enable_physics: Whether to enable physics on the object. Defaults to False.
            object_to_avoid: If True, run collision checks to ensure the object will not
                collide with any other objects in the scene. If collision is detected,
                the object will be moved. Defaults to False.
            primary_target_object: If not None, the added object will be positioned so
                that it does not obscure the initial view of the primary target object
                (which avoiding collision alone cannot guarantee). Used when adding
                multiple objects. Defaults to None.

        Returns:
            The newly added object.

        TODO: This add_object interface is elevated from HabitatSim.add_object and is
              quite specific to HabitatSim implementation. We should consider
              refactoring this to be more generic.
        """
        pass

    @abc.abstractmethod
    def step(self, actions: Sequence[Action]) -> Dict[Any, Dict]:
        """Apply the given actions to the environment.

        Args:
            actions: The actions to apply to the environment.

        Returns:
            The current observations and other environment information (i.e. sensor
            pose) after the actions are applied.

        Note:
            If the actions are an empty sequence, the current observations are returned.
        """
        pass

    @abc.abstractmethod
    def get_state(self):
        """Return the state of the environment (and agent)."""
        pass

    @abc.abstractmethod
    def remove_all_objects(self):
        """Remove all objects from the environment.

        TODO: This remove_all_objects interface is elevated from
              HabitatSim.remove_all_objects and is quite specific to HabitatSim
              implementation. We should consider refactoring this to be more generic.
        """
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
