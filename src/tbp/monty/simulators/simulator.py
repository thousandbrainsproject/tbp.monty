# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from typing import Dict, List, Optional, Protocol, Sequence

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.environments.embodied_environment import (
    QuaternionWXYZ,
    VectorXYZ,
)


class Simulator(Protocol):
    """A Protocol defining a simulator for use in simulated environments.

    A Simulator is responsible for a simulated environment that contains objects to
    interact with, agents to do the interacting, and for collecting observations and
    proprioceptive state to send to Monty.
    """

    # TODO - do we need a way to abstract the concept of "agent"?
    def initialize_agent(self, agent_id, agent_state):
        """Update agent runtime state."""
        ...

    def remove_all_objects(self):
        """Remove all objects from the simulated environment."""
        ...

    def add_object(
        self,
        name: str,
        position: VectorXYZ = (0.0, 0.0, 0.0),
        rotation: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0),
        scale: VectorXYZ = (1.0, 1.0, 1.0),
        semantic_id: Optional[str] = None,
        enable_physics: bool = False,
        object_to_avoid: bool = False,
        primary_target_bb: Optional[List] = None,
    ) -> None:
        """Add new object to simulated environment.

        Adds a new object based on the named object. This assumes that the set of
        available objects are preloaded and keyed by name.

        Args:
            name: Registered object name.
            position: Initial absolute position of the object.
            rotation: Initial orientation of the object.
            scale: Initial object scale.
            semantic_id: Optional override for the object's semantic ID.
            enable_physics: Whether to enable physics on the object.
            object_to_avoid: If True, ensure the object is not colliding with
              other objects.
            primary_target_bb: If not None, this is a list of the min and
              max corners of a bounding box for the primary object, used to prevent
              obscuring the primary object with the new object.
        """
        ...

    @property
    def num_objects(self) -> int:
        """Return the number of instantiated objects in the environment."""
        ...

    @property
    def action_space(self):
        """Returns the set of all available actions."""
        ...

    def get_agent(self, agent_id):
        """Return agent instance."""
        ...

    @property
    def observations(self):
        """Get sensor observations."""
        ...

    @property
    def states(self):
        """Get agent and sensor states."""
        ...

    def apply_actions(self, actions: Sequence[Action]) -> Dict[str, Dict]:
        """Execute the given actions in the environment.

        Args:
            actions: The actions to execute.

        Returns:
            A dictionary with the observations grouped by agent_id.

        Note:
            If the actions are an empty sequence, the current observations are returned.
        """
        ...

    def reset(self):
        """Reset the simulator."""
        ...

    def close(self):
        """Close any resources used by the simulator."""
        ...
