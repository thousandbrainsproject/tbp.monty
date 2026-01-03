# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Protocol

from tbp.monty.frameworks.environments.embodied_environment import (
    ObjectID,
    ObjectInfo,
    QuaternionWXYZ,
    SemanticID,
    VectorXYZ,
)
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_system_state import ProprioceptiveState


class ExperimentSimulator(Protocol):
    """A Protocol defining how experiments can interact with simulated environments.

    An ExperimentSimulator is responsible for adding/removing objects to/from a
    simulated environment and resetting object states.
    """

    def remove_all_objects(self) -> None:
        """Remove all objects from the simulated environment."""
        ...

    def add_object(
        self,
        name: str,
        position: VectorXYZ = (0.0, 0.0, 0.0),
        rotation: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0),
        scale: VectorXYZ = (1.0, 1.0, 1.0),
        semantic_id: SemanticID | None = None,
        primary_target_object: ObjectID | None = None,
    ) -> ObjectInfo:
        """Add a new object to simulated environment.

        Adds a new object based on the named object. This assumes that the set of
        available objects are preloaded and keyed by name.

        Args:
            name: Registered object name.
            position: Initial absolute position of the object.
            rotation: Initial orientation of the object.
            scale: Initial object scale.
            semantic_id: Optional override for the object's semantic ID.
            primary_target_object: ID of the primary target object. If not None, the
                added object will be positioned so that it does not obscure the initial
                view of the primary target object (which avoiding collision alone cannot
                guarantee). Used when adding multiple objects. Defaults to None.

        Returns:
            The added object's information.
        """
        ...

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        """Reset the simulator.

        Returns:
            The initial observations from the simulator and proprioceptive state.
        """
        ...
