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

from dataclasses import dataclass
from typing import NewType, Protocol, Sequence, Tuple

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_system_state import ProprioceptiveState

__all__ = [
    "Environment",
    "ObjectEnvironment",
    "ObjectID",
    "ObjectInfo",
    "QuaternionWXYZ",
    "ResettableEnvironment",
    "SemanticID",
    "SimulatedEnvironment",
    "SimulatedObjectEnvironment",
    "VectorXYZ",
]

ObjectID = NewType("ObjectID", int)
"""Unique identifier for an object in the environment."""

SemanticID = NewType("SemanticID", int)
"""Unique identifier for an object's semantic class."""

VectorXYZ = Tuple[float, float, float]
QuaternionWXYZ = Tuple[float, float, float, float]


@dataclass
class ObjectInfo:
    """Contains the identifying information of an object created in the environment."""

    object_id: ObjectID
    semantic_id: SemanticID | None


class Environment(Protocol):
    """Base protocol for all environments that support steppable actions."""

    def step(
        self, actions: Sequence[Action]
    ) -> tuple[Observations, ProprioceptiveState]:
        """Apply the given actions to the environment.

        Args:
            actions: The actions to apply to the environment.

        Returns:
            The current observations and proprioceptive state.

        Note:
            If the actions are an empty sequence, the current observations are returned.
        """
        ...

    def close(self) -> None:
        """Close the environment and release all resources.

        Any call to any other environment method may raise an exception.
        """
        ...


class ObjectEnvironment(Protocol):
    """Protocol for environments that support adding and removing objects."""

    def add_object(
        self,
        name: str,
        position: VectorXYZ = (0.0, 0.0, 0.0),
        rotation: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0),
        scale: VectorXYZ = (1.0, 1.0, 1.0),
        semantic_id: SemanticID | None = None,
        primary_target_object: ObjectID | None = None,
    ) -> ObjectID:
        """Add an object to the environment.

        Args:
            name: The name of the object to add.
            position: The initial absolute position of the object.
            rotation: The initial rotation WXYZ quaternion of the object. Defaults to
                (1,0,0,0).
            scale: The scale of the object to add. Defaults to (1,1,1).
            semantic_id: Optional override for the object semantic ID. Defaults to None.
            primary_target_object: The ID of the primary target object. If not None, the
                added object will be positioned so that it does not obscure the initial
                view of the primary target object (which avoiding collision alone cannot
                guarantee). Used when adding multiple objects. Defaults to None.

        Returns:
            The ID of the added object.
        """
        ...

    def remove_all_objects(self) -> None:
        """Remove all objects from the environment.

        TODO: This remove_all_objects interface is elevated from
              HabitatSim.remove_all_objects and is quite specific to HabitatSim
              implementation. We should consider refactoring this to be more generic.
        """
        ...


class ResettableEnvironment(Protocol):
    """Protocol for environments that can be reset to their initial state."""

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        """Reset the environment to its initial state.

        Returns:
            The environment's initial observations and proprioceptive state.
        """
        ...


class SimulatedEnvironment(Environment, ResettableEnvironment, Protocol):
    pass


class SimulatedObjectEnvironment(
    Environment, ObjectEnvironment, ResettableEnvironment, Protocol
):
    pass
