# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import abc
import collections.abc
from typing import NewType, Tuple

from typing_extensions import deprecated

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_system_state import ProprioceptiveState

__all__ = [
    "ActionSpace",
    "EmbodiedEnvironment",
    "EulerAnglesXYZ",
    "ObjectID",
    "QuaternionWXYZ",
    "QuaternionXYZW",
    "SemanticID",
    "VectorXYZ",
]

ObjectID = NewType("ObjectID", int)
"""Unique identifier for an object in the environment."""
SemanticID = NewType("SemanticID", int)
"""Unique identifier for an object's semantic class."""
EulerAnglesXYZ = NewType("EulerAnglesXYZ", Tuple[float, float, float])
VectorXYZ = NewType("VectorXYZ", Tuple[float, float, float])
QuaternionWXYZ = NewType("QuaternionWXYZ", Tuple[float, float, float, float])
QuaternionXYZW = NewType("QuaternionXYZW", Tuple[float, float, float, float])

@deprecated("Use `ActionSampler` instead.")
class ActionSpace(collections.abc.Container):
    """Represents the environment action space."""

    @abc.abstractmethod
    def sample(self):
        """Sample the action space returning a random action."""
        pass


class EmbodiedEnvironment(abc.ABC):
    @abc.abstractmethod
    def add_object(
        self,
        name: str,
        position: VectorXYZ | None = None,
        rotation: QuaternionWXYZ | None = None,
        scale: VectorXYZ | None = None,
        semantic_id: SemanticID | None = None,
        primary_target_object: ObjectID | None = None,
    ) -> ObjectID:
        """Add an object to the environment.

        Args:
            name: The name of the object to add.
            position: The initial absolute position of the object. Defaults to None.
            rotation: The initial rotation WXYZ quaternion of the object. Defaults to
                None.
            scale: The scale of the object to add. Defaults to None.
            semantic_id: Optional override for the object semantic ID. Defaults to None.
            primary_target_object: The ID of the primary target object. If not None, the
                added object will be positioned so that it does not obscure the initial
                view of the primary target object (which avoiding collision alone cannot
                guarantee). Used when adding multiple objects. Defaults to None.

        Returns:
            The ID of the added object.
        """
        pass

    @abc.abstractmethod
    def step(self, action: Action) -> tuple[Observations, ProprioceptiveState]:
        """Apply the given action to the environment.

        Returns:
            The current observations and proprioceptive state.
        """
        pass

    @abc.abstractmethod
    def remove_all_objects(self) -> None:
        """Remove all objects from the environment.

        TODO: This remove_all_objects interface is elevated from
              HabitatSim.remove_all_objects and is quite specific to HabitatSim
              implementation. We should consider refactoring this to be more generic.
        """
        pass

    @abc.abstractmethod
    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        """Reset enviroment to its initial state.

        Returns:
            The environment's initial observations and proprioceptive state.
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
