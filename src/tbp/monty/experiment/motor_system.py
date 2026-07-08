# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Protocol

__all__ = [
    "ExperimentMotorPolicy",
    "ExperimentMotorPolicySelector",
    "ExperimentMotorSystem",
]


class ExperimentMotorSystem(Protocol):
    """Experiment interface to a Motor System."""

    def reset(self) -> None:
        """Reset the internal state of this Motor System."""
        ...

    @property
    def motor_only_step(self) -> bool:
        """When `True`, suppress Learning Module processing."""
        ...

    @motor_only_step.setter
    def motor_only_step(self, value: bool) -> None: ...


class ExperimentMotorPolicySelector(Protocol):
    """Experiment interface to a Motor Policy Selector."""

    def fixme_provide_motor_system(self, motor_system: ExperimentMotorSystem) -> None:
        """Provide access to the Motor System during initialization.

        This is part of the work to remove `reset()` in favor or Hydra instantiation.
        It is used to provide a reference to the Motor System so the `SurfacePolicy`
        and its subclasses can override the `motor_only_step` property.

        TODO: This whole mechanism is a hack for the benefit of `SurfacePolicy` et. al.
        What we should be doing is supporting more complex actions, like
        "follow surface in this direction," whose details are left to the simulator.

        Args:
            motor_system: The associated Motor System.
        """
        ...

    def reset(self) -> None:
        """Reset the internal state of this Motor Policy Selector."""
        ...


class ExperimentMotorPolicy(Protocol):
    """Experiment interface to a Motor Policy."""

    def fixme_provide_motor_system(self, motor_system: ExperimentMotorSystem) -> None:
        """Provide access to the Motor System during initialization.

        This is part of the work to remove `reset()` in favor or Hydra instantiation.
        It is used to provide a reference to the Motor System so the `SurfacePolicy`
        and its subclasses can override the `motor_only_step` property.

        TODO: This whole mechanism is a hack for the benefit of `SurfacePolicy` et. al.
        What we should be doing is supporting more complex actions, like
        "follow surface in this direction," whose details are left to the simulator.

        Args:
            motor_system: The associated Motor System.
        """
        ...

    def reset(self) -> None:
        """Reset the internal state of this Motor Policy."""
        ...
