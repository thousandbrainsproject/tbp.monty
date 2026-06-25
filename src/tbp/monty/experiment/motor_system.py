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

    def reset(self, motor_system: ExperimentMotorSystem) -> None:
        """Reset the internal state of this Motor Policy Selector.

        Args:
            motor_system: The associated Motor System.
        """
        ...


class ExperimentMotorPolicy(Protocol):
    """Experiment interface to a Motor Policy."""

    def reset(self, motor_system: ExperimentMotorSystem) -> None:
        """Reset the internal state of this Motor Policy.

        Args:
            motor_system: The associated Motor System.
        """
        ...
