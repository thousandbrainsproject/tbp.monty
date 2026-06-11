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
    "ExperimentSensorModule",
]


class ExperimentSensorModule(Protocol):
    """Experiment interface to a Sensor Module."""

    def reset(self) -> None:
        """Reset the internal state of this Sensor Module."""
        pass
        ...
