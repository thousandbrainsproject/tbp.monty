# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from typing import Protocol, Sequence

from typing_extensions import Self

from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.sensor_modules import SensorModule
from tbp.monty.sensor_modules.sensor_module import RuntimeSensorModule, Transform

__all__ = [
    "ExperimentSensorModule",
]


class ExperimentSensorModule(RuntimeSensorModule, Protocol):
    """Experiment interface to a Sensor Module."""

    def reset(self) -> None:
        """Reset the internal state of this Sensor Module."""
        ...


class ExperimentTransform(Transform, Protocol):
    """An experimental transform with experiment affordances.

    Allows for resetting the internal state of the transform.
    """

    def reset(self: Self) -> None:
        """Reset the internal state of this transform.

        The protocol offers a default implementation that does nothing.
        """
        pass


class ExperimentalSensorModule(SensorModule, ExperimentSensorModule):
    """An experimental wrapper around a Sensor Module with experiment affordances.

    Allows for resetting the internal state of the transforms in the transform pipeline.
    """

    _transforms: Sequence[ExperimentTransform]

    def __init__(
        self: Self,
        sensor_module_id: str,
        sensor_id: SensorID,
        transforms: Sequence[ExperimentTransform],
    ) -> None:
        super().__init__(sensor_module_id, sensor_id, transforms)

    def reset(self: Self) -> None:
        """Reset the internal state of this Sensor Module."""
        for transform in self._transforms:
            transform.reset()
