# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from pydantic import BeforeValidator, ConfigDict, validate_call
from typing_extensions import Annotated

from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.sensor_modules.sensor_module import (
    SensorModule,
    Transform,
)

__all__ = [
    "ExperimentSensorModule",
    "ExperimentTransform",
    "ResettableSensorModule",
]


@runtime_checkable
class ResettableSensorModule(Protocol):
    """A Sensor Module that can be reset."""

    def reset(self) -> None:
        """Reset the internal state of this Sensor Module."""
        ...


@runtime_checkable
class ExperimentTransform(Transform, Protocol):
    """An experimental transform with experiment affordances.

    Allows for resetting the internal state of the transform.
    """

    def reset(self) -> None:
        """Reset the internal state of this transform.

        The protocol offers a default implementation that does nothing.
        """
        pass


class ExperimentSensorModule(SensorModule, ResettableSensorModule):
    """An experimental wrapper around a Sensor Module with experiment affordances.

    Allows for resetting the internal state of the transforms in the transform pipeline.
    """

    _transforms: Sequence[ExperimentTransform]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        sensor_module_id: str,
        sensor_id: SensorID,
        transforms: Annotated[Sequence[ExperimentTransform], BeforeValidator(tuple)],
    ) -> None:
        super().__init__(sensor_module_id, sensor_id, transforms)

    def reset(self) -> None:
        """Reset the internal state of this Sensor Module."""
        for transform in self._transforms:
            transform.reset()
