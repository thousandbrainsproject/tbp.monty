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

import uuid
from dataclasses import dataclass, field
from typing import Tuple

import quaternion as qt
from habitat_sim.sensor import CameraSensorSpec, SensorSpec, SensorType

from tbp.monty.frameworks.models.abstract_monty_classes import SensorObservations

__all__ = [
    "RGBDSensorConfig",
    "SemanticSensorConfig",
    "SensorConfig",
]

Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]
Size = Tuple[int, int]


@dataclass(frozen=True)
class SensorConfig:
    """Sensor configuration.

    Every Habitat sensor will inherit from this class.

    Attributes:
        sensor_id: Optional sensor ID unique within the sensor module.
            If given, then observations made by this sensor will be
            prefixed by this ID, i.e. "`sensor_id`.data"
        position: Sensor position relative to :class:`HabitatAgent`.
            Defaults to (0, 0, 0).
        rotation: Sensor rotation quaternion. Defaults to (1, 0, 0, 0).
    """

    sensor_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    position: Vector3 = (0.0, 0.0, 0.0)
    rotation: Quaternion = (1.0, 0.0, 0.0, 0.0)

    def get_specs(self) -> list[SensorSpec]:
        """Return a list of Habitat sensor specs to pass to `habitat-sim`."""
        return []

    def process_observations(
        self, sensor_obs: SensorObservations
    ) -> SensorObservations:
        """Callback used to process Habitat raw sensor observations.

        Args:
            sensor_obs: Raw habitat-sim observations from this sensor

        Returns:
            The processed observations grouped by agent_id
        """
        return sensor_obs


@dataclass(frozen=True)
class RGBDSensorConfig(SensorConfig):
    """RGBD camera sensor configuration.

    Use this class to configure two different Habitat sensors simultaneously,
    one for RGBA observations and another for depth. Both sensors will use the
    same pose, resolution and zoom factor. RGB observations are named
    "rgba", depth observations are named "depth".

    Attributes:
        sensor_id: Sensor ID unique within the sensor module. If given, then
            observations made by this sensor will be prefixed by this ID.
            i.e. "`sensor_id`.rgba"
        resolution: Camera resolution (width, height). Defaults to (64, 64).
        position: Sensor position relative to :class:`HabitatAgent`.
            Defaults to (0, 0, 0).
        rotation: Sensor rotation quaternion. Defaults to (1, 0, 0, 0).
        zoom: Camera zoom multiplier. Use >1 to increase and 0 < factor < 1 to decrease.
            Defaults to 1.0 (no zoom).
    """

    resolution: Size = (64, 64)
    zoom: float = 1.0

    def get_specs(self) -> list[SensorSpec]:
        """Return a list of Habitat sensor specs to pass to `habitat-sim`."""
        orientation = qt.as_rotation_vector(qt.quaternion(*self.rotation))

        # Configure RGBA camera
        rgba = CameraSensorSpec()
        rgba.uuid = f"{self.sensor_id}.rgba"
        rgba.sensor_type = SensorType.COLOR
        rgba.resolution = self.resolution
        rgba.position = self.position
        rgba.orientation = orientation
        # TODO: Make a parameter in config
        # rgba.noise_model = "GaussianNoiseModel"
        # rgba.noise_model_kwargs = dict(sigma=1)

        # Configure depth camera
        depth = CameraSensorSpec()
        depth.uuid = f"{self.sensor_id}.depth"
        depth.sensor_type = SensorType.DEPTH
        depth.resolution = self.resolution
        depth.position = self.position
        depth.orientation = orientation
        # This noise model doesn't seem to work with our sensor patch resolution.
        # depth.noise_model = "RedwoodDepthNoiseModel"
        # depth.noise_model_kwargs = dict(noise_multiplier=1)

        return [rgba, depth]


@dataclass(frozen=True)
class SemanticSensorConfig(SensorConfig):
    """Semantic object sensor configuration.

    Use this class to configure a sensor to observe known objects in the
    scene, returning their semantic IDs (ground truth) at each XY position.
    Semantic observations are named "semantic".

    Attributes:
        sensor_id: Optional sensor ID unique within the sensor module. If given, then
            observations made by this sensor will be prefixed by this ID.
            i.e. "`sensor_id`.semantic"
        resolution: Camera resolution (width, height). Defaults to (64, 64).
        position: Sensor position relative to :class:`HabitatAgent`.
            Defaults to (0, 0, 0).
        rotation: Sensor rotation quaternion. Defaults to (1, 0, 0, 0).
        zoom: Camera zoom multiplier. Use >1 to increase and 0 < factor < 1 to decrease.
            Defaults to 1.0 (no zoom).
    """

    resolution: Size = (64, 64)
    zoom: float = 1.0

    def get_specs(self) -> list[SensorSpec]:
        """Return a list of Habitat sensor specs to pass to `habitat-sim`."""
        orientation = qt.as_rotation_vector(qt.quaternion(*self.rotation))

        # Configure semantic camera
        semantic = CameraSensorSpec()
        semantic.channels = 1
        semantic.uuid = f"{self.sensor_id}.semantic"
        semantic.sensor_type = SensorType.SEMANTIC
        semantic.resolution = self.resolution
        semantic.position = self.position
        semantic.orientation = orientation

        return [semantic]
