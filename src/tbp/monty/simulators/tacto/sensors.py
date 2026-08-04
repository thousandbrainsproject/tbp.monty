# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Tacto touch sensor implementation in habitat-sim.

See Also:
    https://github.com/facebookresearch/tacto
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from habitat_sim.sensor import CameraSensorSpec, SensorSpec, SensorType

from tbp.monty.simulators.habitat import SensorConfig
from tbp.monty.simulators.tacto import DIGIT, TactoSensorSpec

__all__ = ["TactoSensor"]


@dataclass(frozen=True)
class TactoSensor(SensorConfig):
    """Base class common for all tacto sensors.

    Each specific sensor implementation should inherit this class and pass the
    appropriate configuration file ('digit' or 'omnitact').

    Attributes:
        sensor_id: Sensor ID unique within the sensor module.
            The observations made by this sensor will be prefixed by
            this id, i.e. "`sensor_id`.cam0"
        resolution: Camera resolution (width, height). Default (32, 48).
        position: Sensor position relative to :class:`HabitatAgent`.
            Default (0, 0, 0).
        rotation: Sensor rotation quaternion. Default (1, 0, 0, 0).
        config: Tacto Sensor specification (DIGIT, OMNITACT)
    """

    resolution: list[float] = field(default_factory=lambda: [32.0, 48.0])
    config: TactoSensorSpec = DIGIT
    _depths: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        # Curved gel surface
        gel = self.config.gel
        x0, y0, z0 = gel.origin
        n = self.resolution[0]
        m = self.resolution[1]
        r = gel.R
        x_max = gel.curvatureMax

        # Create curvature depth map
        y = np.linspace(y0 - gel.width / 2, y0 + gel.width / 2, n)
        z = np.linspace(z0 - gel.height / 2, z0 + gel.height / 2, m)
        yy, zz = np.meshgrid(y, z)
        hh = r - np.maximum(0, r**2 - (yy - y0) ** 2 - (zz - z0) ** 2) ** 0.5
        depth = x0 - x_max * hh / hh.max()

        # Save curvature depth map
        for name in self.config.camera:
            self._depths[name] = depth

    def get_specs(self) -> list[SensorSpec]:
        # rotation = Rotation.from_quat(self.rotation)
        # origin = np.array(self.position)
        specs = []
        for name, config in self.config.camera.items():
            # Create habitat depth camera based on tacto camera specs
            camera = CameraSensorSpec()
            camera.uuid = f"{self.sensor_id}.{name}"
            camera.sensor_type = SensorType.DEPTH
            camera.channels = 1
            camera.near = config.znear
            camera.hfov = config.yfov
            camera.resolution = self.resolution
            camera.position = self.position

            specs.append(camera)

        return specs

    def process_observations(self, sensor_obs):
        for name in sensor_obs:
            if name in self._depths:
                # Get gel curvature depth map
                depth0 = self._depths[name]
                obs = np.clip(sensor_obs[name], 0.0, depth0)
                sensor_obs[name] = obs

        return sensor_obs
