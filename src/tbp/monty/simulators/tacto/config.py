# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Tacto touch sensor configuration.

See Also:
    https://github.com/facebookresearch/tacto
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml
from importlib_resources import files

from tbp.monty.simulators.resources import tacto

__all__ = [
    "DIGIT",
    "OMNITACT",
    "TactoSensorSpec",
]


@dataclass
class Gel:
    """Sensor gel specification.

    Attributes:
        origin: Center coordinate of the gel, in meters
        width: Width of the gel, y-axis, in meters
        height: Height of the gel, z-axis, in meters
        curvature: Whether or not to model the gel as curve
        curvatureMax: Deformation of the gel due to convexity
        R: Radius of curved gel
        countW: Number of samples for horizontal direction
        mesh: Gel mesh
    """

    origin: list[float]
    width: float
    height: float
    curvature: bool
    curvatureMax: float | None = 0.005  # noqa: N815 - Original YAML name
    R: float | None = 0.1
    countW: int | None = 100  # noqa: N815 - Original YAML name
    mesh: str | None = None


@dataclass
class Camera:
    """Sensor camera specification.

    Attributes:
        position: Camera position
        orientation: Euler angles, "xyz", in degrees
        yfov: Vertical field of view in degrees
        znear: Distance to the near clipping plane, in meters
        lightIDList: Select light ID list for rendering
    """

    position: list[float]
    orientation: list[float]
    yfov: float
    znear: float
    lightIDList: list[int]  # noqa: N815 - Original YAML name


@dataclass
class Lights:
    """Sensor lights specification.

    Attributes:
        origin: center of the light plane, in meters
        spot: Whether or not to use spot light
        polar: Whether or not to use polar coordinates
        coords: Cartesian coordinates (polar=False)
        xs: x coordinate of the y-z plane (polar=True)
        rs: r in polar coordinates (polar=True)
        thetas: theta in polar coordinates (polar=True)
        colors: List of RGB colors
        intensities: List of light intensities
    """

    origin: list[float]
    polar: bool
    colors: list[float]
    intensities: list[float]
    spot: bool | None = False
    shadow: bool | None = False
    coords: list[float] | None = None
    xs: list[float] | None = None
    rs: list[float] | None = None
    thetas: list[float] | None = None

    def __post_init__(self):
        if self.spot:
            raise NotImplementedError("Habitat does not support spot lights")


@dataclass
class Noise:
    """Gaussian noise calibrated on the output color.

    Attributes:
        mean: Noise mean [0-255]
        std: Noise std [0-255]
    """

    mean: int
    std: int


@dataclass
class Force:
    """Sensor force configuration.

    Attributes:
        enable: Whether or not to enable force feedback
        range_force: Dynamic range of forces used to simulate the elastomer
            deformations [0-100]
        max_deformation: Max pose depth adjustment, in meters
    """

    enable: bool
    range_force: list[int]
    max_deformation: float


@dataclass
class TactoSensorSpec:
    """Tacto sensor specifications.

    Attributes:
        name: Sensor name ('omnitact' or 'digit')
        camera: Camera specifications. One camera for DIGIT, five for OmniTact
        gel: Sensor elastomer gel configuration
        lights: Sensor LED light configuration
        noise: Gaussian noise calibration
        force: Elastomer force feedback specification

    .. _Tacto:
        https://github.com/facebookresearch/tacto
    """

    name: str
    camera: Mapping[str, Camera]
    gel: Gel
    lights: Lights
    noise: Noise | None = None
    force: Force | None = None

    def __post_init__(self):
        for k, v in self.camera.items():
            if isinstance(v, dict):
                self.camera[k] = Camera(**v)
        if isinstance(self.gel, dict):
            self.gel = Gel(**self.gel)
        if isinstance(self.lights, dict):
            self.lights = Lights(**self.lights)
        if isinstance(self.noise, dict):
            self.noise = Noise(**self.noise)
        if isinstance(self.force, dict):
            self.force = Force(**self.force)

    @classmethod
    def from_yaml(cls, yaml_file):
        with Path(yaml_file).open() as f:
            config = yaml.safe_load(f)["sensor"]
        return cls(**config)


# Digit sensor configuration. See https://arxiv.org/abs/2005.14679
DIGIT = TactoSensorSpec.from_yaml(str(files(tacto) / "config_digit.yml"))

# Omnitact sensor configuration. See https://arxiv.org/pdf/2003.06965.pdf
OMNITACT = TactoSensorSpec.from_yaml(str(files(tacto) / "config_omnitact.yml"))
