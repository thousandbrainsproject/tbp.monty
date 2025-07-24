# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from vedo import (
    Button,
    Line,
    Mesh,
    Plotter,
    Slider2D,
    Sphere,
    Text2D,
    settings,
)
