# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np
from hypothesis import given

from tbp.monty.frameworks.models.two_d_sensor_module import TwoDSensorModule
from tests.unit.frameworks.utils.edge_detection_test import (
    PATCH_SIZE,
    sensor_observation,
)

DEFAULT_FEATURES = [
    "pose_vectors",
    "principal_curvatures",
    "edge_strength",
    "coherence",
]


class TwoDSensorModuleTest(unittest.TestCase):
    @given(obs=sensor_observation(patterns=["horizontal_edge"]))
    def test_basic_step(self, obs):
        obs.update(
            semantic_3d=np.ones((PATCH_SIZE * PATCH_SIZE, 4), dtype=int),
            sensor_frame_data=None,
        )
        sm = TwoDSensorModule("test", features=DEFAULT_FEATURES)
        msg = sm.step(ctx=Mock(), observation=obs, motor_only_step=False)
        print(msg)
