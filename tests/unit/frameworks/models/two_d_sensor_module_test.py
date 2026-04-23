# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as nptest
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tbp.monty.cmp import Message
from tbp.monty.frameworks.models.two_d_sensor_module import TwoDSensorModule
from tbp.monty.math import DEFAULT_TOLERANCE

MODULE_PATH = "tbp.monty.frameworks.models.two_d_sensor_module"
a_3d_location = np.zeros(3)

_FLAT_POSE = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)

def make_message() -> Message:
    return Message(
    )

def make_module():
    pass

class TestInit(unittest.TestCase):
    def test_initial_internal_state(self):
        sm = make_module()
        assert sm._previous_3d_location is None
        assert sm._tangent_frame is None
        assert sm._previous_2d_location is None

class TestExtract2dEdge(unittest.TestCase):
    # What to test here that isn't covered by EdgeDetetor test...
    pass




