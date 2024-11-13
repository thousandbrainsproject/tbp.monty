# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

from tests.unit.frameworks.actions.fakes.actuator import FakeActuator


class FakeActuatorTest(unittest.TestCase):
    def test_fake_actuator_implements_actuator(self):
        """Will fail if FakeActuator does not implement all methods of Actuator."""
        actuator = FakeActuator()
        self.assertTrue(isinstance(actuator, FakeActuator))
