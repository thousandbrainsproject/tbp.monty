# Copyright 2026 Thousand Brains Project
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

from tbp.monty.frameworks.models.sm_goal_generation import NoopSMGoalGenerator


class NoopSMGoalGeneratorTest(unittest.TestCase):
    def test_never_proposes(self) -> None:
        gsg = NoopSMGoalGenerator()
        self.assertEqual(gsg(Mock(), Mock()), [])
        self.assertEqual(gsg(Mock(), Mock(), motor_only_step=True), [])
        self.assertEqual(gsg.state_dict(), {})
