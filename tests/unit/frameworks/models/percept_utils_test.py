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

import numpy as np

from tbp.monty.frameworks.models.percept_utils import (
    location_only,
    sm_location_mean,
    sm_percepts,
)
from tests.unit.frameworks.models.buffer_test import create_mock_message


class PerceptUtilsTest(unittest.TestCase):
    def sm(self, sender_id, location, contains_features=True):
        return create_mock_message(
            sender_id,
            "SM",
            location,
            on_object=True,
            contains_features=contains_features,
        )

    def lm(self, sender_id, location, contains_features=True):
        return create_mock_message(
            sender_id,
            "LM",
            location,
            on_object=True,
            contains_features=contains_features,
        )

    def test_sm_percepts_keeps_only_sm_senders(self):
        sm_a = self.sm("sm_0", np.zeros(3))
        sm_b = self.sm("sm_1", np.ones(3))
        lm = self.lm("lm_0", np.full(3, 2.0))

        self.assertEqual(sm_percepts([sm_a, lm, sm_b]), [sm_a, sm_b])

    def test_sm_location_mean_averages_sm_locations(self):
        sm_a = self.sm("sm_0", np.array([0.0, 0.0, 0.0]))
        sm_b = self.sm("sm_1", np.array([2.0, 4.0, 6.0]))
        lm = self.lm("lm_0", np.array([100.0, 0.0, 0.0]))

        np.testing.assert_array_equal(
            sm_location_mean([sm_a, sm_b, lm]), np.array([1.0, 2.0, 3.0])
        )

    def test_location_only_ignores_lm_features(self):
        location = np.zeros(3)
        sm_location_only = self.sm("sm_0", location, contains_features=False)
        sm_with_features = self.sm("sm_0", location, contains_features=True)
        lm_with_features = self.lm("lm_0", location, contains_features=True)

        self.assertTrue(location_only([sm_location_only, lm_with_features]))
        self.assertFalse(location_only([sm_with_features]))


if __name__ == "__main__":
    unittest.main()
