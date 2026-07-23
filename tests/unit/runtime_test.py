# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import unittest

import numpy as np

from tbp.monty.runtime import is_location_only_step
from tests.unit.frameworks.models.buffer_test import create_mock_message


class IsLocationOnlyStepTest(unittest.TestCase):
    def sm(self, location, process_features_in_lm=True):
        return create_mock_message(
            "sm_0",
            "SM",
            location,
            on_object=True,
            process_features_in_lm=process_features_in_lm,
        )

    def lm(self, location, process_features_in_lm=True):
        return create_mock_message(
            "lm_0",
            "LM",
            location,
            on_object=True,
            process_features_in_lm=process_features_in_lm,
        )

    def test_ignores_lm_features(self):
        location = np.zeros(3)
        sm_location_only = self.sm(location, process_features_in_lm=False)
        sm_with_features = self.sm(location, process_features_in_lm=True)
        lm_with_features = self.lm(location, process_features_in_lm=True)

        self.assertTrue(is_location_only_step([sm_location_only, lm_with_features]))
        self.assertFalse(is_location_only_step([sm_with_features]))


if __name__ == "__main__":
    unittest.main()
