# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

from tools.plot.correct_percentage_per_episode import (
    plot_correct_percentage_per_episode,
)


class TestCorrectPercentagePerEpisode(unittest.TestCase):
    def test_exit_1_if_exp_path_does_not_exist(self):
        exit_code = plot_correct_percentage_per_episode("nonexistent_path")
        self.assertEqual(exit_code, 1)
