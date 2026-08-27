# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


import unittest

import pandas as pd

from tbp.monty.frameworks.run_parallel import (
    parse_episode_spec,
    per_lm_stats,
)


class PerLMStatsTest(unittest.TestCase):
    def test_reconstructs_named_accuracy(self):
        eval_stats = pd.DataFrame(
            [
                ("LM_0", "correct"),
                ("LM_1", "confused"),
                ("LM_0", "correct_mlh"),
                ("LM_1", "confused_mlh"),
                ("LM_0", "no_match"),
                ("LM_1", "time_out"),
                ("LM_0", "confused"),
                ("LM_1", "correct"),
            ],
            columns=["lm_id", "primary_performance"],
        )

        self.assertEqual(
            per_lm_stats(eval_stats),
            {
                "LM_0/overall/percent_correct": 50.0,
                "LM_1/overall/percent_correct": 25.0,
            },
            "per-LM accuracy should match the expected LM_0 and LM_1 keys and values",
        )


class ParseEpisodeSpecTest(unittest.TestCase):
    def test_selects_all(self):
        self.assertEqual(parse_episode_spec(None, total=5), [0, 1, 2, 3, 4])
        self.assertEqual(parse_episode_spec("", total=3), [0, 1, 2])
        self.assertEqual(parse_episode_spec(":", total=3), [0, 1, 2])
        self.assertEqual(parse_episode_spec("all", total=4), [0, 1, 2, 3])

    def test_single_index_valid(self):
        self.assertEqual(parse_episode_spec("3", total=6), [3])

    def test_single_index_out_of_bounds_raises(self):
        with self.assertRaisesRegex(ValueError, "not a valid index"):
            parse_episode_spec("5", total=5)

    def test_closed_range(self):
        self.assertEqual(parse_episode_spec("2:5", total=10), [2, 3, 4])

    def test_closed_range_equal_start_end_invalid(self):
        with self.assertRaisesRegex(ValueError, "not a valid range"):
            parse_episode_spec("3:3", total=10)

    def test_closed_range_start_greater_than_end_invalid(self):
        with self.assertRaisesRegex(ValueError, "not a valid range"):
            parse_episode_spec("5:3", total=10)

    def test_closed_range_end_equals_total_ok(self):
        self.assertEqual(parse_episode_spec("0:4", total=4), [0, 1, 2, 3])

    def test_closed_range_end_greater_than_total_raises(self):
        with self.assertRaisesRegex(ValueError, "not a valid range"):
            parse_episode_spec("0:6", total=5)

    def test_open_left(self):
        self.assertEqual(parse_episode_spec(":3", total=10), [0, 1, 2])

    def test_open_left_requires_positive_end(self):
        with self.assertRaisesRegex(ValueError, "not a valid range"):
            parse_episode_spec(":0", total=10)

    def test_open_right(self):
        self.assertEqual(parse_episode_spec("7:", total=10), [7, 8, 9])

    def test_open_right_start_zero_ok(self):
        self.assertEqual(parse_episode_spec("0:", total=3), [0, 1, 2])

    def test_open_right_start_equals_total_invalid(self):
        with self.assertRaisesRegex(ValueError, "not a valid range"):
            parse_episode_spec("5:", total=5)

    def test_mixed_selection(self):
        self.assertEqual(
            parse_episode_spec("0,3,5:8", total=10),
            [0, 3, 5, 6, 7],
        )

    def test_duplicates_eliminated_and_sorted(self):
        self.assertEqual(
            parse_episode_spec("2,2,1:3,0,0", total=5),
            [0, 1, 2],
        )

    def test_whitespace_is_ignored(self):
        self.assertEqual(
            parse_episode_spec("  0 ,  2 : 4 , 3  ", total=10),
            [0, 2, 3],
        )

    def test_letters_in_closed_range_invalid(self):
        with self.assertRaisesRegex(ValueError, "not a valid selection"):
            parse_episode_spec("a:b", total=10)

    def test_double_colon_invalid(self):
        with self.assertRaisesRegex(ValueError, "not a valid selection"):
            parse_episode_spec("3::5", total=10)

    def test_word_in_list_invalid(self):
        with self.assertRaisesRegex(ValueError, "not a valid selection"):
            parse_episode_spec("0,foo,2", total=10)

    def test_total_zero_all_is_empty(self):
        self.assertEqual(parse_episode_spec("all", total=0), [])

    def test_total_zero_none_is_empty(self):
        self.assertEqual(parse_episode_spec(None, total=0), [])

    def test_total_zero_any_specific_raises(self):
        with self.assertRaises(ValueError):
            parse_episode_spec("0", total=0)
        with self.assertRaises(ValueError):
            parse_episode_spec(":1", total=0)
        with self.assertRaises(ValueError):
            parse_episode_spec("0:", total=0)
