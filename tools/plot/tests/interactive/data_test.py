# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest

from tools.plot.interactive.data import DataLocator, DataLocatorStep, DataParser


class FakeDataParser(DataParser):
    """Parser class that bypasses loading by injecting data directly."""

    def __init__(self):
        self.data = {
            "runs": [
                {
                    "episodes": [
                        {"steps": [{"value": "r0e0s0"}, {"value": "r0e0s1"}]},
                        {"steps": [{"value": "r0e1s0"}]},
                    ]
                },
                {
                    "episodes": [
                        {
                            "steps": [
                                {"value": "r1e0s0"},
                                {"value": "r1e0s1"},
                                {"value": "r1e0s2"},
                            ]
                        }
                    ]
                },
            ],
            "mlh": {"graph_id": "mug", "pose": [0, 0, 0]},
        }


def make_run_episode_step_locator(
    run_val: int | None = None,
    ep_val: int | None = None,
    step_val: int | None = None,
) -> DataLocator:
    """Returns a locator for: runs[run_idx].episodes[ep_idx].steps[step_idx].value."""
    return DataLocator(
        path=[
            DataLocatorStep.key("runs", "runs"),
            DataLocatorStep.index("run_idx", run_val),
            DataLocatorStep.key("episodes", "episodes"),
            DataLocatorStep.index("ep_idx", ep_val),
            DataLocatorStep.key("steps", "steps"),
            DataLocatorStep.index("step_idx", step_val),
            DataLocatorStep.key("value", "value"),
        ]
    )


class BaseWithSampleData(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = FakeDataParser()

    def test_extract_success_all_kwargs(self) -> None:
        loc = make_run_episode_step_locator()
        out = self.parser.extract(loc, run_idx=1, ep_idx=0, step_idx=2)
        self.assertEqual(out, "r1e0s2")

    def test_extract_success_mixed_fixed_and_kwargs(self) -> None:
        # Fix run=0 and ep=1 in the locator, leave step unresolved
        loc = make_run_episode_step_locator(run_val=0, ep_val=1)
        out = self.parser.extract(loc, step_idx=0)
        self.assertEqual(out, "r0e1s0")

    def test_extract_missing_required_step_raises(self) -> None:
        loc = make_run_episode_step_locator()
        with self.assertRaises(ValueError) as ctx:
            self.parser.extract(loc, run_idx=0, ep_idx=0)

    def test_extract_type_mismatch_raises(self) -> None:
        loc = make_run_episode_step_locator()
        cases = [
            {"run_idx": "0", "ep_idx": 0, "step_idx": 0},
            {"run_idx": 0, "ep_idx": "0", "step_idx": 0},
            {"run_idx": 0, "ep_idx": 0, "step_idx": "0"},
        ]
        for bad in cases:
            with self.assertRaises(ValueError) as ctx:
                self.parser.extract(loc, **bad)

    def test_query_returns_index_candidates(self) -> None:
        # first missing is run_idx (index)
        loc = make_run_episode_step_locator()
        candidates = self.parser.query(loc)
        self.assertEqual(candidates, [0, 1])

    def test_query_returns_key_candidates(self) -> None:
        loc = DataLocator(
            path=[
                DataLocatorStep.key("mlh", "mlh"),
                DataLocatorStep.key("field", None),
            ]
        )
        candidates = self.parser.query(loc)
        self.assertEqual(sorted(candidates), ["graph_id", "pose"])

    def test_query_raises_when_no_missing(self) -> None:
        loc = make_run_episode_step_locator(run_val=0, ep_val=0, step_val=1)
        with self.assertRaises(ValueError):
            self.parser.query(loc)

    def test_missing_steps_identifies_none_values(self) -> None:
        loc = DataLocator(
            path=[
                DataLocatorStep.key("runs", "runs"),
                DataLocatorStep.index("run_idx", None),
                DataLocatorStep.key("episodes", "episodes"),
                DataLocatorStep.index("ep_idx", None),
            ]
        )
        missing = loc.missing_steps()
        self.assertEqual([s.name for s in missing], ["run_idx", "ep_idx"])

    def test_extend_clones_and_appends(self) -> None:
        base = DataLocator(path=[DataLocatorStep.key("runs", "runs")])
        ext = base.extend([DataLocatorStep.index("run_idx", 1)])

        self.assertEqual(len(base.path), 1)
        self.assertEqual([s.name for s in ext.path], ["runs", "run_idx"])

        # deep copy: mutate ext and ensure base is unchanged
        ext.path[0].value = "changed"
        self.assertEqual(base.path[0].value, "runs")

    def test_classmethods(self) -> None:
        s1 = DataLocatorStep.key("episode", "0")
        s2 = DataLocatorStep.index("step", 3)

        self.assertEqual((s1.type, s1.name, s1.value), ("key", "episode", "0"))
        self.assertEqual((s2.type, s2.name, s2.value), ("index", "step", 3))

    def tearDown(self) -> None:
        self.parser = None
        self.sample_data = None


if __name__ == "__main__":
    unittest.main()
