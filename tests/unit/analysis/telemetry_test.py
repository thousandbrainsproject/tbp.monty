# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as nptest

from analysis.telemetry import (
    EpisodeTelemetry,
    available_episodes,
    resolve_run_dir,
)
from tbp.monty.frameworks.loggers.npz_handler import EXPERIMENT_SUBDIRECTORY, NpzHandler

STEPS, HEIGHT, WIDTH = 3, 4, 5


def episode_stats() -> dict:
    # A small episode in the NpzHandler layout.
    return {
        "target": {"primary_target_object": "mug"},
        "LM_0": {
            # Every object's evidences are regular over steps except cube's.
            "evidences": [
                {"mug": np.arange(4.0), "cube": np.ones(step + 1)}
                for step in range(STEPS)
            ],
            "attention_regions": [None] * STEPS,
        },
        "SM_2": {
            "raw_observations": [
                {"rgba": np.full((HEIGHT, WIDTH, 4), step, np.uint8)}
                for step in range(STEPS)
            ],
            "goals": [
                {
                    "locations": np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
                    "confidences": np.array([0.9, 0.5]),
                    "sender_ids": np.array(["view_finder"] * 2),
                }
            ]
            * STEPS,
            "attention_regions": [
                {
                    "locations": np.zeros((2, 3)),
                    "weights": np.ones(2),
                    "sender_id": "v",
                },
                None,
                {
                    "locations": np.ones((1, 3)),
                    "weights": np.zeros(1),
                    "sender_id": "v",
                },
            ],
        },
        "attention_system": {"voxel_size": 0.01},
        "motor_system": {"action_sequence": []},
    }


class EpisodeTelemetryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls.tmp.name)
        path = cls.run_dir / EXPERIMENT_SUBDIRECTORY / "episode_000000.npz"
        path.parent.mkdir()
        NpzHandler(float_dtype=None).write(0, episode_stats(), path)
        cls.ep = EpisodeTelemetry.load(cls.run_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_load_resolves_runs_and_rejects_missing_episodes(self) -> None:
        self.assertEqual(available_episodes(self.run_dir), [0])
        self.assertEqual(resolve_run_dir(self.run_dir), self.run_dir)
        with self.assertRaises(FileNotFoundError):
            EpisodeTelemetry.load(self.run_dir, episode=1)
        with self.assertRaises(FileNotFoundError):
            resolve_run_dir(self.run_dir / "nowhere")

    def test_names_the_modules_by_block(self) -> None:
        self.assertEqual(self.ep.sensor_modules, ["SM_2"])
        self.assertEqual(self.ep.learning_modules, ["LM_0"])

    def test_steps_maps_lm_records_onto_processed_steps(self) -> None:
        ep = EpisodeTelemetry(
            {
                "LM_0": {
                    "lm_processed_steps": [True, False, True, False],
                    "evidences": [{"mug": [1.0]}, {"mug": [2.0]}],
                    "attention_regions": [None] * 4,
                },
                "SM_0": {"salience_maps": [np.zeros((2, 2))] * 3},
            }
        )

        nptest.assert_array_equal(ep.episode_steps("LM_0/evidences"), [0, 2])
        # One record per episode step (per-step telemetry) stays on every step.
        nptest.assert_array_equal(
            ep.episode_steps("LM_0/attention_regions"), [0, 1, 2, 3]
        )
        nptest.assert_array_equal(ep.episode_steps("SM_0/salience_maps"), [0, 1, 2])

    def test_steps_requires_lm_processed_steps_for_lm_records(self) -> None:
        ep = EpisodeTelemetry({"LM_0": {"attention_regions": [None, None]}})

        with self.assertRaisesRegex(KeyError, "lm_processed_steps"):
            ep.episode_steps("LM_0/attention_regions")

    def test_call_gets_stacks_or_collects_by_template(self) -> None:
        frames = self.ep("SM_2/raw_observations/*/rgba")
        self.assertEqual(frames.shape, (STEPS, HEIGHT, WIDTH, 4))
        self.assertEqual(self.ep("SM_2/raw_observations/1/rgba")[0, 0, 0], 1)
        self.assertEqual(self.ep("target/primary_target_object"), "mug")

        # Steps stack when they can; dict keys (object names) never stack.
        self.assertEqual(
            self.ep("LM_0/evidences/{step}/{object}", object="mug").shape, (STEPS, 4)
        )
        ragged = self.ep("LM_0/evidences/{step}/{object}", object="cube")
        self.assertEqual(list(ragged), [(0,), (1,), (2,)])
        nptest.assert_array_equal(ragged[(1,)], np.ones(2))
        by_object = self.ep("LM_0/evidences/0/{object}")
        self.assertEqual(set(by_object), {("mug",), ("cube",)})
        self.assertEqual(list(self.ep("*/evidences/0/mug")), [("LM_0",)])
