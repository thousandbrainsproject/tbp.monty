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
import numpy.testing as nptest

from analysis import views
from analysis.telemetry import EpisodeTelemetry
from analysis.views import Stream

STEPS = 4


def episode() -> EpisodeTelemetry:
    # A 4-step episode in which SM_0 skipped step 1 (exploring) and LM_0
    # processed steps 0 and 2 only.
    rgba = np.zeros((3, 2, 2, 4), np.uint8)
    rgba[:, 0, 0, 0] = [0, 2, 3]
    locations = np.zeros((3, 4, 4))
    locations[:, 3, 0] = [0.0, 2.0, 3.0]  # the center pixel of a 2x2 frame
    return EpisodeTelemetry(
        {
            "SM_0": {
                "raw_observations": [
                    {"rgba": rgba[i], "semantic_3d": locations[i]} for i in range(3)
                ],
                "salience_maps": [np.full((2, 2), i) for i in range(3)],
                "segmentation_maps": [np.array([[1, 0], [0, 0]]), None, None],
                "attention_regions": [
                    {
                        "locations": np.ones((2, 3)),
                        "weights": [1.0, 0.5],
                        "sender_id": "s",
                    },
                    None,
                    {
                        "locations": np.zeros((1, 3)),
                        "weights": [-1.0],
                        "sender_id": "s",
                    },
                ],
                "goals": [
                    {
                        "locations": np.ones((2, 3)),
                        "confidences": [0.1, 0.9],
                        "sender_ids": ["s", "s"],
                    }
                ]
                * 3,
            },
            "LM_0": {
                "attention_regions": [
                    {
                        "locations": np.ones((2, 3)),
                        "weights": [1.0, 0.5],
                        "sender_id": "s",
                    },
                    None,
                ],
                "evidences": [
                    {"mug": [1.0, 3.0], "cube": [2.0]},
                    {"mug": [4.0], "cube": [0.5, 0.7]},
                ],
                "lm_processed_steps": [True, False, True, False],
            },
            "attention_system": {
                "voxel_size": 0.1,
                "grids": [{"voxel_size": 0.1, "voxels": [[1, 2, 3]], "weight": [0.7]}]
                * STEPS,
            },
            "motor_system": {
                "goals_in": [
                    {
                        "locations": np.zeros((1, 3)),
                        "confidences": [0.5],
                        "sender_ids": ["s"],
                    }
                ]
                * STEPS
            },
        }
    )


class StreamTest(unittest.TestCase):
    def setUp(self) -> None:
        self.stream = Stream(["a", "b", "c"], np.array([0, 2, 3]))

    def test_spans_the_episode_steps(self) -> None:
        self.assertEqual(len(self.stream), 3)
        self.assertEqual(self.stream.n_steps, 4)
        self.assertEqual(list(self.stream), ["a", "b", "c"])

    def test_latest_holds_between_recordings(self) -> None:
        ticks = [self.stream.latest(s) for s in range(4)]

        self.assertEqual([t.value for t in ticks], ["a", "a", "b", "c"])
        self.assertEqual([t.index for t in ticks], [0, 0, 1, 2])
        self.assertEqual([t.fresh for t in ticks], [True, False, True, True])

    def test_map_transforms_values_and_keeps_the_steps(self) -> None:
        doubled = self.stream.map(lambda value: value * 2)

        self.assertEqual(list(doubled), ["aa", "bb", "cc"])
        self.assertEqual(doubled.steps.tolist(), [0, 2, 3])

    def test_nothing_before_the_first_recording(self) -> None:
        late = Stream(["a"], np.array([2]))

        self.assertIsNone(late.latest(1))
        self.assertEqual(Stream([], np.array([])).n_steps, 0)


class PointsTest(unittest.TestCase):
    def test_concat_stacks_locations_and_features(self) -> None:
        a = views.Points(np.zeros((2, 3)), {"weight": np.array([1.0, 2.0])})
        b = views.Points(np.ones((1, 3)), {"weight": np.array([3.0])})

        both = views.Points.concat([a, views.Points(np.empty((0, 3)), {}), b])

        self.assertEqual(both.locations.shape, (3, 3))
        self.assertEqual(list(both["weight"]), [1.0, 2.0, 3.0])

    def test_concat_of_nothing_is_empty(self) -> None:
        self.assertEqual(len(views.Points.concat([])), 0)


class ViewsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ep = episode()

    def test_sensor_streams_record_every_step(self) -> None:
        frames = views.rgba_stream(self.ep, "SM_0")

        self.assertEqual(frames.steps.tolist(), [0, 1, 2])
        self.assertEqual(frames.latest(1).value[0, 0, 0], 2)
        salience = views.salience_map_stream(self.ep, "SM_0")
        self.assertEqual(salience.latest(2).value[0, 0], 2)
        nptest.assert_array_equal(
            views.fixation_point_stream(self.ep, "SM_0").latest(2).value,
            [3.0, 0, 0],
        )

    def test_segmentation_maps_keep_the_recorded_steps(self) -> None:
        masks = views.segmentation_map_stream(self.ep, "SM_0")

        self.assertEqual(masks.steps.tolist(), [0, 1, 2])
        self.assertIsNone(masks.latest(1).value)  # recorded, but no mask

    def test_segmentation_overlays_tick_with_the_frames(self) -> None:
        overlays = views.segmentation_overlay_stream(self.ep, "SM_0")

        self.assertEqual(overlays.steps.tolist(), [0, 1, 2])
        overlay = overlays.latest(0).value
        self.assertEqual(overlay.shape, (2, 2, 4))
        self.assertGreater(overlay[0, 0, 3], 0)  # masked: translucent green
        self.assertEqual(overlay[0, 0, 1], 255)
        self.assertEqual(overlay[1, 1, 3], 0)  # unmasked: fully transparent
        held = overlays.latest(1)  # maskless step: no tint held
        self.assertEqual(held.value.max(), 0)

    def test_no_overlays_when_no_masks_were_recorded(self) -> None:
        self.ep.blocks["SM_0"]["segmentation_maps"] = [None, None, None]

        self.assertEqual(len(views.segmentation_overlay_stream(self.ep, "SM_0")), 0)

    def test_point_streams_tick_at_their_modules_rate(self) -> None:
        regions = views.attention_region_stream(self.ep, "SM_0")
        goals = views.goal_point_stream(self.ep)
        voxels = views.attention_grid_stream(self.ep)

        self.assertEqual(regions.steps.tolist(), [0, 1, 2])
        self.assertEqual(list(regions.latest(0).value["weight"]), [1.0, 0.5])
        self.assertEqual(len(regions.latest(1).value), 0)  # proposed nothing: empty
        self.assertEqual(list(regions.latest(2).value["weight"]), [-1.0])
        self.assertEqual(goals.n_steps, STEPS)
        self.assertEqual(goals.latest(3).value.locations.shape, (1, 3))
        self.assertEqual(list(goals.latest(3).value.features), ["confidence"])
        nptest.assert_allclose(voxels.latest(0).value.locations, [[0.1, 0.2, 0.3]])

    def test_lm_point_streams_land_on_the_steps_it_processed(self) -> None:
        regions = views.attention_region_stream(self.ep, "LM_0")

        self.assertEqual(regions.steps.tolist(), [0, 2])
        self.assertEqual(len(regions.latest(1).value), 2)  # held from LM step 0
        self.assertEqual(len(regions.latest(2).value), 0)  # LM proposed nothing

    def test_recognized_steps_run_from_the_terminal_match_onward(self) -> None:
        self.ep.blocks["LM_0"]["individual_ts_reached_at_step"] = 1
        self.ep.blocks["LM_0"]["individual_ts_object"] = "mug"

        nptest.assert_array_equal(views.recognized_steps(self.ep, "LM_0"), [2])

    def test_no_recognized_steps_without_a_terminal_match(self) -> None:
        # The fixture's LM_0 never reached a terminal match.
        self.assertEqual(len(views.recognized_steps(self.ep, "LM_0")), 0)

    def test_lm_records_of_every_step_stay_on_every_step(self) -> None:
        # One record per episode step (per-step telemetry), not per
        # processed step: the count tells them apart.
        self.ep.blocks["LM_0"]["attention_regions"] = [None] * STEPS

        regions = views.attention_region_stream(self.ep, "LM_0")

        self.assertEqual(regions.steps.tolist(), [0, 1, 2, 3])

    def test_evidence_stream_lands_on_the_steps_the_lm_processed(self) -> None:
        evidences = views.evidence_stream(self.ep, "LM_0")
        maxes = evidences.map(views.max_evidence)
        counts = evidences.map(views.hypothesis_count)

        self.assertEqual(evidences.steps.tolist(), [0, 2])
        self.assertEqual(
            [evidences.latest(s).index for s in range(4)], [0, 0, 1, 1]
        )
        self.assertEqual(maxes.latest(1).value, {"mug": 3.0, "cube": 2.0})
        self.assertEqual(maxes.latest(3).value, {"mug": 4.0, "cube": 0.7})
        self.assertEqual(list(counts), [3, 3])

    def test_evidence_stream_is_empty_for_a_module_without_any(self) -> None:
        self.ep.blocks["LM_1"] = {"evidences": [], "lm_processed_steps": []}

        self.assertEqual(len(views.evidence_stream(self.ep, "LM_1")), 0)
