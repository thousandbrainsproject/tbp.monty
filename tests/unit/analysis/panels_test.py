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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nptest

from analysis import panels
from analysis.panels import OKABE_ITO
from analysis.views import Points, Stream


def setUpModule() -> None:
    mpl.use("Agg")


def image_stream() -> Stream[np.ndarray]:
    # Three frames at steps 0, 2 and 3; step 1 was skipped.
    frames = np.zeros((3, 2, 2, 4), np.uint8)
    frames[:, 0, 0, 0] = [10, 20, 30]
    return Stream(frames, np.array([0, 2, 3]))


def overlay_stream() -> Stream[np.ndarray]:
    overlay = np.zeros((2, 2, 4), np.uint8)
    overlay[0, 0] = (0, 255, 0, 110)
    return Stream([overlay], np.array([2]))


def point_stream() -> Stream[Points]:
    # Two points at step 0; at step 2 the module recorded an empty cloud.
    populated = Points(
        np.array([[0.0, 0.0, 0.0], [4.0, 2.0, 3.0]]),
        {"weight": np.array([-1.0, 2.0])},
    )
    empty = Points(np.empty((0, 3)), {"weight": np.empty(0)})
    return Stream([populated, empty], np.array([0, 2]))


def max_stream(steps: tuple[int, ...] = (0, 2)) -> Stream[dict]:
    # Per LM step, each object's max evidence; mug peaks highest.
    return Stream(
        [{"mug": 1.0, "cube": 2.0}, {"mug": 3.0, "cube": 0.5}], np.array(steps)
    )


class ImageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fig = plt.figure()
        self.addCleanup(plt.close, self.fig)
        self.ax = self.fig.add_subplot()

    def test_holds_the_latest_frame_on_skipped_steps(self) -> None:
        panel = panels.Image(image_stream(), self.ax, title="Camera")

        panel.update(1)

        nptest.assert_array_equal(self.ax.images[0].get_array()[0, 0], [10, 0, 0, 0])
        self.assertIn("Step 1/3", self.ax.get_title())

    def test_overlay_is_held_like_the_frames(self) -> None:
        panel = panels.Image(image_stream(), self.ax, overlay=overlay_stream())
        overlay_image = self.ax.images[1]

        panel.update(0)
        self.assertFalse(overlay_image.get_visible())  # nothing recorded yet

        panel.update(2)
        self.assertTrue(overlay_image.get_visible())

        panel.update(3)
        self.assertTrue(overlay_image.get_visible())  # latest recording held

    def test_an_empty_overlay_stream_is_ignored(self) -> None:
        panel = panels.Image(
            image_stream(), self.ax, overlay=Stream([], np.array([], dtype=int))
        )

        panel.update(0)

        self.assertEqual(len(self.ax.images), 1)


class Scatter3DTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fig = plt.figure()
        self.addCleanup(plt.close, self.fig)
        self.ax = self.fig.add_subplot(projection="3d")
        self.panel = panels.Scatter3D(
            point_stream(),
            self.ax,
            color="weight",
            title="Points",
            fixation=Stream([np.array([9.0, 9.0, 9.0])], np.array([2])),
        )
        self.scatter, self.marker = self.ax.collections[:2]

    def test_moves_the_scatter_to_the_steps_points_in_place(self) -> None:
        self.panel.update(0)

        nptest.assert_array_equal(self.scatter._offsets3d[0], [0.0, 4.0])
        nptest.assert_array_equal(self.scatter.get_array(), [-1.0, 2.0])
        self.assertIn("(2, Step 0)", self.ax.get_title())
        self.assertIs(self.scatter, self.ax.collections[0])

    def test_holds_the_latest_points_between_recordings(self) -> None:
        self.panel.update(1)

        nptest.assert_array_equal(self.scatter._offsets3d[0], [0.0, 4.0])
        self.assertFalse(self.ax.texts[0].get_visible())

    def test_an_empty_record_shows_the_empty_text(self) -> None:
        self.panel.update(0)
        self.panel.update(2)

        self.assertEqual(len(self.scatter._offsets3d[0]), 0)
        self.assertTrue(self.ax.texts[0].get_visible())

    def test_the_fixation_marker_holds_the_latest_fixation(self) -> None:
        self.panel.update(0)
        self.assertFalse(self.marker.get_visible())

        self.panel.update(3)
        self.assertTrue(self.marker.get_visible())
        nptest.assert_array_equal(self.marker._offsets3d, [[9.0], [9.0], [9.0]])


class EvidenceTracesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fig = plt.figure()
        self.addCleanup(plt.close, self.fig)
        self.ax = self.fig.add_subplot()
        self.panel = panels.EvidenceTraces(
            max_stream(),
            Stream([3, 3], np.array([0, 2])),
            self.ax,
            title="LM_0 Max Evidence",
        )

    def test_reveals_traces_up_to_the_latest_processed_step(self) -> None:
        lines = {line.get_label(): line for line in self.ax.get_lines()}

        self.panel.update(1)  # the LM processed episode steps 0 and 2
        nptest.assert_array_equal(lines["mug"].get_ydata(), [1.0])

        self.panel.update(2)
        nptest.assert_array_equal(lines["mug"].get_ydata(), [1.0, 3.0])
        self.assertIn("LM step 1/1", self.ax.get_title())


class EvidenceRecognitionTest(unittest.TestCase):
    def test_recognition_shading_appears_once_reached(self) -> None:
        fig = plt.figure()
        self.addCleanup(plt.close, fig)
        ax = fig.add_subplot()
        panel = panels.EvidenceTraces(
            max_stream(),
            Stream([3, 3], np.array([0, 2])),
            ax,
            title="LM_0 Max Evidence",
            recognized=np.array([2]),  # episode step 2 = LM step 1
        )
        (span,) = [p for p in ax.patches if p.get_label() == "recognized"]

        panel.update(0)
        self.assertFalse(span.get_visible())

        panel.update(2)
        self.assertTrue(span.get_visible())
        self.assertEqual(span.get_width(), 1.0)  # clipped at the cursor


class EvidenceRankingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fig = plt.figure()
        self.addCleanup(plt.close, self.fig)
        self.ax = self.fig.add_subplot()

    def test_ranks_objects_at_the_latest_processed_step(self) -> None:
        panel = panels.EvidenceRanking(max_stream(), self.ax, title="LM_0 Ranking")

        panel.update(2)
        panel.update(3)

        self.assertEqual(len(self.ax.tables), 1)  # rebuilt, not accumulated
        table = self.ax.tables[0]
        self.assertEqual(table[1, 0].get_text().get_text(), "mug")
        self.assertEqual(table[1, 1].get_text().get_text(), "3.00")

    def test_shows_a_note_before_the_first_processed_step(self) -> None:
        panel = panels.EvidenceRanking(
            max_stream(steps=(2, 3)), self.ax, title="LM_0 Ranking"
        )

        panel.update(1)

        self.assertEqual(len(self.ax.tables), 0)
        self.assertTrue(self.ax.texts[0].get_visible())


class EvidenceReductionsTest(unittest.TestCase):
    def test_burst_steps_where_the_hypothesis_count_grew(self) -> None:
        counts = Stream([3, 3, 5, 5, 8], np.arange(5))

        nptest.assert_array_equal(panels.burst_steps(counts), [2, 4])

    def test_objects_sorted_by_max_evidence_ranks_by_peak(self) -> None:
        self.assertEqual(
            panels.objects_sorted_by_max_evidence(max_stream()), ["mug", "cube"]
        )


class EvidenceColorsTest(unittest.TestCase):
    def test_strongest_objects_take_the_palette_in_order(self) -> None:
        colors = panels.evidence_colors(max_stream())

        self.assertEqual(list(colors), ["mug", "cube"])
        self.assertEqual(colors["mug"], OKABE_ITO[0])
        self.assertEqual(colors["cube"], OKABE_ITO[1])

    def test_objects_beyond_the_palette_share_gray(self) -> None:
        many = Stream(
            [{f"obj_{i}": float(-i) for i in range(len(OKABE_ITO) + 2)}],
            np.array([0]),
        )

        colors = panels.evidence_colors(many)

        self.assertEqual(colors[f"obj_{len(OKABE_ITO) + 1}"], "0.8")


class SharedScaleTest(unittest.TestCase):
    def test_covers_every_stream_and_anchors_the_color_scale(self) -> None:
        bounds, vlim = panels.shared_scale(point_stream(), color="weight")

        (x_low, x_high), _, _ = bounds
        self.assertLessEqual(x_low, 0.0)
        self.assertGreaterEqual(x_high, 4.0)
        self.assertEqual(vlim, (-1.0, 2.0))

    def test_defaults_to_the_unit_scale_without_points(self) -> None:
        empty = Stream([], np.array([], dtype=int))

        bounds, vlim = panels.shared_scale(empty, color="weight")

        self.assertEqual(vlim, (0.0, 1.0))
        self.assertEqual(bounds, ([-1, 1], [-1, 1], [-1, 1]))
