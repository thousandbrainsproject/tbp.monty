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

import numpy as np

from tools.plot.interactive.utils import (
    Bounds,
    CoordinateMapper,
    Location2D,
    Location3D,
)


class Location2DTests(unittest.TestCase):
    def setUp(self) -> None:
        self.p = Location2D(1.5, -2.0)

    def test_equality_same_values(self) -> None:
        q = Location2D(1.5, -2.0)
        self.assertEqual(self.p, q)

    def test_equality_different_values(self) -> None:
        self.assertNotEqual(self.p, Location2D(1.5, -2.1))

    def test_equality_different_class_is_false(self) -> None:
        self.assertNotEqual(self.p, Location3D(1.5, -2.0, 0.0))

    def test_to_3d(self) -> None:
        p3 = self.p.to_3d(7.0)
        self.assertIsInstance(p3, Location3D)
        self.assertEqual((p3.x, p3.y, p3.z), (1.5, -2.0, 7.0))

    def test_to_numpy(self) -> None:
        arr = self.p.to_numpy()
        self.assertEqual(arr.shape, (2,))
        np.testing.assert_array_equal(arr, np.array([1.5, -2.0], dtype=float))

    def tearDown(self) -> None:
        self.p = None


class Location3DTests(unittest.TestCase):
    def setUp(self) -> None:
        self.p = Location3D(3.0, -4.0, 5.5)

    def test_equality_same_values(self) -> None:
        q = Location3D(3.0, -4.0, 5.5)
        self.assertEqual(self.p, q)

    def test_equality_different_values(self) -> None:
        self.assertNotEqual(self.p, Location3D(3.0, -4.0, 5.6))

    def test_equality_different_class_is_false(self) -> None:
        self.assertNotEqual(self.p, Location2D(3.0, -4.0))  # different class

    def test_to_2d(self) -> None:
        p2 = self.p.to_2d()
        self.assertIsInstance(p2, Location2D)
        self.assertEqual((p2.x, p2.y), (3.0, -4.0))

    def test_to_numpy(self) -> None:
        arr = self.p.to_numpy()
        self.assertEqual(arr.shape, (3,))
        np.testing.assert_array_equal(arr, np.array([3.0, -4.0, 5.5], dtype=float))

    def tearDown(self) -> None:
        self.p = None


class BoundsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.b = Bounds(xmin=0.0, xmax=15.0, ymin=0.0, ymax=10.0)

    def test_width_height(self) -> None:
        self.assertEqual(self.b.width(), 15.0)
        self.assertEqual(self.b.height(), 10.0)

    def test_contains_interior(self) -> None:
        self.assertTrue(self.b.contains(Location2D(5.0, 5.0)))

    def test_contains_on_edges_inclusive(self) -> None:
        self.assertTrue(self.b.contains(Location2D(0.0, 0.0)))
        self.assertTrue(self.b.contains(Location2D(15.0, 10.0)))
        self.assertTrue(self.b.contains(Location2D(0.0, 10.0)))
        self.assertTrue(self.b.contains(Location2D(15.0, 0.0)))

    def test_contains_outside(self) -> None:
        self.assertFalse(self.b.contains(Location2D(-0.1, 5.0)))
        self.assertFalse(self.b.contains(Location2D(5.0, 10.1)))
        self.assertFalse(self.b.contains(Location2D(15.1, 10.1)))

    def tearDown(self) -> None:
        self.b = None


class CoordinateMapperTests(unittest.TestCase):
    def setUp(self) -> None:
        # GUI space: a 200x200 square starting at (100, 50)
        self.gui = Bounds(xmin=100.0, xmax=300.0, ymin=50.0, ymax=250.0)
        # Data space: x in [-1, 1], y in [0, 10]
        self.data = Bounds(xmin=-1.0, xmax=1.0, ymin=0.0, ymax=10.0)
        self.mapper = CoordinateMapper(gui=self.gui, data=self.data)

    def test_invalid_gui_bounds_raises(self) -> None:
        with self.assertRaises(ValueError):
            CoordinateMapper(gui=Bounds(0, 0, 0, 10), data=self.data)  # zero width
        with self.assertRaises(ValueError):
            CoordinateMapper(gui=Bounds(5, -5, 0, 10), data=self.data)  # negative width
        with self.assertRaises(ValueError):
            CoordinateMapper(gui=Bounds(0, 10, 3, 3), data=self.data)  # zero height

    def test_invalid_data_bounds_raises(self) -> None:
        with self.assertRaises(ValueError):
            CoordinateMapper(gui=self.gui, data=Bounds(-1, 1, 5, 5))  # zero height
        with self.assertRaises(ValueError):
            CoordinateMapper(gui=self.gui, data=Bounds(2, -2, 0, 10))  # negative width

    def test_map_click_to_data_coords_corners_and_center(self) -> None:
        # Corners
        bl = self.mapper.map_click_to_data_coords(Location2D(100.0, 50.0))
        tr = self.mapper.map_click_to_data_coords(Location2D(300.0, 250.0))
        self.assertAlmostEqual(bl.x, -1.0)
        self.assertAlmostEqual(bl.y, 0.0)
        self.assertAlmostEqual(tr.x, 1.0)
        self.assertAlmostEqual(tr.y, 10.0)

        # Center
        center = self.mapper.map_click_to_data_coords(Location2D(200.0, 150.0))
        self.assertAlmostEqual(center.x, 0.0)
        self.assertAlmostEqual(center.y, 5.0)

    def test_map_data_coords_to_world_corners_and_center(self) -> None:
        # Corners
        bl = self.mapper.map_data_coords_to_world(Location2D(-1.0, 0.0))
        tr = self.mapper.map_data_coords_to_world(Location2D(1.0, 10.0))
        self.assertAlmostEqual(bl.x, 100.0)
        self.assertAlmostEqual(bl.y, 50.0)
        self.assertAlmostEqual(tr.x, 300.0)
        self.assertAlmostEqual(tr.y, 250.0)

        # Center
        center = self.mapper.map_data_coords_to_world(Location2D(0.0, 5.0))
        self.assertAlmostEqual(center.x, 200.0)
        self.assertAlmostEqual(center.y, 150.0)

    def test_round_trip_gui_to_data_to_gui(self) -> None:
        pts_gui = [
            Location2D(150.0, 100.0),
            Location2D(225.0, 175.0),
            Location2D(275.0, 225.0),
        ]
        for p in pts_gui:
            data_p = self.mapper.map_click_to_data_coords(p)
            gui_back = self.mapper.map_data_coords_to_world(data_p)
            self.assertAlmostEqual(p.x, gui_back.x)
            self.assertAlmostEqual(p.y, gui_back.y)

    def test_round_trip_data_to_gui_to_data(self) -> None:
        pts_data = [Location2D(-0.5, 2.5), Location2D(0.25, 6.25), Location2D(0.9, 9.5)]
        for p in pts_data:
            gui_p = self.mapper.map_data_coords_to_world(p)
            data_back = self.mapper.map_click_to_data_coords(gui_p)
            self.assertAlmostEqual(p.x, data_back.x)
            self.assertAlmostEqual(p.y, data_back.y)

    def tearDown(self) -> None:
        self.gui = None
        self.data = None
        self.mapper = None
