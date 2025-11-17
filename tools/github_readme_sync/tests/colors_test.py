# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import importlib
import os
import unittest
from unittest import mock

import pytest

from tools.github_readme_sync import colors  # Import the colors module initially


@pytest.fixture
def isatty_true():
    with mock.patch("sys.stdout.isatty", return_value=True):
        yield


@pytest.fixture
def isatty_false():
    with mock.patch("sys.stdout.isatty", return_value=False):
        yield


class TestColors(unittest.TestCase):
    @mock.patch.dict(os.environ, {"CI": "true"})
    def test_colors_disabled_in_ci(self):
        # Reload colors module after changing the environment
        importlib.reload(colors)
        self.assertFalse(colors._supports_color())
        self.assertEqual(colors.RED, "")
        self.assertEqual(colors.GRAY, "")
        self.assertEqual(colors.GREEN, "")
        self.assertEqual(colors.WHITE, "")
        self.assertEqual(colors.CYAN, "")
        self.assertEqual(colors.BLUE, "")
        self.assertEqual(colors.RESET, "")

    @mock.patch.dict(os.environ, {"CI": "false"})
    @mock.patch("sys.platform", "win32")
    def test_colors_enabled_on_windows(self):
        # Reload colors module after changing the environment
        importlib.reload(colors)
        self.assertTrue(colors._supports_color())
        self.assertNotEqual(colors.RED, "")
        self.assertNotEqual(colors.GRAY, "")
        self.assertNotEqual(colors.GREEN, "")
        self.assertNotEqual(colors.WHITE, "")
        self.assertNotEqual(colors.CYAN, "")
        self.assertNotEqual(colors.BLUE, "")
        self.assertNotEqual(colors.RESET, "")

    @mock.patch.dict(os.environ, {"CI": "false"})
    @pytest.mark.usefixtures("isatty_true")
    @mock.patch("sys.platform", "linux")
    def test_colors_enabled_on_linux_with_tty(self):
        # Reload colors module after changing the environment
        importlib.reload(colors)
        self.assertTrue(colors._supports_color())
        self.assertNotEqual(colors.RED, "")
        self.assertNotEqual(colors.GRAY, "")
        self.assertNotEqual(colors.GREEN, "")
        self.assertNotEqual(colors.WHITE, "")
        self.assertNotEqual(colors.CYAN, "")
        self.assertNotEqual(colors.BLUE, "")
        self.assertNotEqual(colors.RESET, "")

    @mock.patch.dict(os.environ, {"CI": "false"})
    @pytest.mark.usefixtures("isatty_false")
    @mock.patch("sys.platform", "linux")
    def test_colors_disabled_on_linux_without_tty(self):
        # Reload colors module after changing the environment
        importlib.reload(colors)
        self.assertFalse(colors._supports_color())
        self.assertEqual(colors.RED, "")
        self.assertEqual(colors.GRAY, "")
        self.assertEqual(colors.GREEN, "")
        self.assertEqual(colors.WHITE, "")
        self.assertEqual(colors.CYAN, "")
        self.assertEqual(colors.BLUE, "")
        self.assertEqual(colors.RESET, "")


if __name__ == "__main__":
    unittest.main()
