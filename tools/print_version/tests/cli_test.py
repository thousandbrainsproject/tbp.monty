# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import sys
import unittest
from unittest.mock import patch

from tools.print_version.cli import main


class TestPrintVersionScript(unittest.TestCase):
    @patch("tools.print_version.cli.get_version")
    def test_full_version_argument(self, mock_get_version):
        mock_get_version.return_value = "1.2.3-beta"
        test_args = ["print_version.py", "full"]
        with patch.object(sys, "argv", test_args):
            with patch("builtins.print") as mock_print:
                main()
                mock_print.assert_called_once_with("1.2.3-beta")

    @patch("tools.print_version.cli.get_version")
    def test_major_version_argument(self, mock_get_version):
        mock_get_version.return_value = "1.2.3-beta"
        test_args = ["print_version.py", "major"]
        with patch.object(sys, "argv", test_args):
            with patch("builtins.print") as mock_print:
                main()
                mock_print.assert_called_once_with("1")

    @patch("tools.print_version.cli.get_version")
    def test_minor_version_argument(self, mock_get_version):
        mock_get_version.return_value = "1.2.3-beta"
        test_args = ["print_version.py", "minor"]
        with patch.object(sys, "argv", test_args):
            with patch("builtins.print") as mock_print:
                main()
                mock_print.assert_called_once_with("1.2")

    @patch("tools.print_version.cli.get_version")
    def test_patch_version_argument(self, mock_get_version):
        mock_get_version.return_value = "1.2.3-beta"
        test_args = ["print_version.py", "patch"]
        with patch.object(sys, "argv", test_args):
            with patch("builtins.print") as mock_print:
                main()
                mock_print.assert_called_once_with("1.2.3")

    def test_invalid_argument(self):
        test_args = ["print_version.py", "invalid"]
        with patch.object(sys, "argv", test_args):
            with self.assertRaises(SystemExit):
                main()


if __name__ == "__main__":
    unittest.main()
