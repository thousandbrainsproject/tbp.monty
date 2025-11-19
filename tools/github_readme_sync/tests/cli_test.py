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

import pytest

from tools.github_readme_sync.cli import main


class TestCLIArgumentErrors:
    def test_export_wrong_args(self):
        # Simulate missing version argument for the 'export' command
        test_args = ["cli.py", "export", "some_folder"]
        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                main()

    def test_check_wrong_args(self):
        # Simulate missing folder argument for the 'check' command
        test_args = ["cli.py", "check"]
        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                main()

    def test_upload_wrong_args(self):
        # Simulate missing version argument for the 'upload' command
        test_args = ["cli.py", "upload", "some_folder"]
        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                main()


if __name__ == "__main__":
    unittest.main()
