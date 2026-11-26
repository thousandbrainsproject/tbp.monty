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
from unittest.mock import patch

import pytest

from tools.github_readme_sync.cli import main


class TestPreviewCLI:
    def test_preview_version(self, capsys):
        with patch.object(sys, "argv", ["cli.py", "preview-version", "0.60", "317"]):
            main()

        assert capsys.readouterr().out == "0.60-monty.pr.317\n"

    @patch("tools.github_readme_sync.cli.delete_pr_previews")
    def test_delete_pr_previews(self, mock_delete_pr_previews):
        argv = [
            "cli.py",
            "delete-pr-previews",
            "317",
            "--legacy-version",
            "0.60-legacy-docs",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.dict("os.environ", {"README_API_KEY": "test-key"}),
        ):
            main()

        mock_delete_pr_previews.assert_called_once_with(
            317, legacy_version="0.60-legacy-docs"
        )


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
