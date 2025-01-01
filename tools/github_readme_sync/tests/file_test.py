# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest
from unittest.mock import patch

from tools.github_readme_sync.file import get_folders


class TestGetFolders(unittest.TestCase):
    @patch("os.listdir")
    @patch("os.path.isdir")
    def test_get_folders(self, mock_isdir, mock_listdir):
        # Mocking os.listdir to return a specific list of files/folders
        mock_listdir.return_value = ["folder1", "folder2", "file1.txt", "file2.txt"]

        # Mocking os.path.isdir to return True for folders and False for files
        mock_isdir.side_effect = lambda x: x.endswith("folder1") or x.endswith(
            "folder2"
        )

        # The expected output should only include the folders
        expected_folders = ["folder1", "folder2"]

        # Call the function with a dummy path
        result = get_folders("/dummy/path")

        # Assert that the result matches the expected folders
        self.assertEqual(result, expected_folders)

        # Ensure os.listdir was called with the correct path
        mock_listdir.assert_called_once_with("/dummy/path")

        # Ensure os.path.isdir was called correctly for each item
        mock_isdir.assert_any_call("/dummy/path/folder1")
        mock_isdir.assert_any_call("/dummy/path/folder2")
        mock_isdir.assert_any_call("/dummy/path/file1.txt")
        mock_isdir.assert_any_call("/dummy/path/file2.txt")


if __name__ == "__main__":
    unittest.main()
