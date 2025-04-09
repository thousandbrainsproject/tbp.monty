# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.github_readme_sync.file import get_folders


class TestGetFolders(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        os.makedirs(Path(self.temp_dir).joinpath("folder1"))
        os.makedirs(Path(self.temp_dir).joinpath("folder2"))
        open(Path(self.temp_dir).joinpath("file1.txt"), "w").close()
        open(Path(self.temp_dir).joinpath("file2.txt"), "w").close()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_get_folders(self):
        # The expected output should only include the folders
        expected_folders = ["folder1", "folder2"]

        result = get_folders(self.temp_dir)

        # Assert that the result matches the expected folders
        self.assertEqual(sorted(result), sorted(expected_folders))

if __name__ == "__main__":
    unittest.main()
