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

from tools.github_readme_sync.file import find_markdown_files, get_folders


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


class TestFindMarkdownFiles(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_directories(self, dir_paths):
        """Helper function to create multiple directories.

        Returns:
            Dictionary mapping directory paths to Path objects.
        """
        created_dirs = {}
        for path in dir_paths:
            full_path = Path(self.temp_dir) / path
            full_path.mkdir(parents=True)
            created_dirs[path] = full_path
        return created_dirs

    def test_find_markdown_files_ignores_dot_directories(self):
        """Test that find_markdown_files ignores directories that start with a dot."""
        dirs = self._create_directories(
            ["regular_docs", ".git", ".vscode", "regular_docs/.hidden"]
        )
        regular_dir = dirs["regular_docs"]
        dot_git_dir = dirs[".git"]
        dot_vscode_dir = dirs[".vscode"]
        nested_dot_dir = dirs["regular_docs/.hidden"]

        (regular_dir / "readme.md").write_text("# Regular readme")
        (regular_dir / "guide.md").write_text("# Guide")
        (dot_git_dir / "config.md").write_text("# Git config")
        (dot_vscode_dir / "settings.md").write_text("# VSCode settings")
        (nested_dot_dir / "secret.md").write_text("# Secret doc")

        (regular_dir / "regular.txt").write_text("Not a markdown file")

        result = find_markdown_files(self.temp_dir)

        result_basenames = [Path(path).name for path in result]

        self.assertIn("readme.md", result_basenames)
        self.assertIn("guide.md", result_basenames)
        self.assertNotIn("config.md", result_basenames)
        self.assertNotIn("settings.md", result_basenames)
        self.assertNotIn("secret.md", result_basenames)

        self.assertEqual(len(result), 2)

    def test_find_markdown_files_with_relative_path_containing_dotdot(self):
        docs_dir = Path(self.temp_dir) / "docs"
        docs_dir.mkdir()
        (docs_dir / "readme.md").write_text("# Readme")
        (docs_dir / "guide.md").write_text("# Guide")
        (docs_dir / "subdir").mkdir()
        (docs_dir / "subdir" / "nested.md").write_text("# Nested")

        subdir = Path(self.temp_dir) / "subdir"
        subdir.mkdir()

        original_cwd = Path.cwd()
        try:
            os.chdir(subdir)
            relative_path = "../docs"

            result = find_markdown_files(relative_path)

            result_basenames = sorted([Path(path).name for path in result])
            expected_basenames = sorted(["readme.md", "guide.md", "nested.md"])
            self.assertEqual(result_basenames, expected_basenames)
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
