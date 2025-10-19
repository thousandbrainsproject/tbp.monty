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
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from tools.github_readme_sync.export import export
from tools.github_readme_sync.readme import ReadMe


class TestExport(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the test
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_output_dir = self.test_dir.name

    def tearDown(self):
        # Clean up the temporary directory after the test
        self.test_dir.cleanup()

    def test_export(self):
        # Create a mock instance of the ReadMe class
        mock_readme = MagicMock(spec=ReadMe)

        # Inline mock return values for clarity
        mock_readme.get_categories.return_value = [
            {"title": "Category 1", "slug": "category-1"},
            {"title": "Category 2", "slug": "category-2"},
        ]

        # Explicitly differentiate the documents for each category
        mock_readme.get_category_docs.side_effect = lambda category: (
            [{"title": "Doc 1", "slug": "doc-1"}]
            if category["slug"] == "category-1"
            else [{"title": "Doc 2", "slug": "doc-2"}]
        )

        # Explicitly differentiate the document content by slug
        mock_readme.get_doc_by_slug.side_effect = lambda slug: (
            "Content of Doc 1" if slug == "doc-1" else "Content of Doc 2"
        )

        # Call the function under test
        hierarchy = export(self.test_output_dir, mock_readme)

        # Assert the hierarchy is constructed correctly
        expected_hierarchy = [
            {
                "title": "Category 1",
                "slug": "category-1",
                "children": [{"title": "Doc 1", "slug": "doc-1", "children": []}],
            },
            {
                "title": "Category 2",
                "slug": "category-2",
                "children": [{"title": "Doc 2", "slug": "doc-2", "children": []}],
            },
        ]
        self.assertEqual(hierarchy, expected_hierarchy)

        # Assert that the directory structure is created as expected
        self.assertTrue(Path(os.path.join(self.test_output_dir, "category-1")).is_dir())
        self.assertTrue(Path(os.path.join(self.test_output_dir, "category-2")).is_dir())

        # Assert that the document files are created as expected
        self.assertTrue(
            Path(os.path.join(self.test_output_dir, "category-1", "doc-1.md")).is_file()
        )
        self.assertTrue(
            Path(os.path.join(self.test_output_dir, "category-2", "doc-2.md")).is_file()
        )

        # Assert the content of the files is correct
        with open(os.path.join(self.test_output_dir, "category-1", "doc-1.md")) as f:
            self.assertEqual(f.read(), "Content of Doc 1")

        with open(os.path.join(self.test_output_dir, "category-2", "doc-2.md")) as f:
            self.assertEqual(f.read(), "Content of Doc 2")


if __name__ == "__main__":
    unittest.main()
