# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.github_readme_sync.index import generate_index


class TestGenerateIndex(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_file_and_generate_index(
        self, subdir: str, frontmatter_fields: str
    ) -> list:
        subdir_path = Path(self.temp_dir) / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)

        content = f"---\ntitle: test doc\n{frontmatter_fields}\n---\n"
        md_file_path = subdir_path / "test-doc.md"
        with open(md_file_path, "w", encoding="utf-8") as f:
            f.write(content)

        output_file_path = os.path.join(self.temp_dir, "index.json")
        index_file_path = generate_index(self.temp_dir, output_file_path)

        self.assertTrue(os.path.exists(index_file_path))

        with open(index_file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_generate_index_with_frontmatter(self):
        frontmatter = 'status: "completed"\n'

        index_data = self._create_file_and_generate_index("category1", frontmatter)

        self.assertEqual(len(index_data), 1)

        entry = index_data[0]

        self.assertEqual(entry["status"], "completed")
        self.assertEqual(entry["title"], "test doc")
        self.assertTrue(entry["path"].endswith("/category1/test-doc.md"))
        self.assertEqual(entry["path1"], "category1")
        self.assertNotIn("path2", entry)
        self.assertEqual(entry["slug"], "test-doc")

    def test_generate_index_with_subdirs(self):
        frontmatter = 'status: "completed"\n'

        index_data = self._create_file_and_generate_index(
            "category/subcategory/subsubcategory", frontmatter
        )

        self.assertEqual(len(index_data), 1)

        entry = index_data[0]

        self.assertEqual(entry["status"], "completed")
        self.assertTrue(
            entry["path"].endswith("/category/subcategory/subsubcategory/test-doc.md")
        )
        self.assertEqual(entry["path1"], "category")
        self.assertEqual(entry["path2"], "subcategory")
        self.assertEqual(entry["path3"], "subsubcategory")


if __name__ == "__main__":
    unittest.main()
