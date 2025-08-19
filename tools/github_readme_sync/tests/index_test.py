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

    def test_generate_index_with_frontmatter(self):
        markdown_content = """---
title: "Test Document"
status: "completed"
rfc: "optional"
tags: "tag1, tag2"
skills: "skill1, skill2"
owner: "test-owner"
estimated-scope: "small"
---

# Test Document

This is a test document with front matter.
"""

        subdir_path = Path(self.temp_dir) / "category1" / "subcategory2"
        subdir_path.mkdir(parents=True, exist_ok=True)

        md_file_path = subdir_path / "test-doc.md"
        with open(md_file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        index_file_path = generate_index(self.temp_dir)

        self.assertTrue(os.path.exists(index_file_path))

        with open(index_file_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        self.assertEqual(len(index_data), 1)

        entry = index_data[0]

        self.assertEqual(entry["title"], "Test Document")
        self.assertEqual(entry["status"], "completed")
        self.assertEqual(entry["rfc"], "optional")
        self.assertEqual(entry["tags"], ["tag1", "tag2"])
        self.assertEqual(entry["skills"], ["skill1", "skill2"])
        self.assertEqual(entry["owner"], ["test-owner"])
        self.assertEqual(entry["estimated-scope"], "small")
        self.assertEqual(entry["slug"], "test-doc")
        self.assertTrue(entry["path"].endswith("/category1/subcategory2/test-doc.md"))
        self.assertEqual(entry["path1"], "category1")
        self.assertEqual(entry["path2"], "subcategory2")


if __name__ == "__main__":
    unittest.main()
