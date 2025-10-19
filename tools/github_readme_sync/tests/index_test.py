# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json
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

        index_file_path = generate_index(
            self.temp_dir, str(Path(self.temp_dir) / "index.json")
        )

        self.assertTrue(Path(index_file_path).exists())

        with open(index_file_path, encoding="utf-8") as f:
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

    def test_generate_index_invalid_parameters(self):
        """Test various invalid parameter combinations."""
        test_cases = [
            ("None docs_dir", None, "valid_output.json", ValueError),
            ("empty docs_dir", "", "valid_output.json", ValueError),
            ("None output_file", "valid_dir", None, ValueError),
            ("empty output_file", "valid_dir", "", (ValueError, OSError)),
        ]

        for case_name, docs_dir, output_file_path, expected_exception in test_cases:
            with self.subTest(case=case_name):
                actual_docs_dir = docs_dir
                actual_output_file = output_file_path

                if docs_dir == "valid_dir":
                    actual_docs_dir = self.temp_dir
                if output_file_path == "valid_output.json":
                    actual_output_file = str(Path(self.temp_dir) / "index.json")

                with self.assertRaises(expected_exception):
                    generate_index(actual_docs_dir, actual_output_file)

    def test_generate_index_nonexistent_docs_folder(self):
        nonexistent_dir = str(Path(self.temp_dir) / "nonexistent")
        output_file_path = str(Path(self.temp_dir) / "index.json")

        with self.assertRaises(ValueError) as context:
            generate_index(nonexistent_dir, output_file_path)

        self.assertIn("does not exist", str(context.exception))

    def test_malicious_frontmatter_sanitization(self):
        """Test that malicious frontmatter fields are properly sanitized."""
        frontmatter = (
            "malicious_field: \"<script>alert('xss')</script>\"\nother_field: "
            '"<img src=x onerror=alert(1)>"\ngood_field: "safe_value"\n'
        )

        index_data = self._create_file_and_generate_index("safe_category", frontmatter)

        self.assertEqual(len(index_data), 1)
        entry = index_data[0]

        self.assertNotIn("malicious_field", entry)
        self.assertEqual(entry["other_field"], '<img src="x">')
        self.assertEqual(entry["good_field"], "safe_value")

        self.assertEqual(entry["title"], "test doc")
        self.assertEqual(entry["path1"], "safe_category")


if __name__ == "__main__":
    unittest.main()
