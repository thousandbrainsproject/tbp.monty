# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from tools.future_work_widget.validator import RecordValidator


class TestRecordValidator(unittest.TestCase):
    def setUp(self):
        self.temp_path = Path(tempfile.mkdtemp())
        self.expected_tags = [
            "accuracy",
            "pose",
            "learning",
            "multiobj",
        ]
        self.expected_skills = [
            "python",
            "github-actions",
            "JS",
            "HTML",
        ]

    def tearDown(self):
        shutil.rmtree(self.temp_path)

    def test_validation_files_loading(self):
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir()

        tags_file = snippets_dir / "future-work-tags.md"
        with open(tags_file, "w", encoding="utf-8") as f:
            f.write(" ".join(f"`{tag}`" for tag in self.expected_tags))

        skills_file = snippets_dir / "future-work-skills.md"
        with open(skills_file, "w", encoding="utf-8") as f:
            f.write(" ".join(f"`{skill}`" for skill in self.expected_skills))

        validator = RecordValidator(snippets_dir)

        self.assertIn("tags", validator.exact_values)
        self.assertIn("skills", validator.exact_values)

        self.assertEqual(
            sorted(validator.exact_values["tags"]), sorted(self.expected_tags)
        )
        self.assertEqual(
            sorted(validator.exact_values["skills"]), sorted(self.expected_skills)
        )

    def test_missing_validation_files_graceful(self):
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir()

        validator = RecordValidator(snippets_dir)

        self.assertEqual(len(validator.exact_values), 0)
        record = {"path1": "future-work", "path2": "test"}
        _, errors = validator.validate(record)
        self.assertEqual(len(errors), 0)

    def test_nonexistent_snippets_directory(self):
        nonexistent_dir = self.temp_path / "nonexistent"

        validator = RecordValidator(nonexistent_dir)

        self.assertEqual(len(validator.exact_values), 0)
        record = {"path1": "future-work", "path2": "test"}
        _, errors = validator.validate(record)
        self.assertEqual(len(errors), 0)

    def test_word_boundary_wrapping(self):
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir()

        tags_file = snippets_dir / "future-work-tags.md"
        with open(tags_file, "w", encoding="utf-8") as f:
            f.write("`simple-word` `accuracy` `learning`")

        validator = RecordValidator(snippets_dir)

        expected_values = [
            "simple-word",
            "accuracy",
            "learning",
        ]

        self.assertEqual(
            sorted(validator.exact_values["tags"]), sorted(expected_values)
        )

    def test_direct_validation_success(self):
        """Test RecordValidator.validate() method directly."""
        validator = RecordValidator(Path())

        record = {
            "path1": "future-work",
            "path2": "test-item",
            "title": "Test item",
            "tags": "accuracy,learning",
            "skills": "python,javascript",
            "contributor": "alice,bob",
        }

        result, errors = validator.validate(record)

        self.assertIsNotNone(result)
        self.assertEqual(len(errors), 0)
        self.assertEqual(result["path1"], "future-work")
        self.assertEqual(result["path2"], "test-item")
        self.assertEqual(result["tags"], ["accuracy", "learning"])
        self.assertEqual(result["skills"], ["python", "javascript"])
        self.assertEqual(result["contributor"], ["alice", "bob"])

    def test_direct_validation_filters_non_future_work(self):
        """Test that non-future-work items are filtered out."""
        validator = RecordValidator(Path())

        record = {
            "path1": "other-section",
            "path2": "test-item",
            "title": "Test item",
        }

        result, errors = validator.validate(record)
        self.assertIsNone(result)
        self.assertEqual(len(errors), 0)

    def test_comma_separated_field_limits(self):
        """Test limits on comma-separated fields."""
        validator = RecordValidator(Path())
        max_items = RecordValidator.MAX_COMMA_SEPARATED_ITEMS

        # Create a record with too many tags
        too_many_tags = ",".join([f"tag{i}" for i in range(max_items + 1)])
        record = {
            "path1": "future-work",
            "path2": "test-item",
            "title": "Test item",
            "tags": too_many_tags,
        }

        result, errors = validator.validate(record)

        self.assertIsNone(result)
        self.assertEqual(len(errors), 1)
        self.assertIn("tags field cannot have more than", errors[0].message)
        self.assertIn(str(max_items), errors[0].message)


if __name__ == "__main__":
    unittest.main()
