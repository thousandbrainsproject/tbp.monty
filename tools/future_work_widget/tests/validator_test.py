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
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_validation_files_loading(self):
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir()

        tags_file = snippets_dir / "future-work-tags.md"
        with open(tags_file, "w", encoding="utf-8") as f:
            f.write("`accuracy` `pose` `learning` `multiobj`")

        skills_file = snippets_dir / "future-work-skills.md"
        with open(skills_file, "w", encoding="utf-8") as f:
            f.write("`python` `github-actions` `JS` `HTML`")

        validator = RecordValidator(str(snippets_dir))

        self.assertIn("tags", validator.validation_sets)
        self.assertIn("skills", validator.validation_sets)

        expected_tags = [
            "\\baccuracy\\b",
            "\\bpose\\b",
            "\\blearning\\b",
            "\\bmultiobj\\b",
        ]
        expected_skills = [
            "\\bpython\\b",
            "\\bgithub-actions\\b",
            "\\bJS\\b",
            "\\bHTML\\b",
        ]

        self.assertEqual(
            sorted(validator.validation_sets["tags"]), sorted(expected_tags)
        )
        self.assertEqual(
            sorted(validator.validation_sets["skills"]), sorted(expected_skills)
        )

    def test_missing_validation_files_graceful(self):
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir()

        validator = RecordValidator(str(snippets_dir))

        self.assertEqual(len(validator.validation_sets), 0)
        record = {"path1": "future-work", "path2": "test"}
        _, errors = validator.validate(record)
        self.assertEqual(len(errors), 0)

    def test_nonexistent_snippets_directory(self):
        nonexistent_dir = str(self.temp_path / "nonexistent")

        validator = RecordValidator(nonexistent_dir)

        self.assertEqual(len(validator.validation_sets), 0)
        record = {"path1": "future-work", "path2": "test"}
        _, errors = validator.validate(record)
        self.assertEqual(len(errors), 0)

    def test_word_boundary_wrapping(self):
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir()

        tags_file = snippets_dir / "future-work-tags.md"
        with open(tags_file, "w", encoding="utf-8") as f:
            f.write("`simple-word` `complex.pattern` `https://github.com/.*`")

        validator = RecordValidator(str(snippets_dir))

        expected_patterns = [
            "\\bsimple-word\\b",
            "complex.pattern",
            "https://github.com/.*",
        ]

        self.assertEqual(
            sorted(validator.validation_sets["tags"]), sorted(expected_patterns)
        )

    def test_direct_validation_success(self):
        """Test RecordValidator.validate() method directly."""
        validator = RecordValidator()

        record = {
            "path1": "future-work",
            "path2": "test-item",
            "title": "Test item",
            "tags": "accuracy,learning",
            "skills": "python,javascript",
            "owner": "alice,bob",
        }

        result, errors = validator.validate(record)

        self.assertIsNotNone(result)
        self.assertEqual(len(errors), 0)
        self.assertEqual(result["path1"], "future-work")
        self.assertEqual(result["path2"], "test-item")
        self.assertEqual(result["tags"], ["accuracy", "learning"])
        self.assertEqual(result["skills"], ["python", "javascript"])
        self.assertEqual(result["owner"], ["alice", "bob"])

    def test_direct_validation_filters_non_future_work(self):
        """Test that non-future-work items are filtered out."""
        validator = RecordValidator()

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
        validator = RecordValidator()
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

        self.assertIsNotNone(result)
        self.assertEqual(len(errors), 1)
        self.assertIn("tags field cannot have more than", errors[0].message)
        self.assertIn(str(max_items), errors[0].message)


if __name__ == "__main__":
    unittest.main()
