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

from pydantic import ValidationError

from tools.future_work_widget.validator import (
    MAX_COMMA_SEPARATED_ITEMS,
    FutureWorkRecord,
    load_allowed_values,
)


class TestLoadAllowedValues(unittest.TestCase):
    def setUp(self):
        self.temp_path = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_path)

    def test_validation_files_loading(self):
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir()

        expected_tags = ["accuracy", "pose", "learning", "multiobj"]
        tags_file = snippets_dir / "future-work-tags.md"
        with open(tags_file, "w", encoding="utf-8") as f:
            f.write(" ".join(f"`{tag}`" for tag in expected_tags))

        expected_skills = ["python", "github-actions", "JS", "HTML"]
        skills_file = snippets_dir / "future-work-skills.md"
        with open(skills_file, "w", encoding="utf-8") as f:
            f.write(" ".join(f"`{skill}`" for skill in expected_skills))

        allowed_values = load_allowed_values(snippets_dir)

        self.assertIn("tags", allowed_values)
        self.assertIn("skills", allowed_values)
        self.assertEqual(sorted(allowed_values["tags"]), sorted(expected_tags))
        self.assertEqual(sorted(allowed_values["skills"]), sorted(expected_skills))

    def test_missing_validation_files_graceful(self):
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir()

        allowed_values = load_allowed_values(snippets_dir)
        self.assertEqual(len(allowed_values), 0)

    def test_nonexistent_snippets_directory(self):
        nonexistent_dir = self.temp_path / "nonexistent"
        allowed_values = load_allowed_values(nonexistent_dir)
        self.assertEqual(len(allowed_values), 0)


class TestFutureWorkRecord(unittest.TestCase):
    def test_validation_success(self):
        record = {
            "path": "future-work/test-item.md",
            "path1": "future-work",
            "path2": "test-item",
            "title": "Test item",
            "tags": "accuracy,learning",
            "skills": "python,javascript",
            "contributor": "alice,bob",
            "output-type": "documentation,website",
            "improved-metric": "community-engagement,infrastructure",
        }

        validated = FutureWorkRecord.model_validate(record)
        self.assertEqual(validated.path1, "future-work")
        self.assertEqual(validated.path2, "test-item")
        self.assertEqual(validated.tags, ["accuracy", "learning"])
        self.assertEqual(validated.skills, ["python", "javascript"])
        self.assertEqual(validated.contributor, ["alice", "bob"])
        self.assertEqual(validated.output_type, ["documentation", "website"])
        self.assertEqual(
            validated.improved_metric, ["community-engagement", "infrastructure"]
        )

    def test_path_field_required(self):
        record = {
            "path1": "future-work",
            "path2": "test-item",
            "title": "Test item",
        }

        with self.assertRaises(ValidationError) as cm:
            FutureWorkRecord.model_validate(record)

        errors = cm.exception.errors()
        self.assertTrue(any("path" in str(e["loc"]) for e in errors))

    def test_comma_separated_field_limits(self):
        max_items = MAX_COMMA_SEPARATED_ITEMS
        too_many_tags = ",".join([f"tag{i}" for i in range(max_items + 1)])

        record = {
            "path": "future-work/test-item.md",
            "path1": "future-work",
            "path2": "test-item",
            "title": "Test item",
            "tags": too_many_tags,
        }

        with self.assertRaises(ValidationError) as cm:
            FutureWorkRecord.model_validate(record)

        errors = cm.exception.errors()
        self.assertTrue(any("Cannot have more than" in e["msg"] for e in errors))

    def test_comma_separated_output_type_limits(self):
        max_items = MAX_COMMA_SEPARATED_ITEMS
        too_many_output_types = ",".join([f"type{i}" for i in range(max_items + 1)])

        record = {
            "path": "future-work/test-item.md",
            "path1": "future-work",
            "path2": "test-item",
            "title": "Test item",
            "output-type": too_many_output_types,
        }

        with self.assertRaises(ValidationError) as cm:
            FutureWorkRecord.model_validate(record)

        errors = cm.exception.errors()
        self.assertTrue(any("Cannot have more than" in e["msg"] for e in errors))

    def test_comma_separated_improved_metric_limits(self):
        max_items = MAX_COMMA_SEPARATED_ITEMS
        too_many_metrics = ",".join([f"metric{i}" for i in range(max_items + 1)])

        record = {
            "path": "future-work/test-item.md",
            "path1": "future-work",
            "path2": "test-item",
            "title": "Test item",
            "improved-metric": too_many_metrics,
        }

        with self.assertRaises(ValidationError) as cm:
            FutureWorkRecord.model_validate(record)

        errors = cm.exception.errors()
        self.assertTrue(any("Cannot have more than" in e["msg"] for e in errors))

    def test_validation_with_allowed_values_context(self):
        record = {
            "path": "future-work/test-item.md",
            "path1": "future-work",
            "path2": "test-item",
            "title": "Test item",
            "tags": "accuracy",
        }

        allowed_values = {"tags": ["accuracy", "pose"]}

        validated = FutureWorkRecord.model_validate(
            record, context={"allowed_values": allowed_values}
        )
        self.assertEqual(validated.tags, ["accuracy"])

    def test_validation_with_invalid_allowed_values(self):
        record = {
            "path": "future-work/test-item.md",
            "path1": "future-work",
            "path2": "test-item",
            "title": "Test item",
            "tags": "invalid_tag",
        }

        allowed_values = {"tags": ["accuracy", "pose"]}

        with self.assertRaises(ValidationError) as cm:
            FutureWorkRecord.model_validate(
                record, context={"allowed_values": allowed_values}
            )

        errors = cm.exception.errors()
        self.assertTrue(any("Invalid tags value" in e["msg"] for e in errors))

    def test_validation_with_invalid_output_type(self):
        record = {
            "path": "future-work/test-item.md",
            "path1": "future-work",
            "path2": "test-item",
            "title": "Test item",
            "output-type": "invalid_type",
        }

        allowed_values = {"output_type": ["documentation", "website"]}

        with self.assertRaises(ValidationError) as cm:
            FutureWorkRecord.model_validate(
                record, context={"allowed_values": allowed_values}
            )

        errors = cm.exception.errors()
        self.assertTrue(any("Invalid output-type value" in e["msg"] for e in errors))

    def test_validation_with_invalid_improved_metric(self):
        record = {
            "path": "future-work/test-item.md",
            "path1": "future-work",
            "path2": "test-item",
            "title": "Test item",
            "improved-metric": "invalid_metric",
        }

        allowed_values = {"improved_metric": ["community-engagement", "infrastructure"]}

        with self.assertRaises(ValidationError) as cm:
            FutureWorkRecord.model_validate(
                record, context={"allowed_values": allowed_values}
            )

        errors = cm.exception.errors()
        self.assertTrue(
            any("Invalid improved-metric value" in e["msg"] for e in errors)
        )

    def test_validation_with_valid_output_type_and_improved_metric(self):
        record = {
            "path": "future-work/test-item.md",
            "path1": "future-work",
            "path2": "test-item",
            "title": "Test item",
            "output-type": "documentation,website",
            "improved-metric": "community-engagement,infrastructure",
        }

        allowed_values = {
            "output_type": ["documentation", "website"],
            "improved_metric": ["community-engagement", "infrastructure"],
        }

        validated = FutureWorkRecord.model_validate(
            record, context={"allowed_values": allowed_values}
        )
        self.assertEqual(validated.output_type, ["documentation", "website"])
        self.assertEqual(
            validated.improved_metric, ["community-engagement", "infrastructure"]
        )

    def test_github_username_validation(self):
        record = {
            "path": "future-work/test-item.md",
            "path1": "future-work",
            "path2": "test-item",
            "title": "Test item",
            "contributor": "valid-user",
        }

        validated = FutureWorkRecord.model_validate(record)
        self.assertEqual(validated.contributor, ["valid-user"])

    def test_invalid_github_username(self):
        record = {
            "path": "future-work/test-item.md",
            "path1": "future-work",
            "path2": "test-item",
            "title": "Test item",
            "contributor": "-invalid",
        }

        with self.assertRaises(ValidationError) as cm:
            FutureWorkRecord.model_validate(record)

        errors = cm.exception.errors()
        self.assertTrue(any("Invalid contributor username" in e["msg"] for e in errors))


if __name__ == "__main__":
    unittest.main()
