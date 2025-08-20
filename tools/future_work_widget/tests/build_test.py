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
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from tools.future_work_widget.build import RecordValidator, build


class TestBuild(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.output_dir = tempfile.mkdtemp()
        self.output_path = Path(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        shutil.rmtree(self.output_dir)

    def _run_build_test(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Helper method to run build with test data and return results.

        Returns:
            The processed data from the output JSON file.
        """
        index_file = self.temp_path / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        build(str(index_file), str(self.output_path))

        data_file = self.output_path / "data.json"
        with open(data_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _expect_build_failure(
        self,
        input_data: List[Dict[str, Any]],
        expected_error_fragment: str = "Validation failed"
    ):
        """Helper method to test that build fails with expected error."""
        index_file = self.temp_path / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        with self.assertRaises(ValueError) as context:
            build(str(index_file), str(self.output_path))

        self.assertIn(expected_error_fragment, str(context.exception))

    def test_build_filters_future_work_items(self):
        """Test that only future-work items are included in the output."""
        input_data = [
            {
                "path1": "future-work",
                "path2": "voting-improvements",
                "title": "Improve voting mechanism",
                "content": "Test content for voting",
                "estimated-scope": "medium",
                "rfc": "required"
            },
            {
                "path1": "how-monty-works",
                "path2": "learning-modules",
                "title": "Learning modules overview",
                "content": "Test content for learning"
            }
        ]

        result_data = self._run_build_test(input_data)

        self.assertEqual(len(result_data), 1)
        future_work_titles = [item["title"] for item in result_data]
        self.assertIn("Improve voting mechanism", future_work_titles)
        self.assertNotIn("Learning modules overview", future_work_titles)

        for item in result_data:
            self.assertEqual(item["path1"], "future-work")

    def test_data_transformations(self):
        """Test various data transformations with parameterized inputs."""
        test_cases = [
            {
                "name": "converts_tags_to_array",
                "input": {
                    "path1": "future-work",
                    "path2": "test-item",
                    "title": "Test item",
                    "content": "Test content",
                    "tags": "machine-learning,voting,algorithms",
                    "estimated-scope": "medium",
                    "rfc": "required",
                },
                "expected_result": {
                    "tags": ["machine-learning", "voting", "algorithms"]
                },
            },
            {
                "name": "handles_rfc_url",
                "input": {
                    "path1": "future-work",
                    "path2": "test-item",
                    "title": "Test item with RFC URL",
                    "content": "Test content",
                    "estimated-scope": "medium",
                    "rfc": "https://github.com/thousandbrainsproject/tbp.monty/pull/1223",
                },
                "expected_result": {
                    "rfc": "https://github.com/thousandbrainsproject/tbp.monty/pull/1223"
                },
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                result_data = self._run_build_test([case["input"]])
                self.assertEqual(len(result_data), 1)
                item = result_data[0]

                for key, expected_value in case["expected_result"].items():
                    self.assertEqual(item[key], expected_value)

    def test_validation_failures(self):
        """Test various validation failure scenarios."""
        # Create validation files for field validation tests
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir()

        # Create snippet files for fields that need validation
        scope_file = snippets_dir / "future-work-estimated-scope.md"
        with open(scope_file, "w", encoding="utf-8") as f:
            f.write("`small` `medium` `large` `unknown`")

        rfc_file = snippets_dir / "future-work-rfc.md"
        with open(rfc_file, "w", encoding="utf-8") as f:
            f.write("`required` `optional` `not-required`")

        status_file = snippets_dir / "future-work-status.md"
        with open(status_file, "w", encoding="utf-8") as f:
            f.write("`completed` `in-progress`")

        failure_cases = [
            {
                "name": "invalid_estimated_scope",
                "input": [
                    {
                        "path1": "future-work",
                        "path2": "test-item",
                        "title": "Test item",
                        "content": "Test content",
                        "estimated-scope": "huge",
                        "rfc": "required",
                    }
                ],
                "error_fragment": "Validation failed",
                "needs_snippets": True,
            },
            {
                "name": "invalid_rfc_value",
                "input": [
                    {
                        "path1": "future-work",
                        "path2": "test-item",
                        "title": "Test item",
                        "content": "Test content",
                        "estimated-scope": "medium",
                        "rfc": "invalid-value",
                    }
                ],
                "error_fragment": "Validation failed",
                "needs_snippets": True,
            },
            {
                "name": "invalid_status",
                "input": [
                    {
                        "path1": "future-work",
                        "path2": "test-item",
                        "title": "Test item",
                        "content": "Test content",
                        "estimated-scope": "medium",
                        "rfc": "required",
                        "status": "pending",
                    }
                ],
                "error_fragment": "Validation failed",
                "needs_snippets": True,
            },
            {
                "name": "missing_required_fields",
                "input": [
                    {
                        "path1": "future-work",
                        "path2": "test-item",
                        "title": "Test item missing required fields",
                        "content": "Test content",
                    }
                ],
                "error_fragment": "Validation failed",
                "needs_snippets": False,
            },
            {
                "name": "empty_required_fields",
                "input": [
                    {
                        "path1": "future-work",
                        "path2": "test-item",
                        "title": "Test item with empty required fields",
                        "content": "Test content",
                        "estimated-scope": "",
                        "rfc": "  ",
                    }
                ],
                "error_fragment": "Validation failed",
                "needs_snippets": False,
            },
        ]

        for case in failure_cases:
            with self.subTest(case=case["name"]):
                if case["needs_snippets"]:
                    self._expect_build_failure_with_snippets(
                        case["input"], case["error_fragment"], str(snippets_dir)
                    )
                else:
                    self._expect_build_failure(case["input"], case["error_fragment"])

    def _expect_build_failure_with_snippets(
        self,
        input_data: List[Dict[str, Any]],
        expected_error_fragment: str,
        snippets_dir: str,
    ):
        """Helper method to test build fails with expected error and snippets."""
        index_file = self.temp_path / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        with self.assertRaises(ValueError) as context:
            build(str(index_file), str(self.output_path), snippets_dir)

        self.assertIn(expected_error_fragment, str(context.exception))

    def test_comma_separated_field_limits(self):
        """Test comma-separated field limits."""
        max_items = RecordValidator.MAX_COMMA_SEPARATED_ITEMS

        too_many_tags = ",".join([f"tag{i}" for i in range(max_items + 1)])
        self._expect_build_failure([{
            "path1": "future-work",
            "path2": "test-item",
            "title": "Test item with too many tags",
            "content": "Test content",
            "tags": too_many_tags,
            "estimated-scope": "medium",
            "rfc": "required"
        }])

        exactly_max_tags = ",".join([f"tag{i}" for i in range(max_items)])
        result_data = self._run_build_test([{
            "path1": "future-work",
            "path2": "test-item",
            "title": "Test item with exactly max tags",
            "content": "Test content",
            "tags": exactly_max_tags,
            "estimated-scope": "medium",
            "rfc": "required"
        }])

        self.assertEqual(len(result_data), 1)
        self.assertEqual(len(result_data[0]["tags"]), max_items)

    def test_validation_files_loading(self):
        """Test loading validation files from docs/snippets directory."""
        # Create temporary validation files
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir()

        # Create future-work-tags.md
        tags_file = snippets_dir / "future-work-tags.md"
        with open(tags_file, "w", encoding="utf-8") as f:
            f.write("`accuracy` `pose` `learning` `multiobj`")

        # Create future-work-skills.md
        skills_file = snippets_dir / "future-work-skills.md"
        with open(skills_file, "w", encoding="utf-8") as f:
            f.write("`python` `github-actions` `JS` `HTML`")

        validator = RecordValidator(str(snippets_dir))

        # Check that validation sets were loaded
        self.assertIn("tags", validator.validation_sets)
        self.assertIn("skills", validator.validation_sets)

        # Check that validation sets contain regex patterns
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

    def test_tag_validation_success(self):
        """Test successful tag validation against loaded validation files."""
        # Create temporary validation files
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir()

        tags_file = snippets_dir / "future-work-tags.md"
        with open(tags_file, "w", encoding="utf-8") as f:
            f.write("`accuracy` `pose` `learning` `multiobj`")

        input_data = [
            {
                "path1": "future-work",
                "path2": "test-item",
                "title": "Test item with valid tags",
                "content": "Test content",
                "tags": "accuracy,learning",
                "estimated-scope": "medium",
                "rfc": "required",
            }
        ]

        index_file = self.temp_path / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        # Should not raise any exceptions
        build(str(index_file), str(self.output_path), str(snippets_dir))

        # Verify the data was processed correctly
        data_file = self.output_path / "data.json"
        with open(data_file, "r", encoding="utf-8") as f:
            result_data = json.load(f)

        self.assertEqual(len(result_data), 1)
        self.assertEqual(result_data[0]["tags"], ["accuracy", "learning"])

    def test_tag_validation_failure(self):
        """Test tag validation failure with invalid tags."""
        # Create temporary validation files
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir()

        tags_file = snippets_dir / "future-work-tags.md"
        with open(tags_file, "w", encoding="utf-8") as f:
            f.write("`accuracy` `pose` `learning` `multiobj`")

        input_data = [
            {
                "path1": "future-work",
                "path2": "test-item",
                "title": "Test item with invalid tags",
                "content": "Test content",
                "tags": "accuracy,invalid-tag,learning",
                "estimated-scope": "medium",
                "rfc": "required",
            }
        ]

        index_file = self.temp_path / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        with self.assertRaises(ValueError) as context:
            build(str(index_file), str(self.output_path), str(snippets_dir))

        error_message = str(context.exception)
        self.assertIn("Invalid tags value 'invalid-tag'", error_message)
        self.assertIn("accuracy, learning, multiobj, pose", error_message)

    def test_skills_validation(self):
        """Test skills field validation and comma-separated handling."""
        # Create temporary validation files
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir()

        skills_file = snippets_dir / "future-work-skills.md"
        with open(skills_file, "w", encoding="utf-8") as f:
            f.write("`python` `github-actions` `JS` `HTML` `CSS`")

        # Test valid skills
        input_data = [
            {
                "path1": "future-work",
                "path2": "test-item",
                "title": "Test item with valid skills",
                "content": "Test content",
                "skills": "python,JS,HTML",
                "estimated-scope": "medium",
                "rfc": "required",
            }
        ]

        index_file = self.temp_path / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        build(str(index_file), str(self.output_path), str(snippets_dir))

        # Verify the data was processed correctly
        data_file = self.output_path / "data.json"
        with open(data_file, "r", encoding="utf-8") as f:
            result_data = json.load(f)

        self.assertEqual(len(result_data), 1)
        self.assertEqual(result_data[0]["skills"], ["python", "JS", "HTML"])

        # Test invalid skills
        input_data[0]["skills"] = "python,invalid-skill"

        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        with self.assertRaises(ValueError) as context:
            build(str(index_file), str(self.output_path), str(snippets_dir))

        error_message = str(context.exception)
        self.assertIn("Invalid skills value 'invalid-skill'", error_message)

    def test_missing_validation_files_graceful(self):
        """Test graceful handling of missing validation files."""
        # Create snippets directory but no validation files
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir()

        input_data = [
            {
                "path1": "future-work",
                "path2": "test-item",
                "title": "Test item without validation files",
                "content": "Test content",
                "tags": "any-tag",
                "skills": "any-skill",
                "estimated-scope": "medium",
                "rfc": "required",
            }
        ]

        index_file = self.temp_path / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        # Should not raise exceptions - validation should be skipped
        build(str(index_file), str(self.output_path), str(snippets_dir))

        # Verify the data was processed
        data_file = self.output_path / "data.json"
        with open(data_file, "r", encoding="utf-8") as f:
            result_data = json.load(f)

        self.assertEqual(len(result_data), 1)
        self.assertEqual(result_data[0]["tags"], ["any-tag"])
        self.assertEqual(result_data[0]["skills"], ["any-skill"])

    def test_nonexistent_snippets_directory(self):
        """Test handling of nonexistent snippets directory."""
        nonexistent_dir = self.temp_path / "nonexistent"

        input_data = [
            {
                "path1": "future-work",
                "path2": "test-item",
                "title": "Test item with nonexistent snippets dir",
                "content": "Test content",
                "estimated-scope": "medium",
                "rfc": "required",
            }
        ]

        index_file = self.temp_path / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        # Should not raise exceptions
        build(str(index_file), str(self.output_path), str(nonexistent_dir))

        # Verify the data was processed
        data_file = self.output_path / "data.json"
        with open(data_file, "r", encoding="utf-8") as f:
            result_data = json.load(f)

        self.assertEqual(len(result_data), 1)

    def test_build_without_snippets_dir(self):
        """Test build function without docs_snippets_dir parameter."""
        input_data = [
            {
                "path1": "future-work",
                "path2": "test-item",
                "title": "Test item without snippets validation",
                "content": "Test content",
                "estimated-scope": "medium",
                "rfc": "required",
            }
        ]

        # This should work the same as before (no validation files)
        result_data = self._run_build_test(input_data)
        self.assertEqual(len(result_data), 1)

    def test_estimated_scope_validation(self):
        """Test estimated-scope validation against snippet file."""
        # Create temporary validation files
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir()

        scope_file = snippets_dir / "future-work-estimated-scope.md"
        with open(scope_file, "w", encoding="utf-8") as f:
            f.write("`small` `medium` `large` `unknown`")

        # Test valid estimated-scope
        input_data = [
            {
                "path1": "future-work",
                "path2": "test-item",
                "title": "Test item with valid scope",
                "content": "Test content",
                "estimated-scope": "medium",
                "rfc": "required",
            }
        ]

        index_file = self.temp_path / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        # Should not raise any exceptions
        build(str(index_file), str(self.output_path), str(snippets_dir))

        # Test invalid estimated-scope
        input_data[0]["estimated-scope"] = "huge"

        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        with self.assertRaises(ValueError) as context:
            build(str(index_file), str(self.output_path), str(snippets_dir))

        error_message = str(context.exception)
        self.assertIn(
            "Invalid estimated-scope value",
            error_message,
        )
        self.assertIn("large, medium, small, unknown", error_message)
        self.assertIn("huge", error_message)

    def test_status_validation(self):
        """Test status validation against snippet file."""
        # Create temporary validation files
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir()

        status_file = snippets_dir / "future-work-status.md"
        with open(status_file, "w", encoding="utf-8") as f:
            f.write("`completed` `in-progress`")

        # Test valid status
        input_data = [
            {
                "path1": "future-work",
                "path2": "test-item",
                "title": "Test item with valid status",
                "content": "Test content",
                "estimated-scope": "medium",
                "rfc": "required",
                "status": "completed",
            }
        ]

        index_file = self.temp_path / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        # Should not raise any exceptions
        build(str(index_file), str(self.output_path), str(snippets_dir))

        # Test invalid status
        input_data[0]["status"] = "pending"

        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        with self.assertRaises(ValueError) as context:
            build(str(index_file), str(self.output_path), str(snippets_dir))

        error_message = str(context.exception)
        self.assertIn(
            "Invalid status value",
            error_message,
        )
        self.assertIn("completed, in-progress", error_message)
        self.assertIn("pending", error_message)

    def test_rfc_validation_with_regex(self):
        """Test RFC validation using regex patterns for both simple values and URLs."""
        # Create temporary validation files
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir()

        rfc_file = snippets_dir / "future-work-rfc.md"
        with open(rfc_file, "w", encoding="utf-8") as f:
            f.write(
                "`required` `optional` `not-required` `https://github\\.com/thousandbrainsproject/tbp\\.monty/pull/\\d+`"
            )

        # Test valid predefined values
        input_data = [
            {
                "path1": "future-work",
                "path2": "test-item",
                "title": "Test item with valid RFC",
                "content": "Test content",
                "estimated-scope": "medium",
                "rfc": "required",
            }
        ]

        index_file = self.temp_path / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        # Should not raise any exceptions
        build(str(index_file), str(self.output_path), str(snippets_dir))

        # Test valid URL
        input_data[0]["rfc"] = (
            "https://github.com/thousandbrainsproject/tbp.monty/pull/123"
        )

        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        # Should not raise any exceptions
        build(str(index_file), str(self.output_path), str(snippets_dir))

        # Test invalid RFC value
        input_data[0]["rfc"] = "invalid-rfc"

        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        with self.assertRaises(ValueError) as context:
            build(str(index_file), str(self.output_path), str(snippets_dir))

        error_message = str(context.exception)
        self.assertIn("rfc must be one of:", error_message)
        self.assertIn("not-required, optional, required, valid RFC URL", error_message)

    def test_regex_validation_for_tags(self):
        """Test that tags field also supports regex patterns."""
        # Create temporary validation files with regex pattern
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir()

        tags_file = snippets_dir / "future-work-tags.md"
        with open(tags_file, "w", encoding="utf-8") as f:
            # Mix simple words and regex patterns
            f.write("`accuracy` `pose` `learning.*` `test-\\d+`")

        # Test valid simple tag
        input_data = [
            {
                "path1": "future-work",
                "path2": "test-item",
                "title": "Test item with valid tags",
                "content": "Test content",
                "tags": "accuracy,pose",
                "estimated-scope": "medium",
                "rfc": "required",
            }
        ]

        index_file = self.temp_path / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        # Should not raise any exceptions
        build(str(index_file), str(self.output_path), str(snippets_dir))

        # Test valid regex pattern match
        input_data[0]["tags"] = "learning-module,test-123"

        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        # Should not raise any exceptions
        build(str(index_file), str(self.output_path), str(snippets_dir))

        # Test invalid tag that doesn't match any pattern
        input_data[0]["tags"] = "invalid-tag"

        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        with self.assertRaises(ValueError) as context:
            build(str(index_file), str(self.output_path), str(snippets_dir))

        error_message = str(context.exception)
        self.assertIn("Invalid tags value 'invalid-tag'", error_message)

    def test_word_boundary_wrapping(self):
        """Test that simple alphanumeric words are wrapped with word boundaries."""
        # Create temporary validation files
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir()

        tags_file = snippets_dir / "future-work-tags.md"
        with open(tags_file, "w", encoding="utf-8") as f:
            f.write("`test` `multi-word`")

        # Test that "test" matches exactly due to word boundaries
        input_data = [
            {
                "path1": "future-work",
                "path2": "test-item",
                "title": "Test word boundary validation",
                "content": "Test content",
                "tags": "test",
                "estimated-scope": "medium",
                "rfc": "required",
            }
        ]

        index_file = self.temp_path / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        # Should not raise any exceptions
        build(str(index_file), str(self.output_path), str(snippets_dir))

        # Test that partial matches are rejected due to word boundaries
        input_data[0]["tags"] = "testing"  # Should not match "test"

        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        with self.assertRaises(ValueError) as context:
            build(str(index_file), str(self.output_path), str(snippets_dir))

        error_message = str(context.exception)
        self.assertIn("Invalid tags value 'testing'", error_message)


if __name__ == "__main__":
    unittest.main()
