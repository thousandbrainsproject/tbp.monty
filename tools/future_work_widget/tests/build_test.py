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

from tools.future_work_widget.build import build


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
        expected_error_fragment: str = "Validation failed",
    ):
        """Helper method to test that build fails with expected error."""
        index_file = self.temp_path / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        with self.assertRaises(ValueError) as context:
            build(str(index_file), str(self.output_path))

        self.assertIn(expected_error_fragment, str(context.exception))

    def test_build_filters_future_work_items(self):
        input_data = [
            {
                "path1": "future-work",
                "path2": "voting-improvements",
                "title": "Improve voting mechanism",
                "content": "Test content for voting",
                "estimated-scope": "medium",
                "rfc": "required",
            },
            {
                "path1": "how-monty-works",
                "path2": "learning-modules",
                "title": "Learning modules overview",
                "content": "Test content for learning",
            },
        ]

        result_data = self._run_build_test(input_data)

        self.assertEqual(len(result_data), 1)
        future_work_titles = [item["title"] for item in result_data]
        self.assertIn("Improve voting mechanism", future_work_titles)
        self.assertNotIn("Learning modules overview", future_work_titles)

        for item in result_data:
            self.assertEqual(item["path1"], "future-work")

    def test_data_transformations(self):
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
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir()

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

    def test_tag_validation_success(self):
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

        build(str(index_file), str(self.output_path), str(snippets_dir))

        data_file = self.output_path / "data.json"
        with open(data_file, "r", encoding="utf-8") as f:
            result_data = json.load(f)

        self.assertEqual(len(result_data), 1)
        self.assertEqual(result_data[0]["tags"], ["accuracy", "learning"])

    def test_tag_validation_failure(self):
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
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir()

        skills_file = snippets_dir / "future-work-skills.md"
        with open(skills_file, "w", encoding="utf-8") as f:
            f.write("`python` `github-actions` `JS` `HTML` `CSS`")

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

        data_file = self.output_path / "data.json"
        with open(data_file, "r", encoding="utf-8") as f:
            result_data = json.load(f)

        self.assertEqual(len(result_data), 1)
        self.assertEqual(result_data[0]["skills"], ["python", "JS", "HTML"])

        input_data[0]["skills"] = "python,invalid-skill"

        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        with self.assertRaises(ValueError) as context:
            build(str(index_file), str(self.output_path), str(snippets_dir))

        error_message = str(context.exception)
        self.assertIn("Invalid skills value 'invalid-skill'", error_message)

    def test_build_without_snippets_dir(self):
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

        result_data = self._run_build_test(input_data)
        self.assertEqual(len(result_data), 1)

    def test_estimated_scope_validation(self):
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir()

        scope_file = snippets_dir / "future-work-estimated-scope.md"
        with open(scope_file, "w", encoding="utf-8") as f:
            f.write("`small` `medium` `large` `unknown`")

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

        build(str(index_file), str(self.output_path), str(snippets_dir))

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
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir()

        status_file = snippets_dir / "future-work-status.md"
        with open(status_file, "w", encoding="utf-8") as f:
            f.write("`completed` `in-progress`")

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

        build(str(index_file), str(self.output_path), str(snippets_dir))

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
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir()

        rfc_file = snippets_dir / "future-work-rfc.md"
        with open(rfc_file, "w", encoding="utf-8") as f:
            f.write(
                "`required` `optional` `not-required` `https://github\\.com/thousandbrainsproject/tbp\\.monty/pull/\\d+`"
            )

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

        build(str(index_file), str(self.output_path), str(snippets_dir))

        input_data[0]["rfc"] = (
            "https://github.com/thousandbrainsproject/tbp.monty/pull/123"
        )

        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        build(str(index_file), str(self.output_path), str(snippets_dir))

        input_data[0]["rfc"] = "invalid-rfc"

        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        with self.assertRaises(ValueError) as context:
            build(str(index_file), str(self.output_path), str(snippets_dir))

        error_message = str(context.exception)
        self.assertIn("rfc must be one of:", error_message)
        self.assertIn("not-required, optional, required, valid RFC URL", error_message)

    def test_regex_validation_for_tags(self):
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir()

        tags_file = snippets_dir / "future-work-tags.md"
        with open(tags_file, "w", encoding="utf-8") as f:
            f.write("`accuracy` `pose` `learning.*` `test-\\d+`")

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

        build(str(index_file), str(self.output_path), str(snippets_dir))

        input_data[0]["tags"] = "learning-module,test-123"

        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        build(str(index_file), str(self.output_path), str(snippets_dir))

        input_data[0]["tags"] = "invalid-tag"

        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        with self.assertRaises(ValueError) as context:
            build(str(index_file), str(self.output_path), str(snippets_dir))

        error_message = str(context.exception)
        self.assertIn("Invalid tags value 'invalid-tag'", error_message)


if __name__ == "__main__":
    unittest.main()
