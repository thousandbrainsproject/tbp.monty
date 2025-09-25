# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from tools.future_work_widget.build import build
from tools.future_work_widget.validator import RecordValidator


class TestBuild(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.output_dir = tempfile.mkdtemp()
        self.output_path = Path(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        shutil.rmtree(self.output_dir)

    def _create_snippets(self, snippet_configs: dict[str, str]) -> str:
        """Create snippet files with given configurations.

        Args:
            snippet_configs: Dict mapping filename to content

        Returns:
            Path to snippets directory
        """
        snippets_dir = self.temp_path / "snippets"
        snippets_dir.mkdir(exist_ok=True)

        for filename, content in snippet_configs.items():
            snippet_file = snippets_dir / filename
            with open(snippet_file, "w", encoding="utf-8") as f:
                f.write(content)

        return str(snippets_dir)

    def _run_build_test_with_snippets(
        self, input_data: list[dict[str, Any]], snippets_dir: str
    ) -> list[dict[str, Any]]:
        """Helper method to run build with test data and snippets directory.

        Returns:
            The processed data from the output JSON file.

        Raises:
            ValueError: If the build fails with validation errors.
        """
        index_file = self.temp_path / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        result = build(str(index_file), str(self.output_path), snippets_dir)

        if not result["success"]:
            raise ValueError(result["error_message"])

        data_file = self.output_path / "data.json"
        with open(data_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _create_test_item(self, **overrides) -> dict[str, Any]:
        """Create a test item with default values, allowing field overrides.

        Args:
            **overrides: Fields to override in the default test item

        Returns:
            Dictionary representing a test item
        """
        base_item = {
            "path1": "future-work",
            "path2": "test-item",
            "title": "Test item",
            "content": "Test content",
            "estimated-scope": "medium",
            "rfc": "required",
        }
        base_item.update(overrides)
        return base_item

    def _run_build_test(
        self, input_data: list[dict[str, Any]], snippets_dir: str | None = None
    ) -> list[dict[str, Any]]:
        """Helper method to run build with test data and return results.

        Args:
            input_data: Test data to process
            snippets_dir: Optional snippets directory for validation

        Returns:
            The processed data from the output JSON file.

        Raises:
            ValueError: If the build fails with validation errors.
        """
        index_file = self.temp_path / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        if snippets_dir is None:
            snippets_dir = self._create_snippets({})

        result = build(str(index_file), str(self.output_path), snippets_dir)

        if not result["success"]:
            raise ValueError(result["error_message"])

        data_file = self.output_path / "data.json"
        with open(data_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _expect_build_failure(
        self,
        input_data: list[dict[str, Any]],
        expected_error_fragment: str = "Validation failed",
        snippets_dir: str | None = None,
    ):
        """Helper method to test that build fails with expected error."""
        index_file = self.temp_path / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        if snippets_dir is None:
            snippets_dir = self._create_snippets({})

        result = build(str(index_file), str(self.output_path), snippets_dir)

        self.assertFalse(result["success"])
        self.assertIn(expected_error_fragment, result["error_message"])

    def test_build_filters_future_work_items(self):
        input_data = [
            self._create_test_item(
                path2="voting-improvements",
                title="Improve voting mechanism",
                content="Test content for voting",
            ),
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

    def test_array_field_transformations(self):
        """Test validation and transformation of fields including arrays and URLs."""
        transformation_cases = [
            {
                "name": "tags_transformation",
                "snippet_file": "future-work-tags.md",
                "snippet_content": "`accuracy` `pose` `learning` `multiobj`",
                "field_name": "tags",
                "input_value": "accuracy,learning",
                "expected_result": ["accuracy", "learning"],
            },
            {
                "name": "rfc_url_passthrough",
                "field_name": "rfc",
                "input_value": "https://github.com/thousandbrainsproject/tbp.monty/pull/1223",
                "expected_result": "https://github.com/thousandbrainsproject/tbp.monty/pull/1223",
            },
        ]

        for case in transformation_cases:
            with self.subTest(case=case["name"]):
                test_item = self._create_test_item(
                    title=f"Test item with {case['field_name']}",
                    **{case["field_name"]: case["input_value"]},
                )

                if "snippet_file" in case:
                    snippets_dir = self._create_snippets(
                        {case["snippet_file"]: case["snippet_content"]}
                    )
                    result_data = self._run_build_test_with_snippets(
                        [test_item], snippets_dir
                    )
                else:
                    result_data = self._run_build_test([test_item])

                self.assertEqual(len(result_data), 1)
                self.assertEqual(
                    result_data[0][case["field_name"]], case["expected_result"]
                )

    def test_build_without_snippets_dir(self):
        """Test that build works without snippets directory for validation."""
        input_data = [
            self._create_test_item(title="Test item without snippets validation")
        ]

        result_data = self._run_build_test(input_data)
        self.assertEqual(len(result_data), 1)

    def test_json_output_success(self):
        """Test JSON output mode for successful build."""
        input_data = [
            self._create_test_item(path2="test-success", title="Test success item")
        ]

        index_file = self.temp_path / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        snippets_dir = self._create_snippets({})
        result = build(str(index_file), str(self.output_path), snippets_dir)

        self.assertIsNotNone(result)
        self.assertTrue(result["success"])
        self.assertEqual(result["processed_items"], 1)
        self.assertEqual(result["total_items"], 1)
        self.assertEqual(len(result["errors"]), 0)

    def test_json_output_validation_errors(self):
        """Test JSON output mode with validation errors."""
        snippets_dir = self._create_snippets(
            {"future-work-tags.md": "`accuracy` `pose` `learning`"}
        )

        input_data = [
            self._create_test_item(
                path="docs/future-work/test-item.md",
                path2="test-item",
                title="Test item with invalid tags",
                tags="invalid-tag,accuracy",
            )
        ]

        index_file = self.temp_path / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        result = build(str(index_file), str(self.output_path), snippets_dir)

        self.assertIsNotNone(result)
        self.assertFalse(result["success"])
        self.assertEqual(result["processed_items"], 1)
        self.assertEqual(result["total_items"], 1)
        self.assertEqual(len(result["errors"]), 1)

        error = result["errors"][0]
        self.assertIn("Invalid tags value 'invalid-tag'", error["message"])
        self.assertEqual(error["file"], "docs/future-work/test-item.md")
        self.assertEqual(error["line"], 1)
        self.assertEqual(error["field"], "tags")
        self.assertEqual(error["level"], "error")
        self.assertEqual(error["annotation_level"], "failure")
        self.assertIn("test-item.md", error["title"])

    def test_json_output_too_many_items_error(self):
        """Test JSON output mode with too many comma-separated items."""
        max_items = RecordValidator.MAX_COMMA_SEPARATED_ITEMS
        too_many_tags = ",".join([f"tag{i}" for i in range(max_items + 1)])

        input_data = [
            self._create_test_item(
                path="docs/future-work/test-limits.md",
                path2="test-limits",
                title="Test item with too many tags",
                tags=too_many_tags,
            )
        ]

        index_file = self.temp_path / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f)

        snippets_dir = self._create_snippets({})
        result = build(str(index_file), str(self.output_path), snippets_dir)

        self.assertIsNotNone(result)
        self.assertFalse(result["success"])
        self.assertEqual(result["processed_items"], 1)
        self.assertEqual(result["total_items"], 1)
        self.assertEqual(len(result["errors"]), 1)

        error = result["errors"][0]
        self.assertIn("tags field cannot have more than", error["message"])
        self.assertIn(str(max_items), error["message"])
        self.assertEqual(error["file"], "docs/future-work/test-limits.md")
        self.assertEqual(error["line"], 1)
        self.assertEqual(error["field"], "tags")
        self.assertEqual(error["level"], "error")
        self.assertEqual(error["annotation_level"], "failure")
        self.assertIn("test-limits.md", error["title"])

    def test_field_validation_scenarios(self):
        """Test validation for various fields with valid and invalid values."""
        validation_cases = [
            {
                "name": "estimated_scope_validation",
                "snippet_file": "future-work-estimated-scope.md",
                "snippet_content": "`small` `medium` `large` `unknown`",
                "field_name": "estimated-scope",
                "valid_value": "medium",
                "invalid_value": "huge",
                "expected_error_fragments": [
                    "Invalid estimated-scope value",
                    "huge",
                    "large, medium, small, unknown",
                ],
            },
            {
                "name": "status_validation",
                "snippet_file": "future-work-status.md",
                "snippet_content": "`completed` `in-progress`",
                "field_name": "status",
                "valid_value": "completed",
                "invalid_value": "pending",
                "expected_error_fragments": [
                    "Invalid status value",
                    "pending",
                    "completed, in-progress",
                ],
            },
            {
                "name": "tags_validation",
                "snippet_file": "future-work-tags.md",
                "snippet_content": "`accuracy` `pose` `learning` `multiobj`",
                "field_name": "tags",
                "valid_value": "accuracy,learning",
                "invalid_value": "accuracy,invalid-tag,learning",
                "expected_error_fragments": [
                    "Invalid tags value 'invalid-tag'",
                    "accuracy, learning, multiobj, pose",
                ],
            },
        ]

        for case in validation_cases:
            with self.subTest(case=case["name"]):
                snippets_dir = self._create_snippets(
                    {case["snippet_file"]: case["snippet_content"]}
                )

                valid_item = self._create_test_item(
                    **{case["field_name"]: case["valid_value"]}
                )

                index_file = self.temp_path / "index.json"
                with open(index_file, "w", encoding="utf-8") as f:
                    json.dump([valid_item], f)

                result = build(str(index_file), str(self.output_path), snippets_dir)
                self.assertTrue(result["success"])

                invalid_item = self._create_test_item(
                    **{case["field_name"]: case["invalid_value"]}
                )

                with open(index_file, "w", encoding="utf-8") as f:
                    json.dump([invalid_item], f)

                result = build(str(index_file), str(self.output_path), snippets_dir)
                self.assertFalse(result["success"])

                self.assertEqual(len(result["errors"]), 1)
                error_message = result["errors"][0]["message"]
                for fragment in case["expected_error_fragments"]:
                    self.assertIn(fragment, error_message)

    def test_rfc_validation_with_regex(self):
        """Test RFC validation with various valid values and regex patterns."""
        snippets_dir = self._create_snippets(
            {
                "future-work-rfc.md": "`required` `optional` `not-required` `https://github\\.com/thousandbrainsproject/tbp\\.monty/.*`"
            }
        )

        valid_rfc_values = [
            "required",
            "https://github.com/thousandbrainsproject/tbp.monty/pull/123",
            "https://github.com/thousandbrainsproject/tbp.monty/blob/main/rfcs/0015_future_work.md",
        ]

        for rfc_value in valid_rfc_values:
            with self.subTest(rfc_value=rfc_value):
                test_item = self._create_test_item(
                    title="Test item with valid RFC", rfc=rfc_value
                )
                result_data = self._run_build_test([test_item])
                self.assertEqual(len(result_data), 1)

        invalid_item = self._create_test_item(
            title="Test item with invalid RFC", rfc="invalid-rfc"
        )

        index_file = self.temp_path / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump([invalid_item], f)

        result = build(str(index_file), str(self.output_path), snippets_dir)
        self.assertFalse(result["success"])

        self.assertEqual(len(result["errors"]), 1)
        error_message = result["errors"][0]["message"]
        self.assertIn("Invalid rfc value 'invalid-rfc'", error_message)
        self.assertIn("Valid values are:", error_message)

    def test_regex_validation_for_tags(self):
        """Test regex pattern validation for tags field."""
        snippets_dir = self._create_snippets(
            {"future-work-tags.md": "`accuracy` `pose` `learning.*` `test-\\d+`"}
        )

        valid_tag_cases = [
            "accuracy,pose",  # Exact matches
            "learning-module,test-123",  # Regex matches: learning.* and test-\d+
        ]

        for tags in valid_tag_cases:
            with self.subTest(tags=tags):
                test_item = self._create_test_item(tags=tags)
                result_data = self._run_build_test_with_snippets(
                    [test_item], snippets_dir
                )
                self.assertEqual(len(result_data), 1)

        invalid_item = self._create_test_item(tags="invalid-tag")
        index_file = self.temp_path / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump([invalid_item], f)

        result = build(str(index_file), str(self.output_path), snippets_dir)
        self.assertFalse(result["success"])

        self.assertEqual(len(result["errors"]), 1)
        error_message = result["errors"][0]["message"]
        self.assertIn("Invalid tags value 'invalid-tag'", error_message)


if __name__ == "__main__":
    unittest.main()
