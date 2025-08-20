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
                    "rfc": "required"
                },
                "expected_result": {
                    "tags": ["machine-learning", "voting", "algorithms"]
                }
            },
            {
                "name": "handles_rfc_url",
                "input": {
                    "path1": "future-work",
                    "path2": "test-item",
                    "title": "Test item with RFC URL",
                    "content": "Test content",
                    "estimated-scope": "medium",
                    "rfc": "https://github.com/thousandbrainsproject/pull/1223"
                },
                "expected_result": {
                    "rfc": "https://github.com/thousandbrainsproject/pull/1223"
                }
            }
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
        failure_cases = [
            {
                "name": "non_dict_item",
                "input": ["not_a_dict"],
                "error_fragment": "Validation failed"
            },
            {
                "name": "invalid_estimated_scope",
                "input": [{
                    "path1": "future-work",
                    "path2": "test-item",
                    "title": "Test item",
                    "content": "Test content",
                    "estimated-scope": "huge",
                    "rfc": "required"
                }],
                "error_fragment": "Validation failed"
            },
            {
                "name": "invalid_rfc_value",
                "input": [{
                    "path1": "future-work",
                    "path2": "test-item",
                    "title": "Test item",
                    "content": "Test content",
                    "estimated-scope": "medium",
                    "rfc": "invalid-value"
                }],
                "error_fragment": "Validation failed"
            },
            {
                "name": "invalid_status",
                "input": [{
                    "path1": "future-work",
                    "path2": "test-item",
                    "title": "Test item",
                    "content": "Test content",
                    "estimated-scope": "medium",
                    "rfc": "required",
                    "status": "pending"
                }],
                "error_fragment": "Validation failed"
            },
            {
                "name": "missing_required_fields",
                "input": [{
                    "path1": "future-work",
                    "path2": "test-item",
                    "title": "Test item missing required fields",
                    "content": "Test content"
                }],
                "error_fragment": "Validation failed"
            },
            {
                "name": "empty_required_fields",
                "input": [{
                    "path1": "future-work",
                    "path2": "test-item",
                    "title": "Test item with empty required fields",
                    "content": "Test content",
                    "estimated-scope": "",
                    "rfc": "  "
                }],
                "error_fragment": "Validation failed"
            }
        ]

        for case in failure_cases:
            with self.subTest(case=case["name"]):
                self._expect_build_failure(case["input"], case["error_fragment"])

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


if __name__ == "__main__":
    unittest.main()
