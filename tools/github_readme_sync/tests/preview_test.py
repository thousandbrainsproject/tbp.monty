# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from tools.github_readme_sync.preview import (
    SCHEMA_VERSION,
    render_preview,
)
from tools.github_readme_sync.readme import ReadMe

REPOSITORY = "thousandbrainsproject/tbp.monty"
PR_NUMBER = 1111
HEAD_SHA = "a" * 40
BASE_REF = "main"


class TestRenderPreview(unittest.TestCase):
    @patch.object(ReadMe, "process_markdown")
    @patch("tools.github_readme_sync.preview.load_doc")
    @patch("tools.github_readme_sync.preview.check_hierarchy_file")
    def test_render_preview_returns_none_and_writes_bundle_and_calls_dependencies(
        self,
        mock_check_hierarchy_file,
        mock_load_doc,
        mock_process_markdown,
    ):
        child_node = {"slug": "child", "children": []}
        page_node = {"slug": "page", "children": [child_node]}
        hierarchy = [
            {
                "title": "Category",
                "slug": "category",
                "children": [page_node],
            }
        ]

        mock_check_hierarchy_file.return_value = hierarchy
        mock_load_doc.side_effect = [
            {
                "title": "Page",
                "slug": "page",
                "body": "Source body",
                "hidden": False,
                "description": "Page description",
            },
            {
                "title": "Child",
                "slug": "child",
                "body": "Child source body",
                "hidden": True,
            },
        ]
        mock_process_markdown.side_effect = [
            "Rendered body",
            "Rendered child body",
        ]

        expected_bundle = {
            "schema_version": SCHEMA_VERSION,
            "repository": REPOSITORY,
            "pull_request": {
                "number": PR_NUMBER,
                "head_sha": HEAD_SHA,
                "base_ref": BASE_REF,
            },
            "documents": [
                {
                    "title": "Category",
                    "children": [
                        {
                            "title": "Page",
                            "slug": "page",
                            "body": "Rendered body",
                            "hidden": False,
                            "description": "Page description",
                            "children": [
                                {
                                    "title": "Child",
                                    "slug": "child",
                                    "body": "Rendered child body",
                                    "hidden": True,
                                    "children": [],
                                }
                            ],
                        }
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir) / "docs"
            output = Path(temp_dir) / "artifacts" / "preview.json"
            output.parent.mkdir(parents=True, exist_ok=True)

            render_preview(
                folder,
                output,
                repository=REPOSITORY,
                pr_number=PR_NUMBER,
                head_sha=HEAD_SHA,
                base_ref=BASE_REF,
            )

            with output.open(encoding="utf-8") as file:
                rendered = json.load(file)

        self.assertEqual(rendered, expected_bundle)
        mock_check_hierarchy_file.assert_called_once_with(str(folder))
        self.assertEqual(
            mock_load_doc.call_args_list,
            [
                call(str(folder), "category", page_node),
                call(str(folder), "category/page", child_node),
            ],
        )
        self.assertEqual(
            mock_process_markdown.call_args_list,
            [
                call(
                    "Source body",
                    str(folder / "category"),
                    "page",
                ),
                call(
                    "Child source body",
                    str(folder / "category" / "page"),
                    "child",
                ),
            ],
        )
