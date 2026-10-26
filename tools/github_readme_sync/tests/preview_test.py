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
from unittest.mock import MagicMock, call, patch

from tools.github_readme_sync.preview import (
    SCHEMA_VERSION,
    load_preview_bundle,
    publish_preview,
    render_preview,
    validate_preview_bundle,
)
from tools.github_readme_sync.readme import ReadMe

REPOSITORY = "thousandbrainsproject/tbp.monty"
PR_NUMBER = 1111
HEAD_SHA = "a" * 40
BASE_REF = "main"


def make_document(**overrides):
    document = {
        "title": "Page",
        "slug": "page",
        "body": "Rendered body",
        "hidden": False,
        "children": [],
    }
    document.update(overrides)
    return document


def make_bundle():
    return {
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
                "children": [make_document()],
            }
        ],
    }


class TestRenderPreview(unittest.TestCase):
    @patch("tools.github_readme_sync.preview.validate_preview_bundle")
    @patch.object(ReadMe, "process_markdown")
    @patch("tools.github_readme_sync.preview.load_doc")
    @patch("tools.github_readme_sync.preview.check_hierarchy_file")
    def test_render_preview_returns_none_and_writes_bundle_and_calls_dependencies(
        self,
        mock_check_hierarchy_file,
        mock_load_doc,
        mock_process_markdown,
        mock_validate_preview_bundle,
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
            output_file = Path(temp_dir) / "artifacts" / "preview.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            render_preview(
                folder=folder,
                output_file=output_file,
                repository=REPOSITORY,
                pr_number=PR_NUMBER,
                head_sha=HEAD_SHA,
                base_ref=BASE_REF,
            )

            with output_file.open(encoding="utf-8") as file:
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
        mock_validate_preview_bundle.assert_called_once_with(
            bundle=expected_bundle,
            expected_repository=REPOSITORY,
            expected_pr_number=PR_NUMBER,
            expected_head_sha=HEAD_SHA,
            expected_base_ref=BASE_REF,
        )


class TestPublishPreview(unittest.TestCase):
    def setUp(self):
        self.bundle = make_bundle()
        self.bundle_file = Path("bundle.json")
        self.rdme = MagicMock(spec=ReadMe)

    @patch("tools.github_readme_sync.preview.upload_rendered")
    @patch("tools.github_readme_sync.preview.validate_preview_bundle")
    @patch("tools.github_readme_sync.preview.load_preview_bundle")
    def test_publish_preview_loads_bundle_and_uploads_validated_rendered_documents(
        self,
        mock_load_preview_bundle,
        mock_validate_preview_bundle,
        mock_upload_rendered,
    ):
        self.rdme.version_has_suffix.return_value = True
        mock_load_preview_bundle.return_value = self.bundle

        publish_preview(
            bundle_file=self.bundle_file,
            rdme=self.rdme,
            expected_repository=REPOSITORY,
            expected_pr_number=PR_NUMBER,
            expected_head_sha=HEAD_SHA,
            expected_base_ref=BASE_REF,
        )

        self.rdme.version_has_suffix.assert_called_once_with()
        mock_load_preview_bundle.assert_called_once_with(self.bundle_file)
        mock_validate_preview_bundle.assert_called_once_with(
            bundle=self.bundle,
            expected_repository=REPOSITORY,
            expected_pr_number=PR_NUMBER,
            expected_head_sha=HEAD_SHA,
            expected_base_ref=BASE_REF,
        )
        mock_upload_rendered.assert_called_once_with(
            self.bundle["documents"],
            self.rdme,
        )

    @patch("tools.github_readme_sync.preview.upload_rendered")
    @patch("tools.github_readme_sync.preview.validate_preview_bundle")
    @patch("tools.github_readme_sync.preview.load_preview_bundle")
    def test_publish_preview_raises_value_error_when_version_is_stable(
        self,
        mock_load_preview_bundle,
        mock_validate_preview_bundle,
        mock_upload_rendered,
    ):
        self.rdme.version_has_suffix.return_value = False

        with self.assertRaises(ValueError) as error:
            publish_preview(
                bundle_file=self.bundle_file,
                rdme=self.rdme,
                expected_repository=REPOSITORY,
                expected_pr_number=PR_NUMBER,
                expected_head_sha=HEAD_SHA,
                expected_base_ref=BASE_REF,
            )

        self.assertEqual(
            str(error.exception),
            "publish-preview requires a non-stable preview version",
        )
        self.rdme.version_has_suffix.assert_called_once_with()
        mock_load_preview_bundle.assert_not_called()
        mock_validate_preview_bundle.assert_not_called()
        mock_upload_rendered.assert_not_called()

    @patch("tools.github_readme_sync.preview.upload_rendered")
    @patch("tools.github_readme_sync.preview.validate_preview_bundle")
    @patch("tools.github_readme_sync.preview.load_preview_bundle")
    def test_publish_preview_raises_value_error_when_bundle_validation_fails(
        self,
        mock_load_preview_bundle,
        mock_validate_preview_bundle,
        mock_upload_rendered,
    ):
        self.rdme.version_has_suffix.return_value = True
        mock_load_preview_bundle.return_value = self.bundle
        mock_validate_preview_bundle.side_effect = ValueError("invalid preview bundle")

        with self.assertRaises(ValueError) as error:
            publish_preview(
                bundle_file=self.bundle_file,
                rdme=self.rdme,
                expected_repository=REPOSITORY,
                expected_pr_number=PR_NUMBER,
                expected_head_sha=HEAD_SHA,
                expected_base_ref=BASE_REF,
            )

        self.assertEqual(str(error.exception), "invalid preview bundle")
        self.rdme.version_has_suffix.assert_called_once_with()
        mock_load_preview_bundle.assert_called_once_with(self.bundle_file)
        mock_validate_preview_bundle.assert_called_once_with(
            bundle=self.bundle,
            expected_repository=REPOSITORY,
            expected_pr_number=PR_NUMBER,
            expected_head_sha=HEAD_SHA,
            expected_base_ref=BASE_REF,
        )
        mock_upload_rendered.assert_not_called()


class TestLoadPreviewBundle(unittest.TestCase):
    def test_load_preview_bundle_returns_parsed_bundle(self):
        bundle = make_bundle()

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_path = Path(temp_dir) / "bundle.json"
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

            result = load_preview_bundle(bundle_path)

        self.assertEqual(result, bundle)

    def test_load_preview_bundle_raises_value_error_when_json_is_invalid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_path = Path(temp_dir) / "bundle.json"
            bundle_path.write_text("{not json", encoding="utf-8")

            with self.assertRaises(ValueError) as error:
                load_preview_bundle(bundle_path)

        self.assertEqual(
            str(error.exception),
            "Preview bundle is not valid JSON",
        )

    def test_load_preview_bundle_raises_type_error_when_json_is_not_object(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_path = Path(temp_dir) / "bundle.json"
            bundle_path.write_text("[]", encoding="utf-8")

            with self.assertRaises(TypeError) as error:
                load_preview_bundle(bundle_path)

        self.assertEqual(
            str(error.exception),
            "Preview bundle must be a JSON object",
        )


class TestValidatePreviewBundle(unittest.TestCase):
    def assert_invalid(self, bundle, error_type, expected_message):
        with self.assertRaises(error_type) as error:
            validate_preview_bundle(
                bundle=bundle,
                expected_repository=REPOSITORY,
                expected_pr_number=PR_NUMBER,
                expected_head_sha=HEAD_SHA,
                expected_base_ref=BASE_REF,
            )

        self.assertEqual(str(error.exception), expected_message)

    def test_validate_preview_bundle_raises_value_error_when_top_key_missing(self):
        bundle = make_bundle()
        del bundle["repository"]

        self.assert_invalid(
            bundle,
            ValueError,
            "$ is missing required keys: ['repository']",
        )

    def test_validate_preview_bundle_raises_value_error_when_top_key_unknown(self):
        bundle = make_bundle()
        bundle["extra"] = "unexpected"

        self.assert_invalid(
            bundle,
            ValueError,
            "$ contains unknown keys: ['extra']",
        )

    def test_validate_preview_bundle_raises_type_error_when_schema_not_int(self):
        bundle = make_bundle()
        bundle["schema_version"] = True

        self.assert_invalid(
            bundle,
            TypeError,
            "$.schema_version must be an integer",
        )

    def test_validate_preview_bundle_raises_value_error_when_schema_unsupported(self):
        bundle = make_bundle()
        bundle["schema_version"] = SCHEMA_VERSION + 1

        self.assert_invalid(
            bundle,
            ValueError,
            f"$.schema_version must be {SCHEMA_VERSION}, got {SCHEMA_VERSION + 1!r}",
        )

    def test_validate_preview_bundle_raises_value_error_when_repository_mismatch(self):
        bundle = make_bundle()
        bundle["repository"] = "someone/other-repository"

        self.assert_invalid(
            bundle,
            ValueError,
            "$.repository does not match the current repository",
        )

    def test_validate_preview_bundle_raises_type_error_when_pr_number_nonpositive(self):
        bundle = make_bundle()
        bundle["pull_request"]["number"] = 0

        self.assert_invalid(
            bundle,
            TypeError,
            "$.pull_request.number must be a positive integer",
        )

    def test_validate_preview_bundle_raises_value_error_when_pr_number_mismatch(self):
        bundle = make_bundle()
        bundle["pull_request"]["number"] = PR_NUMBER + 1

        self.assert_invalid(
            bundle,
            ValueError,
            "$.pull_request.number does not match the expected pr_number",
        )

    def test_validate_preview_bundle_raises_value_error_when_head_sha_malformed(self):
        bundle = make_bundle()
        bundle["pull_request"]["head_sha"] = "a" * 39

        self.assert_invalid(
            bundle,
            ValueError,
            "$.pull_request.head_sha must be a 40-character SHA",
        )

    def test_validate_preview_bundle_raises_value_error_when_head_sha_mismatch(self):
        bundle = make_bundle()
        bundle["pull_request"]["head_sha"] = "b" * 40

        self.assert_invalid(
            bundle,
            ValueError,
            "$.pull_request.head_sha does not match the expected head_sha",
        )

    def test_validate_preview_bundle_raises_value_error_when_base_ref_mismatch(self):
        bundle = make_bundle()
        bundle["pull_request"]["base_ref"] = "release"

        self.assert_invalid(
            bundle,
            ValueError,
            "$.pull_request.base_ref does not match the expected base branch",
        )
