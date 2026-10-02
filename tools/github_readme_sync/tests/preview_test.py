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
PR_NUMBER = 1103
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
            folder = str(Path(temp_dir) / "docs")
            output = Path(temp_dir) / "artifacts" / "preview.json"

            result = render_preview(
                folder,
                str(output),
                repository=REPOSITORY,
                pr_number=PR_NUMBER,
                head_sha=HEAD_SHA,
                base_ref=BASE_REF,
            )

            rendered = json.loads(output.read_text(encoding="utf-8"))
            output_text = output.read_text(encoding="utf-8")

        self.assertIsNone(result)
        self.assertEqual(rendered, expected_bundle)
        self.assertTrue(output_text.endswith("\n"))
        mock_check_hierarchy_file.assert_called_once_with(folder)
        self.assertEqual(
            mock_load_doc.call_args_list,
            [
                call(folder, "category", page_node),
                call(folder, "category/page", child_node),
            ],
        )
        self.assertEqual(
            mock_process_markdown.call_args_list,
            [
                call(
                    "Source body",
                    str(Path(folder) / "category"),
                    "page",
                ),
                call(
                    "Child source body",
                    str(Path(folder) / "category" / "page"),
                    "child",
                ),
            ],
        )
        mock_validate_preview_bundle.assert_called_once_with(
            expected_bundle,
            expected_repository=REPOSITORY,
            expected_pr_number=PR_NUMBER,
            expected_head_sha=HEAD_SHA,
            expected_base_ref=BASE_REF,
        )


class TestPublishPreview(unittest.TestCase):
    def setUp(self):
        self.bundle = make_bundle()
        self.rdme = MagicMock(spec=ReadMe)

    @patch("tools.github_readme_sync.preview.upload_rendered")
    @patch("tools.github_readme_sync.preview.validate_preview_bundle")
    @patch("tools.github_readme_sync.preview.load_preview_bundle")
    def test_publish_preview_returns_none_and_uploads_validated_documents(
        self,
        mock_load_preview_bundle,
        mock_validate_preview_bundle,
        mock_upload_rendered,
    ):
        self.rdme.version_has_suffix.return_value = True
        mock_load_preview_bundle.return_value = self.bundle

        result = publish_preview(
            "bundle.json",
            self.rdme,
            expected_repository=REPOSITORY,
            expected_pr_number=PR_NUMBER,
            expected_head_sha=HEAD_SHA,
            expected_base_ref=BASE_REF,
        )

        self.assertIsNone(result)
        self.rdme.version_has_suffix.assert_called_once_with()
        mock_load_preview_bundle.assert_called_once_with("bundle.json")
        mock_validate_preview_bundle.assert_called_once_with(
            self.bundle,
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
                "bundle.json",
                self.rdme,
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
                "bundle.json",
                self.rdme,
                expected_repository=REPOSITORY,
                expected_pr_number=PR_NUMBER,
                expected_head_sha=HEAD_SHA,
                expected_base_ref=BASE_REF,
            )

        self.assertEqual(str(error.exception), "invalid preview bundle")
        self.rdme.version_has_suffix.assert_called_once_with()
        mock_load_preview_bundle.assert_called_once_with("bundle.json")
        mock_validate_preview_bundle.assert_called_once_with(
            self.bundle,
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

            result = load_preview_bundle(str(bundle_path))

        self.assertEqual(result, bundle)

    def test_load_preview_bundle_raises_value_error_when_json_is_invalid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_path = Path(temp_dir) / "bundle.json"
            bundle_path.write_text("{not json", encoding="utf-8")

            with self.assertRaises(ValueError) as error:
                load_preview_bundle(str(bundle_path))

        self.assertEqual(
            str(error.exception),
            "Preview bundle is not valid JSON",
        )

    def test_load_preview_bundle_raises_type_error_when_json_is_not_object(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_path = Path(temp_dir) / "bundle.json"
            bundle_path.write_text("[]", encoding="utf-8")

            with self.assertRaises(TypeError) as error:
                load_preview_bundle(str(bundle_path))

        self.assertEqual(
            str(error.exception),
            "Preview bundle must be a JSON object",
        )

    @patch("tools.github_readme_sync.preview.MAX_BUNDLE_BYTES", 10)
    def test_load_preview_bundle_raises_value_error_when_bundle_too_large(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_path = Path(temp_dir) / "bundle.json"
            bundle_path.write_text("12345678901", encoding="utf-8")

            with self.assertRaises(ValueError) as error:
                load_preview_bundle(str(bundle_path))

        self.assertEqual(
            str(error.exception),
            "Preview bundle is 11 bytes; maximum is 10 bytes",
        )


class TestValidatePreviewBundle(unittest.TestCase):
    def validate(self, bundle):
        return validate_preview_bundle(
            bundle,
            expected_repository=REPOSITORY,
            expected_pr_number=PR_NUMBER,
            expected_head_sha=HEAD_SHA,
            expected_base_ref=BASE_REF,
        )

    def assert_invalid(self, bundle, error_type, expected_message):
        with self.assertRaises(error_type) as error:
            self.validate(bundle)

        self.assertEqual(str(error.exception), expected_message)

    def test_validate_preview_bundle_returns_none_when_bundle_is_valid(self):
        result = self.validate(make_bundle())

        self.assertIsNone(result)

    def test_validate_preview_bundle_returns_none_when_description_is_empty(self):
        bundle = make_bundle()
        bundle["documents"][0]["children"][0]["description"] = ""

        result = self.validate(bundle)

        self.assertIsNone(result)

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
            "$.pull_request.number does not match the validated PR",
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
            "$.pull_request.head_sha does not match the validated PR",
        )

    def test_validate_preview_bundle_raises_value_error_when_base_ref_mismatch(self):
        bundle = make_bundle()
        bundle["pull_request"]["base_ref"] = "release"

        self.assert_invalid(
            bundle,
            ValueError,
            "$.pull_request.base_ref does not match the expected base branch",
        )

    def test_validate_preview_bundle_raises_type_error_when_documents_not_list(self):
        bundle = make_bundle()
        bundle["documents"] = {}

        self.assert_invalid(
            bundle,
            TypeError,
            "$.documents must be an array",
        )

    @patch("tools.github_readme_sync.preview.MAX_CATEGORIES", 1)
    def test_validate_preview_bundle_raises_value_error_when_too_many_categories(self):
        bundle = make_bundle()
        bundle["documents"].append({"title": "Second Category", "children": []})

        self.assert_invalid(
            bundle,
            ValueError,
            "$.documents may contain at most 1 categories",
        )

    def test_validate_preview_bundle_raises_value_error_when_category_key_unknown(self):
        bundle = make_bundle()
        bundle["documents"][0]["slug"] = "category"

        self.assert_invalid(
            bundle,
            ValueError,
            "$.documents[0] contains unknown keys: ['slug']",
        )

    def test_validate_preview_raises_type_error_when_category_children_not_list(self):
        bundle = make_bundle()
        bundle["documents"][0]["children"] = {}

        self.assert_invalid(
            bundle,
            TypeError,
            "$.documents[0].children must be an array",
        )

    def test_validate_preview_bundle_raises_value_error_when_doc_key_unknown(self):
        bundle = make_bundle()
        bundle["documents"][0]["children"][0]["file_path"] = "unexpected"

        self.assert_invalid(
            bundle,
            ValueError,
            "$.documents[0].children[0] contains unknown keys: ['file_path']",
        )

    def test_validate_preview_bundle_raises_value_error_when_doc_key_missing(self):
        bundle = make_bundle()
        del bundle["documents"][0]["children"][0]["body"]

        self.assert_invalid(
            bundle,
            ValueError,
            "$.documents[0].children[0] is missing required keys: ['body']",
        )

    def test_validate_preview_bundle_raises_value_error_when_slug_invalid(self):
        bundle = make_bundle()
        bundle["documents"][0]["children"][0]["slug"] = "Bad_Slug"

        self.assert_invalid(
            bundle,
            ValueError,
            "$.documents[0].children[0].slug must contain only lowercase "
            "letters, digits, and hyphens",
        )

    def test_validate_preview_bundle_raises_value_error_when_doc_slug_duplicate(self):
        bundle = make_bundle()
        bundle["documents"][0]["children"].append(make_document(title="Other Page"))

        self.assert_invalid(
            bundle,
            ValueError,
            "Duplicate document slug in preview bundle: 'page'",
        )

    def test_validate_preview_bundle_raises_type_error_when_body_not_string(self):
        bundle = make_bundle()
        bundle["documents"][0]["children"][0]["body"] = None

        self.assert_invalid(
            bundle,
            TypeError,
            "$.documents[0].children[0].body must be a string",
        )

    @patch("tools.github_readme_sync.preview.MAX_BODY_LENGTH", 5)
    def test_validate_preview_bundle_raises_value_error_when_body_too_long(self):
        bundle = make_bundle()
        bundle["documents"][0]["children"][0]["body"] = "123456"

        self.assert_invalid(
            bundle,
            ValueError,
            "$.documents[0].children[0].body exceeds the maximum length "
            "of 5 characters",
        )

    def test_validate_preview_bundle_raises_value_error_when_body_contains_nul(self):
        bundle = make_bundle()
        bundle["documents"][0]["children"][0]["body"] = "bad\x00body"

        self.assert_invalid(
            bundle,
            ValueError,
            "$.documents[0].children[0].body may not contain NUL characters",
        )

    def test_validate_preview_bundle_raises_type_error_when_hidden_not_bool(self):
        bundle = make_bundle()
        bundle["documents"][0]["children"][0]["hidden"] = "false"

        self.assert_invalid(
            bundle,
            TypeError,
            "$.documents[0].children[0].hidden must be a boolean",
        )

    @patch("tools.github_readme_sync.preview.MAX_DESCRIPTION_LENGTH", 5)
    def test_validate_preview_bundle_raises_value_error_when_description_too_long(self):
        bundle = make_bundle()
        bundle["documents"][0]["children"][0]["description"] = "123456"

        self.assert_invalid(
            bundle,
            ValueError,
            "$.documents[0].children[0].description exceeds the maximum length of 5",
        )

    def test_validate_preview_bundle_raises_type_error_when_doc_children_not_list(self):
        bundle = make_bundle()
        bundle["documents"][0]["children"][0]["children"] = {}

        self.assert_invalid(
            bundle,
            TypeError,
            "$.documents[0].children[0].children must be an array",
        )

    @patch("tools.github_readme_sync.preview.MAX_DOCUMENTS", 1)
    def test_validate_preview_bundle_raises_value_error_when_too_many_documents(self):
        bundle = make_bundle()
        bundle["documents"][0]["children"].append(
            make_document(title="Second Page", slug="second-page")
        )

        self.assert_invalid(
            bundle,
            ValueError,
            "Preview bundle may contain at most 1 documents",
        )

    @patch("tools.github_readme_sync.preview.MAX_DEPTH", 2)
    def test_validate_preview_bundle_raises_value_error_when_depth_limit_exceeded(self):
        bundle = make_bundle()
        deepest = make_document(
            title="Level Three",
            slug="level-three",
        )
        middle = make_document(
            title="Level Two",
            slug="level-two",
            children=[deepest],
        )
        root = make_document(
            title="Level One",
            slug="level-one",
            children=[middle],
        )
        bundle["documents"][0]["children"] = [root]

        self.assert_invalid(
            bundle,
            ValueError,
            "$.documents[0].children[0].children[0].children[0] "
            "exceeds the maximum nesting depth of 2",
        )


if __name__ == "__main__":
    unittest.main()
