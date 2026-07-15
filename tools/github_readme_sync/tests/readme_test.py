# Copyright 2025-2026 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from tools.github_readme_sync.readme import (
    API_PREFIX,
    GITHUB_RAW,
    DocumentNotFound,
    ReadMe,
)


class TestReadme(unittest.TestCase):
    def setUp(self):
        self.version = "1.0.0"
        self.readme = ReadMe(self.version)

    @patch("tools.github_readme_sync.readme.get")
    def test_get_stable_version_uses_v2_stable_alias(self, mock_get):
        mock_get.return_value = {
            "name": "0.40",
            "privacy": {"view": "default"},
        }

        stable_version = self.readme.get_stable_version()

        self.assertEqual(stable_version, "0.40")
        mock_get.assert_called_once_with(f"{API_PREFIX}/branches/stable")

    @patch("tools.github_readme_sync.readme.get")
    def test_get_stable_version_raises_when_alias_is_missing(self, mock_get):
        mock_get.return_value = None

        with self.assertRaisesRegex(ValueError, "No stable version found"):
            self.readme.get_stable_version()

    @patch("tools.github_readme_sync.readme.get")
    def test_get_stable_version_requires_name(self, mock_get):
        mock_get.return_value = {"privacy": {"view": "default"}}

        with self.assertRaisesRegex(
            ValueError,
            "Stable version response did not contain a name",
        ):
            self.readme.get_stable_version()

    @patch("tools.github_readme_sync.readme.get_collection")
    def test_get_categories_uses_branch_scoped_v2_endpoint(
        self,
        mock_get_collection,
    ):
        categories = [
            {"title": "Category 2", "uri": "/categories/Category%202"},
            {"title": "Category 1", "uri": "/categories/Category%201"},
        ]
        mock_get_collection.return_value = categories

        result = self.readme.get_categories()

        # The API response order is preserved; v2 does not return the old
        # category order property that the v1 tests sorted on.
        self.assertEqual(result, categories)
        mock_get_collection.assert_called_once_with(
            f"{API_PREFIX}/branches/{self.version}/categories/guides"
        )

    @patch("tools.github_readme_sync.readme.get_collection")
    def test_get_categories_empty_collection(self, mock_get_collection):
        mock_get_collection.return_value = []

        self.assertEqual(self.readme.get_categories(), [])

    @patch("tools.github_readme_sync.readme.get_collection")
    def test_get_category_docs_uses_quoted_category_title(
        self,
        mock_get_collection,
    ):
        pages = [
            {
                "title": "Doc 1",
                "slug": "doc-1",
                "uri": "/branches/1.0.0/guides/doc-1",
            }
        ]
        mock_get_collection.return_value = pages

        result = self.readme.get_category_docs({"title": "How to Use Monty & More"})

        self.assertEqual(result, pages)
        mock_get_collection.assert_called_once_with(
            f"{API_PREFIX}/branches/{self.version}/categories/"
            "guides/How%20to%20Use%20Monty%20%26%20More/pages"
        )

    @patch.object(ReadMe, "get_category_docs")
    def test_get_category_doc_tree_rebuilds_nested_pages(
        self,
        mock_get_category_docs,
    ):
        parent_uri = "/branches/1.0.0/guides/parent"
        mock_get_category_docs.return_value = [
            {
                "title": "Parent",
                "slug": "parent",
                "uri": parent_uri,
                "parent": None,
            },
            {
                "title": "First Child",
                "slug": "first-child",
                "uri": "/branches/1.0.0/guides/first-child",
                "parent": {"uri": parent_uri},
            },
            {
                "title": "Second Child",
                "slug": "second-child",
                "uri": "/branches/1.0.0/guides/second-child",
                "parent": {"uri": parent_uri},
            },
            {
                "title": "Other Root",
                "slug": "other-root",
                "uri": "/branches/1.0.0/guides/other-root",
                "parent": None,
            },
        ]

        roots = self.readme.get_category_doc_tree({"title": "Category"})

        self.assertEqual(
            [page["slug"] for page in roots],
            ["parent", "other-root"],
        )
        self.assertEqual(
            [child["slug"] for child in roots[0]["children"]],
            ["first-child", "second-child"],
        )
        self.assertEqual(roots[1]["children"], [])

    @patch.object(ReadMe, "get_category_docs")
    def test_get_category_doc_tree_rejects_page_without_uri(
        self,
        mock_get_category_docs,
    ):
        mock_get_category_docs.return_value = [
            {"title": "Missing URI", "slug": "missing-uri"}
        ]

        with self.assertRaisesRegex(ValueError, "has no uri"):
            self.readme.get_category_doc_tree({"title": "Category"})

    @patch.object(ReadMe, "get_category_docs")
    def test_get_category_doc_tree_rejects_duplicate_uri(
        self,
        mock_get_category_docs,
    ):
        duplicate_uri = "/branches/1.0.0/guides/duplicate"
        mock_get_category_docs.return_value = [
            {"title": "One", "slug": "one", "uri": duplicate_uri},
            {"title": "Two", "slug": "two", "uri": duplicate_uri},
        ]

        with self.assertRaisesRegex(ValueError, "duplicate page URI"):
            self.readme.get_category_doc_tree({"title": "Category"})

    @patch.object(ReadMe, "get_category_docs")
    def test_get_category_doc_tree_rejects_missing_parent(
        self,
        mock_get_category_docs,
    ):
        mock_get_category_docs.return_value = [
            {
                "title": "Orphan",
                "slug": "orphan",
                "uri": "/branches/1.0.0/guides/orphan",
                "parent": {"uri": "/branches/1.0.0/guides/missing"},
            }
        ]

        with self.assertRaisesRegex(ValueError, "refers to missing parent URI"):
            self.readme.get_category_doc_tree({"title": "Category"})

    @patch("tools.github_readme_sync.readme.get")
    def test_get_doc_by_slug_exports_v2_content_and_frontmatter(self, mock_get):
        mock_get.return_value = {
            "title": "Test Document",
            "slug": "test-doc",
            "uri": "/branches/1.0.0/guides/test-doc",
            "privacy": {"view": "public"},
            "content": {
                "body": "This is a test document.",
                "excerpt": "A description",
            },
        }

        doc = self.readme.get_doc_by_slug("test-doc")

        self.assertIn("title: Test Document", doc)
        self.assertIn("description: A description", doc)
        self.assertIn("This is a test document.", doc)
        mock_get.assert_called_once_with(
            f"{API_PREFIX}/branches/{self.version}/guides/test-doc"
        )

    @patch("tools.github_readme_sync.readme.get")
    def test_get_doc_by_slug_exports_hidden_page_and_empty_body(self, mock_get):
        mock_get.return_value = {
            "title": "[Test] Document",
            "slug": "test-doc",
            "uri": "/branches/1.0.0/guides/test-doc",
            "privacy": {"view": "anyone_with_link"},
            "content": {"body": None, "excerpt": None},
        }

        doc = self.readme.get_doc_by_slug("test-doc")

        self.assertEqual(
            doc,
            "---\ntitle: '[Test] Document'\nhidden: true\n---\n",
        )

    @patch("tools.github_readme_sync.readme.get")
    def test_get_doc_by_slug_raises_document_not_found(self, mock_get):
        mock_get.return_value = None

        with self.assertRaisesRegex(DocumentNotFound, "Document missing not found"):
            self.readme.get_doc_by_slug("missing")

    @patch("tools.github_readme_sync.readme.get")
    def test_get_doc_rejects_alias_or_changed_slug(self, mock_get):
        mock_get.return_value = {
            "title": "Document",
            "slug": "canonical-slug",
            "uri": "/branches/1.0.0/guides/canonical-slug",
        }

        with self.assertRaisesRegex(
            ValueError,
            "resolved requested slug 'requested-slug' to 'canonical-slug'",
        ):
            self.readme.get_doc("requested-slug")

    @patch("tools.github_readme_sync.readme.get")
    def test_get_doc_requires_resource_uri(self, mock_get):
        mock_get.return_value = {"title": "Document", "slug": "test-doc"}

        with self.assertRaisesRegex(ValueError, "has no uri"):
            self.readme.get_doc("test-doc")

    @patch("tools.github_readme_sync.readme.patch")
    def test_make_version_stable_uses_live_v2_privacy_behavior(self, mock_patch):
        self.readme.make_version_stable()

        mock_patch.assert_called_once_with(
            f"{API_PREFIX}/branches/{self.version}",
            {"privacy": {"view": "default"}},
        )

    @patch("tools.github_readme_sync.readme.patch")
    def test_make_version_stable_skips_preview_version(self, mock_patch):
        self.readme.version = "1.0.0-beta"

        self.readme.make_version_stable()

        mock_patch.assert_not_called()

    @patch.object(ReadMe, "get_stable_version", return_value="0.39")
    @patch("tools.github_readme_sync.readme.post")
    @patch("tools.github_readme_sync.readme.get")
    def test_create_version_if_not_exists_uses_v2_branch_payload(
        self,
        mock_get,
        mock_post,
        mock_get_stable_version,
    ):
        mock_get.return_value = None
        mock_post.return_value = {
            "name": self.version,
            "uri": f"/branches/{self.version}",
        }

        created = self.readme.create_version_if_not_exists()

        self.assertTrue(created)
        mock_get.assert_called_once_with(f"{API_PREFIX}/branches/{self.version}")
        mock_get_stable_version.assert_called_once_with()
        mock_post.assert_called_once_with(
            f"{API_PREFIX}/branches",
            {
                "name": self.version,
                "base": "0.39",
                "privacy": {"view": "hidden"},
            },
        )

    @patch("tools.github_readme_sync.readme.post")
    @patch("tools.github_readme_sync.readme.get")
    def test_create_version_if_not_exists_returns_false_when_branch_exists(
        self,
        mock_get,
        mock_post,
    ):
        mock_get.return_value = {"name": self.version}

        created = self.readme.create_version_if_not_exists()

        self.assertFalse(created)
        mock_post.assert_not_called()

    @patch.object(ReadMe, "get_categories")
    @patch("tools.github_readme_sync.readme.delete")
    def test_delete_categories_uses_category_titles(
        self,
        mock_delete,
        mock_get_categories,
    ):
        mock_get_categories.return_value = [
            {"title": "Category 1"},
            {"title": "Category 2"},
        ]

        self.readme.delete_categories()

        self.assertEqual(
            mock_delete.call_args_list,
            [
                call(
                    f"{API_PREFIX}/branches/{self.version}/"
                    "categories/guides/Category%201"
                ),
                call(
                    f"{API_PREFIX}/branches/{self.version}/"
                    "categories/guides/Category%202"
                ),
            ],
        )

    @patch("tools.github_readme_sync.readme.delete")
    def test_delete_category_quotes_title(self, mock_delete):
        self.readme.delete_category("How to Use Monty & More")

        mock_delete.assert_called_once_with(
            f"{API_PREFIX}/branches/{self.version}/categories/"
            "guides/How%20to%20Use%20Monty%20%26%20More"
        )

    @patch("tools.github_readme_sync.readme.delete")
    def test_delete_doc_uses_branch_scoped_v2_endpoint(self, mock_delete):
        self.readme.delete_doc("doc-1")

        mock_delete.assert_called_once_with(
            f"{API_PREFIX}/branches/{self.version}/guides/doc-1"
        )

    @patch("tools.github_readme_sync.readme.delete")
    def test_delete_version_uses_v2_branch_endpoint(self, mock_delete):
        self.readme.delete_version()

        mock_delete.assert_called_once_with(f"{API_PREFIX}/branches/{self.version}")

    @patch("tools.github_readme_sync.readme.post")
    @patch("tools.github_readme_sync.readme.get")
    def test_create_category_if_not_exists_uses_title_and_resource_uri(
        self,
        mock_get,
        mock_post,
    ):
        mock_get.return_value = None
        mock_post.return_value = {
            "title": "New Category",
            "uri": "/branches/1.0.0/categories/guides/New%20Category",
        }

        category_uri, created = self.readme.create_category_if_not_exists(
            "New Category"
        )

        self.assertTrue(created)
        self.assertEqual(
            category_uri,
            "/branches/1.0.0/categories/guides/New%20Category",
        )
        mock_get.assert_called_once_with(
            f"{API_PREFIX}/branches/{self.version}/categories/guides/New%20Category"
        )
        mock_post.assert_called_once_with(
            f"{API_PREFIX}/branches/{self.version}/categories",
            {"title": "New Category", "section": "guide"},
        )

    @patch("tools.github_readme_sync.readme.post")
    @patch("tools.github_readme_sync.readme.get")
    def test_create_category_if_not_exists_returns_existing_uri(
        self,
        mock_get,
        mock_post,
    ):
        mock_get.return_value = {
            "title": "Existing Category",
            "uri": "/branches/1.0.0/categories/guides/Existing%20Category",
        }

        category_uri, created = self.readme.create_category_if_not_exists(
            "Existing Category"
        )

        self.assertFalse(created)
        self.assertEqual(
            category_uri,
            "/branches/1.0.0/categories/guides/Existing%20Category",
        )
        mock_post.assert_not_called()

    @patch("tools.github_readme_sync.readme.post")
    @patch("tools.github_readme_sync.readme.patch")
    @patch.object(ReadMe, "get_doc")
    @patch.object(ReadMe, "process_markdown", return_value="Processed body")
    def test_create_or_update_doc_creates_v2_guide(
        self,
        mock_process_markdown,
        mock_get_doc,
        mock_patch,
        mock_post,
    ):
        mock_get_doc.return_value = None
        mock_post.return_value = {
            "slug": "new-doc",
            "uri": "/branches/1.0.0/guides/new-doc",
        }
        doc = {
            "title": "New Doc",
            "body": "This is a new doc.",
            "slug": "new-doc",
            "description": "A description",
        }

        doc_uri, created = self.readme.create_or_update_doc(
            order=1,
            category_id="/branches/1.0.0/categories/guides/Category",
            doc=doc,
            parent_id="/branches/1.0.0/guides/parent-doc",
            file_path="docs/category",
        )

        self.assertTrue(created)
        self.assertEqual(doc_uri, "/branches/1.0.0/guides/new-doc")
        mock_process_markdown.assert_called_once_with(
            "This is a new doc.",
            "docs/category",
            "new-doc",
        )
        mock_post.assert_called_once_with(
            f"{API_PREFIX}/branches/{self.version}/guides",
            {
                "title": "New Doc",
                "type": "basic",
                "content": {
                    "body": "Processed body",
                    "excerpt": "A description",
                },
                "category": {"uri": "/branches/1.0.0/categories/guides/Category"},
                "privacy": {"view": "public"},
                "position": 1,
                "parent": {"uri": "/branches/1.0.0/guides/parent-doc"},
                "slug": "new-doc",
            },
            headers={"prefer": "handling=strict"},
        )
        mock_patch.assert_not_called()

    @patch("tools.github_readme_sync.readme.post")
    @patch("tools.github_readme_sync.readme.patch")
    @patch.object(ReadMe, "get_doc")
    @patch.object(ReadMe, "process_markdown", return_value="Updated body")
    def test_create_or_update_doc_updates_without_sending_slug(
        self,
        mock_process_markdown,
        mock_get_doc,
        mock_patch,
        mock_post,
    ):
        mock_get_doc.return_value = {
            "slug": "existing-doc",
            "uri": "/branches/1.0.0/guides/existing-doc",
        }
        doc = {
            "title": "Existing Doc",
            "body": "Updated source",
            "slug": "existing-doc",
            "hidden": True,
        }

        doc_uri, created = self.readme.create_or_update_doc(
            order=2,
            category_id="/branches/1.0.0/categories/guides/Category",
            doc=doc,
            parent_id=None,
            file_path="docs/category",
        )

        self.assertFalse(created)
        self.assertEqual(doc_uri, "/branches/1.0.0/guides/existing-doc")
        expected_payload = {
            "title": "Existing Doc",
            "type": "basic",
            "content": {"body": "Updated body"},
            "category": {"uri": "/branches/1.0.0/categories/guides/Category"},
            "privacy": {"view": "anyone_with_link"},
            "position": 2,
        }
        mock_patch.assert_called_once_with(
            f"{API_PREFIX}/branches/{self.version}/guides/existing-doc",
            expected_payload,
        )
        self.assertNotIn("slug", expected_payload)
        mock_post.assert_not_called()

    @patch("tools.github_readme_sync.readme.post")
    @patch.object(ReadMe, "get_doc", return_value=None)
    @patch.object(ReadMe, "process_markdown", return_value="Body")
    def test_create_or_update_doc_rejects_changed_created_slug(
        self,
        mock_process_markdown,
        mock_get_doc,
        mock_post,
    ):
        mock_post.return_value = {
            "slug": "new-doc-1",
            "uri": "/branches/1.0.0/guides/new-doc-1",
        }

        with self.assertRaisesRegex(ValueError, "expected 'new-doc'"):
            self.readme.create_or_update_doc(
                order=0,
                category_id="/branches/1.0.0/categories/guides/Category",
                doc={"title": "New Doc", "body": "Body", "slug": "new-doc"},
                parent_id=None,
                file_path="docs/category",
            )

    @patch("tools.github_readme_sync.readme.post")
    @patch.object(ReadMe, "get_doc", return_value=None)
    @patch.object(ReadMe, "process_markdown", return_value="Body")
    def test_create_or_update_doc_requires_created_uri(
        self,
        mock_process_markdown,
        mock_get_doc,
        mock_post,
    ):
        mock_post.return_value = {"slug": "new-doc"}

        with self.assertRaisesRegex(ValueError, "has no uri"):
            self.readme.create_or_update_doc(
                order=0,
                category_id="/branches/1.0.0/categories/guides/Category",
                doc={"title": "New Doc", "body": "Body", "slug": "new-doc"},
                parent_id=None,
                file_path="docs/category",
            )

    @patch.dict(os.environ, {"IMAGE_PATH": "user/repo/refs/head/main/docs/figures"})
    def test_correct_image_locations_markdown(self):
        """Test image location correction for Markdown image paths."""
        base_expected = (
            f"![Image 1]({GITHUB_RAW}/user/repo/refs/head/main/docs/figures/image1.png)"
        )

        # Test cases for Markdown image paths
        markdown_paths = [
            "![Image 1](../figures/image1.png)",
            "![Image 1](../../figures/image1.png)",
            "![Image 1](../../../figures/image1.png)",
            "![Image 1](../../../../figures/image1.png)",
            "![Image 1](../../../../../figures/image1.png)",
        ]

        markdown_paths_not_modified = [
            "![Image 1](https://example.com/image1.png)",
            "![Image 1](../figures/docs-only-example.png)",
        ]

        for path in markdown_paths:
            self.assertEqual(self.readme.correct_image_locations(path), base_expected)

        for path in markdown_paths_not_modified:
            self.assertEqual(self.readme.correct_image_locations(path), path)

    def test_parse_images(self):
        images = [
            "![Image 1 Caption](../figures/image1.png#width=300px&height=200px)",
            "![](../figures/image1.png)",
            "![Image 1 Caption](../figures/docs-only-example.png)",
        ]
        expected = [
            '<figure><img src="../figures/image1.png" align="center"'
            ' style="border-radius: 8px; width: 300px; height: 200px">'
            "<figcaption>Image 1 Caption</figcaption></figure>",
            '<figure><img src="../figures/image1.png" align="center"'
            ' style="border-radius: 8px;"></figure>',
            "![Image 1 Caption](../figures/docs-only-example.png)",
        ]
        for i, image in enumerate(images):
            self.assertEqual(self.readme.parse_images(image), expected[i])

    @patch.dict(os.environ, {"IMAGE_PATH": "user/repo"})
    def test_correct_file_locations_markdown(self):
        """Test file location correction for Markdown file paths."""
        base_expected = (
            "[File 1](/docs/slug#sub-heading) and [File 2](/docs/slug2#sub-heading)"
        )

        # Test cases for Markdown file paths
        markdown_paths_with_deep_link = [
            (
                "[File 1](slug.md#sub-heading) and "  # fmt: skip noqa: RUF028
                "[File 2](slug2.md#sub-heading)"
            ),
            (
                "[File 1](contibuting/slug.md#sub-heading) and "
                "[File 2](contibuting/slug2.md#sub-heading)"
            ),
            (
                "[File 1](../contibuting/slug.md#sub-heading) and "
                "[File 2](../contibuting/slug2.md#sub-heading)"
            ),
            (
                "[File 1](../../contibuting/slug.md#sub-heading) and "
                "[File 2](../../contibuting/slug2.md#sub-heading)"
            ),
        ]

        markdown_paths_without_deep_link = [
            "[File 1](slug.md)",
            "[File 1](contibuting/slug.md)",
            "[File 1](../contibuting/slug.md)",
            "[File 1](../../contibuting/slug.md)",
        ]

        markdown_paths_that_should_not_change = [
            "[file 1](placeholder-example-doc.md)",
            "[file 1](../contributing/placeholder-example-doc.md)",
            "[file 1](../contributing/placeholder-example-doc.md#deep-link)",
            "[file 1](../some-existing-doc/blah.md#deep-link)",
        ]

        for path in markdown_paths_with_deep_link:
            self.assertEqual(self.readme.correct_file_locations(path), base_expected)

        for path in markdown_paths_without_deep_link:
            self.assertEqual(
                self.readme.correct_file_locations(path), "[File 1](/docs/slug)"
            )

        for path in markdown_paths_that_should_not_change:
            self.assertEqual(self.readme.correct_file_locations(path), path)

    @patch.dict(os.environ, {"IMAGE_PATH": "user/repo"})
    def test_ignored_path_locations(self):
        # Test cases for ignored paths
        ignored_paths = [
            "[File 1](https://example.com/slug.md#sub-heading)",
            "[File 1](http://example.com/slug.md#sub-heading)",
            "[File 1](mailto:blah@example.com)",
        ]

        for path in ignored_paths:
            self.assertEqual(self.readme.correct_file_locations(path), path)

    @patch.dict(os.environ, {"IMAGE_PATH": "user/repo"})
    def test_correct_image_locations_img_tag(self):
        """Test image location correction for HTML img tag paths."""
        img_tag = '<img src="../figures/image1.jpg" />'
        expected_img = f'<img src="{GITHUB_RAW}/user/repo/image1.jpg" />'
        self.assertEqual(self.readme.correct_image_locations(img_tag), expected_img)

    @patch.dict(os.environ, {"IMAGE_PATH": ""})
    def test_correct_image_locations_no_repo_env(self):
        body = "![Image 1](../figures/image1.png)"
        with self.assertRaises(ValueError) as context:
            self.readme.correct_image_locations(body)
        self.assertEqual(
            str(context.exception), "IMAGE_PATH environment variable not set"
        )

    def test_convert_note_tags_with_link(self):
        input_text = """
        > [!NOTE]
        > You can find our code at https://github.com/thousandbrainsproject/tbp.monty
        >
        > This is our open-source repository. We call it **Monty** after
        """

        expected_output = """
        > 📘
        > You can find our code at https://github.com/thousandbrainsproject/tbp.monty
        >
        > This is our open-source repository. We call it **Monty** after
        """

        self.assertEqual(
            self.readme.convert_note_tags(input_text).strip(), expected_output.strip()
        )
        input_text = """
        > [!NOTE]    This is a note.
        >   [!TIP]    Here's a tip.
        > [!IMPORTANT]  This is important.
        >     [!WARNING] This is a warning.
        > [!CAUTION] Be cautious!
        """

        expected_output = """
        > 📘    This is a note.
        >   👍    Here's a tip.
        > 📘  This is important.
        >     🚧 This is a warning.
        > ❗️ Be cautious!
        """

        # Compare stripped versions to ensure we ignore leading/trailing whitespace
        self.assertEqual(
            self.readme.convert_note_tags(input_text).strip(), expected_output.strip()
        )

    def test_convert_cloudinary_videos(self):
        input_text = """
        [Video Title](https://res.cloudinary.com/demo-cloud/video/upload/v12345/sample.mp4)
        Some text in between
        [Another Video](https://res.cloudinary.com/demo-cloud/video/upload/v67890/test.mp4)
        """

        expected_blocks = [
            {
                "html": (
                    '<div style="display: flex;justify-content: center;">'
                    '<video width="640" height="360" '
                    'style="border-radius: 10px;" controls poster="'
                    "https://res.cloudinary.com/demo-cloud/video/upload/v12345/sample.jpg"
                    '">'
                    '<source src="'
                    "https://res.cloudinary.com/demo-cloud/video/upload/v12345/sample.mp4"
                    '" type="video/mp4">'
                    "Your browser does not support the video tag.</video></div>"
                )
            },
            {
                "html": (
                    '<div style="display: flex;justify-content: center;">'
                    '<video width="640" height="360" '
                    'style="border-radius: 10px;" controls poster="'
                    "https://res.cloudinary.com/demo-cloud/video/upload/v67890/test.jpg"
                    '">'
                    '<source src="'
                    "https://res.cloudinary.com/demo-cloud/video/upload/v67890/test.mp4"
                    '" type="video/mp4">'
                    "Your browser does not support the video tag.</video></div>"
                )
            },
        ]

        result = self.readme.convert_cloudinary_videos(input_text)

        for block in expected_blocks:
            json_str = json.dumps(block, indent=2)
            self.assertIn(json_str, result)

        self.assertIn("[block:html]", result)
        self.assertIn("[/block]", result)

    def test_convert_cloudinary_videos_ignores_example_filename(self):
        input_text = """
        [Example Video](https://res.cloudinary.com/demo-cloud/video/upload/v12345/example-video.mp4)
        """

        expected_output = (
            "\n"
            "        [Example Video](https://res.cloudinary.com/demo-cloud/"
            "video/upload/v12345/example-video.mp4)\n"
            "        "
        )

        result = self.readme.convert_cloudinary_videos(input_text)

        self.assertEqual(result, expected_output)

    def test_convert_youtube_videos(self):
        input_text = """
        [First YouTube Video](https://www.youtube.com/watch?v=dQw4w9WgXcQ)
        """

        expected_html = (
            '<iframe class=\\"embedly-embed\\" src=\\"//cdn.embedly.com/'
            "widgets/media.html?src=https%3A%2F%2Fwww.youtube.com%2Fembed%"
            "2FdQw4w9WgXcQ%3Ffeature%3Doembed&display_name=YouTube&"
            "url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DdQw4w9WgXcQ&"
            "image=https%3A%2F%2Fi.ytimg.com%2Fvi%2FdQw4w9WgXcQ%2F"
            'hqdefault.jpg&type=text%2Fhtml&schema=youtube\\" '
            'width=\\"854\\" height=\\"480\\" scrolling=\\"no\\" '
            'title=\\"YouTube embed\\" frameborder=\\"0\\" '
            'allow=\\"autoplay; fullscreen; encrypted-media; '
            'picture-in-picture;\\" allowfullscreen=\\"true\\"></iframe>'
        )

        expected_output = (
            "[block:embed]\n"
            "{\n"
            f'  "html": "{expected_html}",\n'
            '  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",\n'
            '  "title": "First YouTube Video",\n'
            '  "favicon": "https://www.youtube.com/favicon.ico",\n'
            '  "image": "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg",\n'
            '  "provider": "https://www.youtube.com/",\n'
            '  "href": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",\n'
            '  "typeOfEmbed": "youtube"\n'
            "}\n"
            "[/block]"
        )

        result = self.readme.convert_youtube_videos(input_text)

        self.assertEqual(result, expected_output)

    def test_convert_youtube_videos_ignores_example_video_id(self):
        input_text = """
        [Example Video](https://youtu.be/example-video-id)
        [Real Video](https://youtu.be/dQw4w9WgXcQ)
        """

        expected_html = (
            '<iframe class=\\"embedly-embed\\" src=\\"//cdn.embedly.com/'
            "widgets/media.html?src=https%3A%2F%2Fwww.youtube.com%2Fembed%"
            "2FdQw4w9WgXcQ%3Ffeature%3Doembed&display_name=YouTube&"
            "url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DdQw4w9WgXcQ&"
            "image=https%3A%2F%2Fi.ytimg.com%2Fvi%2FdQw4w9WgXcQ%2F"
            'hqdefault.jpg&type=text%2Fhtml&schema=youtube\\" '
            'width=\\"854\\" height=\\"480\\" scrolling=\\"no\\" '
            'title=\\"YouTube embed\\" frameborder=\\"0\\" '
            'allow=\\"autoplay; fullscreen; encrypted-media; '
            'picture-in-picture;\\" allowfullscreen=\\"true\\"></iframe>'
        )

        expected_output = (
            "\n"
            "        [Example Video](https://youtu.be/example-video-id)\n"
            "[block:embed]\n"
            "{\n"
            f'  "html": "{expected_html}",\n'
            '  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",\n'
            '  "title": "Real Video",\n'
            '  "favicon": "https://www.youtube.com/favicon.ico",\n'
            '  "image": "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg",\n'
            '  "provider": "https://www.youtube.com/",\n'
            '  "href": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",\n'
            '  "typeOfEmbed": "youtube"\n'
            "}\n"
            "[/block]"
        )

        result = self.readme.convert_youtube_videos(input_text)

        self.assertEqual(result, expected_output)

    def test_caption_markdown_images_multiple_per_line(self):
        input_text = (
            "![First Image](path/to/first.png) ![Second Image](path/to/second.png)"
        )

        expected_output = (
            '<figure><img src="path/to/first.png" align="center" '
            'style="border-radius: 8px;">'
            "<figcaption>First Image</figcaption></figure> "
            '<figure><img src="path/to/second.png" align="center" '
            'style="border-radius: 8px;">'
            "<figcaption>Second Image</figcaption></figure>"
        )

        result = self.readme.parse_images(input_text)
        self.assertEqual(result, expected_output)

    def test_convert_csv_to_html_table(self):
        # Create a temporary CSV file for testing
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp:
            writer = csv.writer(tmp)
            writer.writerow(
                [
                    "Name",
                    "Score %|hover Scöre is the 'percentage' correct",
                    "Time (s)|align right",
                    "Time (mins)|align left",
                    "Mixed Column|    align left| hover Mixed Column",
                ]
            )
            writer.writerow(["Test 1", "95.01", "55", "10e4", "123"])
            writer.writerow(["Test 2", "-87.00", "72", "1/2", "456s"])
            tmp_path = tmp.name

        try:
            result = self.readme.convert_csv_to_html_table(f"!table[{tmp_path}]", "")

            # Check overall structure
            self.assertIn('<div class="data-table"><table>', result)
            self.assertIn("</table></div>", result)

            # Check headers
            self.assertIn("<thead>", result)
            self.assertIn("<th>Name</th>", result)
            self.assertIn("<th>Time (s)</th>", result)
            self.assertIn("title=\"Scöre is the 'percentage' correct\"", result)
            self.assertIn('title="Mixed Column"', result)

            # Check data rows
            self.assertIn("<tbody>", result)
            self.assertIn("<td>Test 1</td>", result)
            self.assertIn("<td>95.01</td>", result)
            self.assertIn('<td style="text-align:right">55</td>', result)
            self.assertIn("<td>Test 2</td>", result)
            self.assertIn("<td>-87.00</td>", result)
            self.assertIn('<td style="text-align:right">72</td>', result)
            self.assertIn('<td style="text-align:left">1/2</td>', result)
            self.assertIn('<td style="text-align:left">10e4</td>', result)
            self.assertIn('<td style="text-align:left">123</td>', result)
            self.assertIn('<td style="text-align:left">456s</td>', result)

            # Test with non-existent file
            result = self.readme.convert_csv_to_html_table(
                "!table[non_existent.csv]", ""
            )
            self.assertTrue(result.startswith("[Failed to load table"))
        finally:
            Path(tmp_path).unlink()

    def test_invalid_alignment_value(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp:
            writer = csv.writer(tmp)
            writer.writerow(["Name", "Score %|align wrong"])
            writer.writerow(["Test 1", "95.01"])
            tmp_path = tmp.name

        try:
            result = self.readme.convert_csv_to_html_table(f"!table[{tmp_path}]", "")
            self.assertIn("Must be 'left' or 'right'", result)
        finally:
            Path(tmp_path).unlink()

    def test_convert_csv_to_html_table_hides_hidden_columns(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp:
            writer = csv.writer(tmp)
            writer.writerow(
                [
                    "Experiment",
                    "Correct (%)|align right",
                    "Num Episodes|align right|hidden",
                ]
            )
            writer.writerow(["randrot_noise_10distinctobj_surf_agent", "100.00", "100"])
            writer.writerow(["base_10simobj_surf_agent", "98.57", "140"])
            tmp_path = tmp.name

        try:
            result = self.readme.convert_csv_to_html_table(f"!table[{tmp_path}]", "")

            expected = (
                '<div class="data-table"><table>\n'
                "<thead>\n"
                "<tr><th>Experiment</th><th>Correct (%)</th></tr>\n"
                "</thead>\n"
                "<tbody>\n"
                "<tr><td>randrot_noise_10distinctobj_surf_agent</td>"
                '<td style="text-align:right">100.00</td></tr>\n'
                "<tr><td>base_10simobj_surf_agent</td>"
                '<td style="text-align:right">98.57</td></tr>\n'
                "</tbody>\n"
                "</table></div>"
            )
            self.assertEqual(result, expected)
        finally:
            Path(tmp_path).unlink()

    def test_convert_csv_to_html_table_relative_path(self):
        # Create a temporary directory structure
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)

            # Create subdirectories
            data_dir = tmp_dir / "data"
            docs_dir = tmp_dir / "docs"
            data_dir.mkdir(parents=True)
            docs_dir.mkdir(parents=True)

            # Create a CSV file in the data directory
            csv_path = data_dir / "test.csv"
            with csv_path.open("w") as f:
                writer = csv.writer(f)
                writer.writerow(["Header 1", "Header 2"])
                writer.writerow(["Value 1", "Value 2"])

            # Create a mock markdown file path in the docs directory
            doc_path = docs_dir / "doc.md"

            # Test relative path from doc to csv
            result = self.readme.convert_csv_to_html_table(
                "!table[../../data/test.csv]", doc_path
            )

            # Check the table structure
            self.assertIn('<div class="data-table"><table>', result)
            self.assertIn("<thead>", result)
            self.assertIn("<th>Header 1</th>", result)
            self.assertIn("<th>Header 2</th>", result)
            self.assertIn("<tbody>", result)
            self.assertIn("<td>Value 1</td>", result)
            self.assertIn("<td>Value 2</td>", result)
            self.assertIn("</table></div>", result)

    def test_insert_markdown_snippet(self):
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            docs_dir = tmp_dir / "docs"
            other_dir = tmp_dir / "other"
            docs_dir.mkdir(parents=True)
            other_dir.mkdir(parents=True)

            source_md = other_dir / "source.md"
            with source_md.open("w") as f:
                f.write(
                    "# Test Header\nThis is test content\n* List item 1\n* List item 2"
                )
            doc_path = docs_dir / "doc.md"

            result = self.readme.insert_markdown_snippet(
                "!snippet[../../other/source.md]", doc_path
            )

            expected_content = (
                "# Test Header\nThis is test content\n* List item 1\n* List item 2"
            )
            self.assertEqual(result, expected_content)

            result = self.readme.insert_markdown_snippet(
                "!snippet[../other/nonexistent.md]", doc_path
            )
            self.assertIn("File not found", result)

    def test_sanitize_html_removes_scripts(self):
        html_with_script = """
        <div>
            <h1>Test Content</h1>
            <p>This is a test paragraph</p>
            <script>
                alert('This is a malicious script');
                document.cookie = "session=stolen";
            </script>
            <p>More content after the script</p>
        </div>
        """

        sanitized_html = self.readme.sanitize_html(html_with_script)

        # Verify script tag is removed
        self.assertNotIn("<script>", sanitized_html)
        self.assertNotIn("</script>", sanitized_html)
        self.assertNotIn("alert('This is a malicious script')", sanitized_html)
        self.assertNotIn("document.cookie", sanitized_html)

        # Verify legitimate content is preserved
        self.assertIn("<h1>Test Content</h1>", sanitized_html)
        self.assertIn("<p>This is a test paragraph</p>", sanitized_html)
        self.assertIn("<p>More content after the script</p>", sanitized_html)


if __name__ == "__main__":
    unittest.main()
