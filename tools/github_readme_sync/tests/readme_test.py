# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json
import os
import unittest
from unittest.mock import patch

from tools.github_readme_sync.readme import GITHUB_RAW, ReadMe


class TestReadme(unittest.TestCase):
    def setUp(self):
        self.version = "1.0.0"
        self.readme = ReadMe(self.version)

    @patch("tools.github_readme_sync.readme.get")
    def test_get_stable_version(self, mock_get):
        mock_get.return_value = [
            {
                "version": "0.0-dry-run",
                "version_clean": "0.0.0-dry-run",
                "codename": "",
                "is_stable": False,
                "is_beta": False,
                "is_hidden": True,
                "is_deprecated": False,
                "pdfStatus": "",
                "_id": "66f2da3113724c00253aa01c",
                "createdAt": "2024-09-24T15:26:41.131Z",
            },
            {
                "version": "1.0.0",
                "version_clean": "1.0.0",
                "codename": "",
                "is_stable": True,
                "is_beta": False,
                "is_hidden": False,
                "is_deprecated": False,
                "pdfStatus": "",
                "_id": "6669fcac47cf690019d6192d",
                "createdAt": "2024-06-12T19:53:16.502Z",
            },
        ]
        stable_version = self.readme.get_stable_version()
        self.assertEqual(stable_version, "1.0.0")
        mock_get.assert_called_once_with(
            "https://dash.readme.com/api/v1/version",
        )

    @patch("tools.github_readme_sync.readme.get")
    def test_get_stable_version_none(self, mock_get):
        # Simulate None being returned by the API
        mock_get.return_value = None

        # Assert that ValueError is raised when no versions are returned
        with self.assertRaises(ValueError) as context:
            self.readme.get_stable_version()

        self.assertEqual(
            str(context.exception), "Failed to retrieve versions or no versions found"
        )
        mock_get.assert_called_once_with(
            "https://dash.readme.com/api/v1/version",
        )

    @patch("tools.github_readme_sync.readme.get")
    def test_get_categories(self, mock_get):
        mock_get.return_value = [
            {"order": 2, "name": "Category 2"},
            {"order": 1, "name": "Category 1"},
        ]
        categories = self.readme.get_categories()
        self.assertEqual(len(categories), 2)
        self.assertEqual(categories[0]["name"], "Category 1")
        mock_get.assert_called_once_with(
            "https://dash.readme.com/api/v1/categories",
            {"x-readme-version": self.version},
        )

    @patch("tools.github_readme_sync.readme.get")
    def test_get_categories_none(self, mock_get):
        mock_get.return_value = None
        categories = self.readme.get_categories()
        # assert empty list
        self.assertEqual(categories, [])
        mock_get.assert_called_once_with(
            "https://dash.readme.com/api/v1/categories",
            {"x-readme-version": self.version},
        )

    @patch("tools.github_readme_sync.readme.get")
    def test_get_categories_raises_error(self, mock_get):
        mock_get.side_effect = ValueError("Failed to get categories")
        with self.assertRaises(ValueError) as context:
            self.readme.get_categories()
        self.assertEqual(str(context.exception), "Failed to get categories")

    @patch("tools.github_readme_sync.readme.get")
    def test_get_category_docs(self, mock_get):
        mock_get.return_value = [
            {"order": 2, "name": "Doc 2"},
            {"order": 1, "name": "Doc 1"},
        ]
        docs = self.readme.get_category_docs({"slug": "example-category"})
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0]["name"], "Doc 1")
        mock_get.assert_called_once_with(
            f"https://dash.readme.com/api/v1/categories/example-category/docs",
            {"x-readme-version": self.version},
        )

    @patch("tools.github_readme_sync.readme.get")
    def test_get_category_docs_raises_error(self, mock_get):
        mock_get.side_effect = ValueError("Failed to get category docs")
        with self.assertRaises(ValueError) as context:
            self.readme.get_category_docs({"slug": "example-category"})
        self.assertEqual(str(context.exception), "Failed to get category docs")

    @patch("tools.github_readme_sync.readme.get")
    def test_get_doc_by_slug(self, mock_get):
        mock_get.return_value = {
            "title": "Test Document",
            "body": "This is a test document.",
            "hidden": False,
        }
        doc = self.readme.get_doc_by_slug("test-doc")
        self.assertIn("title: Test Document", doc)
        self.assertIn("This is a test document.", doc)
        mock_get.assert_called_once_with(
            f"https://dash.readme.com/api/v1/docs/test-doc",
            {"x-readme-version": self.version},
        )

    @patch("tools.github_readme_sync.readme.get")
    def test_get_doc_by_slug_escaped(self, mock_get):
        mock_get.return_value = {
            "title": "[Test] Document",
            "body": "This is a test document.",
            "hidden": True,
        }
        doc = self.readme.get_doc_by_slug("test-doc")
        self.assertEqual(
            doc,
            """---
title: '[Test] Document'
hidden: true
---
This is a test document.""",
        )
        mock_get.assert_called_once_with(
            f"https://dash.readme.com/api/v1/docs/test-doc",
            {"x-readme-version": self.version},
        )

    @patch("tools.github_readme_sync.readme.get")
    def test_get_doc_by_slug_raises_error(self, mock_get):
        mock_get.side_effect = ValueError("Failed to get doc")
        with self.assertRaises(ValueError) as context:
            self.readme.get_doc_by_slug("test-doc")
        self.assertEqual(str(context.exception), "Failed to get doc")

    @patch("tools.github_readme_sync.readme.get")
    def test_get_doc_id(self, mock_get):
        mock_get.return_value = {"_id": "123"}
        doc_id = self.readme.get_doc_id("test-doc")
        self.assertEqual(doc_id, "123")
        mock_get.assert_called_once_with(
            f"https://dash.readme.com/api/v1/docs/test-doc",
            {"x-readme-version": self.version},
        )

    @patch("tools.github_readme_sync.readme.put")
    def test_make_version_stable(self, mock_put):
        mock_put.return_value = True
        self.readme.make_version_stable()
        mock_put.assert_called_once_with(
            f"https://dash.readme.com/api/v1/version/{self.version}",
            {"is_stable": True, "is_hidden": False},
        )

    @patch("tools.github_readme_sync.readme.put")
    def test_make_version_stable_with_suffix(self, mock_put):
        self.readme.version = "1.0.0-beta"
        self.readme.make_version_stable()
        mock_put.assert_not_called()

    @patch("tools.github_readme_sync.readme.get")
    @patch("tools.github_readme_sync.readme.post")
    def test_create_version_if_not_exists(self, mock_post, mock_get):
        # Mock the first get call to check if the version exists (returns None)
        mock_get.side_effect = [
            None,  # First get call for checking if the version exists
            [
                {
                    "version": "0.0.0-base",
                    "version_clean": "0.0.0-base",
                    "is_stable": True,
                    "is_beta": False,
                    "is_hidden": False,
                    "is_deprecated": False,
                    "pdfStatus": "",
                    "_id": "66f2da3113724c00253aa01c",
                    "createdAt": "2024-09-24T15:26:41.131Z",
                }
            ],  # Second get call to retrieve stable versions
        ]

        # Mock the post call to create the new version
        mock_post.return_value = True

        # Call the method to test
        created = self.readme.create_version_if_not_exists()

        # Assert the version was created
        self.assertTrue(created)

        # Check if the right API calls were made
        mock_get.assert_any_call(
            f"https://dash.readme.com/api/v1/version/{self.version}"
        )

        mock_get.assert_any_call("https://dash.readme.com/api/v1/version")

        mock_post.assert_called_once_with(
            "https://dash.readme.com/api/v1/version",
            {
                "version": self.version,
                "from": "0.0.0-base",
                "is_stable": False,
                "is_hidden": True,
            },
        )

    @patch.object(ReadMe, "get_categories")
    @patch("tools.github_readme_sync.readme.delete")
    def test_delete_categories(self, mock_delete, mock_get_categories):
        mock_get_categories.return_value = [
            {"slug": "category-1"},
            {"slug": "category-2"},
        ]
        self.readme.delete_categories()
        mock_get_categories.assert_called_once_with()
        self.assertEqual(mock_delete.call_count, 2)

    @patch("tools.github_readme_sync.readme.delete")
    def test_delete_category(self, mock_delete):
        self.readme.delete_category("category-1")
        mock_delete.assert_called_once_with(
            f"https://dash.readme.com/api/v1/categories/category-1",
            {"x-readme-version": self.version},
        )

    @patch("tools.github_readme_sync.readme.delete")
    def test_delete_doc(self, mock_delete):
        self.readme.delete_doc("doc-1")
        mock_delete.assert_called_once_with(
            f"https://dash.readme.com/api/v1/docs/doc-1",
            {"x-readme-version": self.version},
        )

    @patch("tools.github_readme_sync.readme.get")
    @patch("tools.github_readme_sync.readme.post")
    def test_create_category_if_not_exists(self, mock_post, mock_get):
        mock_get.return_value = None
        mock_post.return_value = json.dumps({"_id": "new-category-id"})
        category_id, created = self.readme.create_category_if_not_exists(
            "new-category", "New Category"
        )
        self.assertTrue(created)
        self.assertEqual(category_id, "new-category-id")
        mock_get.assert_called_once_with(
            f"https://dash.readme.com/api/v1/categories/new-category",
            {"x-readme-version": self.version},
        )
        mock_post.assert_called_once_with(
            "https://dash.readme.com/api/v1/categories",
            {"title": "New Category", "type": "guide"},
            {"x-readme-version": self.version},
        )

    @patch("tools.github_readme_sync.readme.get")
    @patch("tools.github_readme_sync.readme.put")
    @patch("tools.github_readme_sync.readme.post")
    @patch.dict(os.environ, {"IMAGE_PATH": "user/repo"})
    def test_create_or_update_doc(self, mock_post, mock_put, mock_get):
        mock_get.return_value = None  # Doc does not exist
        mock_post.return_value = json.dumps({"_id": "new-doc-id"})
        doc_id, created = self.readme.create_or_update_doc(
            order=1,
            category_id="category-id",
            doc={"title": "New Doc", "body": "This is a new doc.", "slug": "new-doc"},
            parent_id="parent-doc-id",
        )
        self.assertTrue(created)
        self.assertEqual(doc_id, "new-doc-id")
        mock_post.assert_called_once_with(
            "https://dash.readme.com/api/v1/docs",
            {
                "title": "New Doc",
                "type": "basic",
                "body": "This is a new doc.",
                "category": "category-id",
                "hidden": False,
                "order": 1,
                "parentDoc": "parent-doc-id",
            },
            {"x-readme-version": self.version},
        )
        mock_put.assert_not_called()

    @patch.dict(os.environ, {"IMAGE_PATH": "user/repo/refs/head/main/docs/figures"})
    def test_correct_image_locations_markdown(self):
        """Test image location correction for Markdown image paths."""
        base_expected = (
            f"![Image 1]({GITHUB_RAW}"
            f"/user/repo/refs/head/main/docs/figures/image1.png)"
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
            print(path)
            print(self.readme.correct_image_locations(path))
            self.assertEqual(self.readme.correct_image_locations(path), base_expected)

        for path in markdown_paths_not_modified:
            self.assertEqual(self.readme.correct_image_locations(path), path)

    def test_caption_markdown_images(self):
        images = [
            "![Image 1 Caption](../figures/image1.png)",
            "![](../figures/image1.png)",
            "![Image 1 Caption](../figures/docs-only-example.png)",
        ]
        expected = [
            '<figure><img src="../figures/image1.png" align="center"'
            ' style="border-radius: 8px;" />'
            "<figcaption>Image 1 Caption</figcaption></figure>",
            '<figure><img src="../figures/image1.png" align="center"'
            ' style="border-radius: 8px;" /></figure>',
            "![Image 1 Caption](../figures/docs-only-example.png)",
        ]
        # iterate
        for i, image in enumerate(images):
            self.assertEqual(self.readme.caption_markdown_images(image), expected[i])

    @patch.dict(os.environ, {"IMAGE_PATH": "user/repo"})
    def test_correct_file_locations_markdown(self):
        """Test file location correction for Markdown file paths."""
        base_expected = f"[File 1](/docs/slug#sub-heading)"

        # Test cases for Markdown file paths
        markdown_paths_with_deep_link = [
            "[File 1](slug.md#sub-heading)",
            "[File 1](contibuting/slug.md#sub-heading)",
            "[File 1](../contibuting/slug.md#sub-heading)",
            "[File 1](../../contibuting/slug.md#sub-heading)",
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
                self.readme.correct_file_locations(path), f"[File 1](/docs/slug)"
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

    def test_convert_note_tags(self):
        input_text = """
        > [!NOTE]    This is a note.
        >   [!TIP]    Here's a tip.
        > [!IMPORTANT]  This is important.
        >     [!WARNING] This is a warning.
        > [!CAUTION] Be cautious!
        """

        expected_output = """
        > 📘 This is a note.
        > 👍 Here's a tip.
        > 📘 This is important.
        > 🚧 This is a warning.
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

        # Check that each expected block appears in the result
        for block in expected_blocks:
            block_str = f"[block:html]\n{json.dumps(block, indent=2)}\n[/block]"
            self.assertIn(block_str, result)


if __name__ == "__main__":
    unittest.main()
