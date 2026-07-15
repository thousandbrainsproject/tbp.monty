# Copyright 2025-2026 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest
from unittest.mock import MagicMock, call, patch

from tools.github_readme_sync.upload import (
    get_all_categories_docs,
    process_children,
    set_do_not_delete,
    upload,
)


class TestUpload(unittest.TestCase):
    @patch("tools.github_readme_sync.upload.get_all_categories_docs")
    @patch("tools.github_readme_sync.upload.process_children")
    def test_upload_cleans_up_before_making_version_stable(
        self,
        mock_process_children,
        mock_get_all_categories_docs,
    ):
        rdme = MagicMock()
        rdme.create_category_if_not_exists.return_value = (
            "/branches/0.40/categories/guides/Category%201",
            True,
        )

        # The inventory is category-first. upload() reverses it so pages are
        # deleted before their categories, then makes the version stable.
        mock_get_all_categories_docs.return_value = [
            {"title": "Old Category", "type": "category"},
            {"slug": "old-doc", "type": "doc"},
        ]

        hierarchy = [
            {
                "slug": "category-1",
                "title": "Category 1",
                "children": [],
            }
        ]

        upload(hierarchy, "/path/to/files", rdme)

        rdme.create_version_if_not_exists.assert_called_once_with()
        rdme.create_category_if_not_exists.assert_called_once_with("Category 1")
        mock_process_children.assert_called_once_with(
            parent=hierarchy[0],
            cat_id="/branches/0.40/categories/guides/Category%201",
            file_path="/path/to/files",
            rdme=rdme,
            to_be_deleted=[
                {"title": "Old Category", "type": "category"},
                {"slug": "old-doc", "type": "doc"},
            ],
        )

        # This call order protects the newly stable version from partial cleanup.
        self.assertLess(
            rdme.method_calls.index(call.delete_doc("old-doc")),
            rdme.method_calls.index(call.delete_category("Old Category")),
        )
        self.assertLess(
            rdme.method_calls.index(call.delete_category("Old Category")),
            rdme.method_calls.index(call.make_version_stable()),
        )

    @patch("tools.github_readme_sync.upload.print_child")
    @patch("tools.github_readme_sync.upload.load_doc")
    def test_process_children_uses_v2_resource_uris(
        self,
        mock_load_doc,
        mock_print_child,
    ):
        rdme = MagicMock()
        mock_load_doc.return_value = {
            "title": "Document",
            "slug": "child-1",
            "body": "Document body",
        }
        rdme.create_or_update_doc.return_value = (
            "/branches/0.40/guides/child-1",
            True,
        )

        parent = {
            "slug": "parent",
            "children": [{"slug": "child-1", "children": []}],
        }
        to_be_deleted = [{"slug": "child-1", "type": "doc"}]

        process_children(
            parent=parent,
            cat_id="/branches/0.40/categories/guides/Category",
            file_path="/path/to/files",
            rdme=rdme,
            to_be_deleted=to_be_deleted,
            parent_doc_id="/branches/0.40/guides/parent-doc",
        )

        mock_load_doc.assert_called_once_with(
            "/path/to/files",
            "parent",
            {"slug": "child-1", "children": []},
        )
        rdme.create_or_update_doc.assert_called_once_with(
            order=0,
            category_id="/branches/0.40/categories/guides/Category",
            doc=mock_load_doc.return_value,
            parent_id="/branches/0.40/guides/parent-doc",
            file_path="/path/to/files/parent",
        )
        mock_print_child.assert_called_once_with(
            0,
            mock_load_doc.return_value,
            True,
        )
        self.assertEqual(to_be_deleted, [])

    def test_set_do_not_delete_removes_document_by_slug(self):
        to_be_deleted = [
            {"slug": "test-doc", "type": "doc"},
            {"title": "Test Category", "type": "category"},
        ]

        set_do_not_delete(to_be_deleted, "test-doc")

        self.assertEqual(
            to_be_deleted,
            [{"title": "Test Category", "type": "category"}],
        )

    def test_set_do_not_delete_removes_category_by_title(self):
        to_be_deleted = [
            {"slug": "test-doc", "type": "doc"},
            {"title": "Test Category", "type": "category"},
        ]

        # API v2 categories are identified by title, not by category slug.
        set_do_not_delete(to_be_deleted, "Test Category")

        self.assertEqual(
            to_be_deleted,
            [{"slug": "test-doc", "type": "doc"}],
        )

    def test_get_all_categories_docs_uses_flat_v2_page_collection(self):
        rdme = MagicMock()
        rdme.get_categories.return_value = [
            {"title": "Category 1"},
            {"title": "Category 2"},
        ]
        rdme.get_category_docs.side_effect = [
            [
                {
                    "slug": "parent-doc",
                    "uri": "/branches/0.40/guides/parent-doc",
                    "parent": None,
                },
                {
                    "slug": "child-doc",
                    "uri": "/branches/0.40/guides/child-doc",
                    "parent": {"uri": "/branches/0.40/guides/parent-doc"},
                },
            ],
            [{"slug": "other-doc"}],
        ]

        result = get_all_categories_docs(rdme)

        # Cleanup only needs category titles and page slugs. Parent nesting is
        # irrelevant because pages are returned as one flat v2 collection.
        self.assertEqual(
            result,
            [
                {"title": "Category 1", "type": "category"},
                {"slug": "parent-doc", "type": "doc"},
                {"slug": "child-doc", "type": "doc"},
                {"title": "Category 2", "type": "category"},
                {"slug": "other-doc", "type": "doc"},
            ],
        )
        self.assertEqual(
            rdme.get_category_docs.call_args_list,
            [
                call({"title": "Category 1"}),
                call({"title": "Category 2"}),
            ],
        )


if __name__ == "__main__":
    unittest.main()
