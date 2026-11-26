# Copyright 2025-2026 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from tools.github_readme_sync.upload import (
    ReadMeItem,
    get_all_categories_docs,
    set_do_not_delete,
    upload,
    upload_rendered,
)


class TestUpload(unittest.TestCase):
    def test_upload_keeps_guide_with_same_name_as_obsolete_category(self):
        hierarchy = [
            {
                "slug": "examples",
                "title": "Examples",
                "children": [
                    {
                        "slug": "advanced",
                        "title": "Advanced",
                        "body": "Example",
                        "children": [],
                    }
                ],
            }
        ]

        for rendered in (False, True):
            with self.subTest(rendered=rendered):
                rdme = MagicMock()
                rdme.get_categories.return_value = [
                    {"title": "advanced"},
                    {"title": "Examples"},
                ]
                rdme.get_category_docs.side_effect = [
                    [{"slug": "obsolete"}],
                    [{"slug": "advanced"}],
                ]
                rdme.create_category_if_not_exists.return_value = (
                    "/branches/0.40/categories/guides/Examples",
                    False,
                )
                page = ("/branches/0.40/guides/advanced", False)

                if rendered:
                    rdme.create_or_update_rendered_doc.return_value = page
                    upload_rendered(hierarchy, rdme)
                    rdme.create_or_update_rendered_doc.assert_called_once()
                else:
                    rdme.create_or_update_doc.return_value = page
                    with tempfile.TemporaryDirectory() as folder:
                        category_path = Path(folder) / "examples"
                        category_path.mkdir()
                        (category_path / "advanced.md").write_text(
                            "---\ntitle: Advanced\n---\nExample\n", encoding="utf-8"
                        )
                        upload(hierarchy, folder, rdme)
                    rdme.create_or_update_doc.assert_called_once()

                rdme.delete_doc.assert_called_once_with("obsolete")
                rdme.delete_category.assert_called_once_with("advanced")

    def test_upload_keeps_category_with_same_name_as_obsolete_guide(self):
        hierarchy = [{"slug": "advanced", "title": "advanced", "children": []}]

        for rendered in (False, True):
            with self.subTest(rendered=rendered):
                rdme = MagicMock()
                rdme.get_categories.return_value = [
                    {"title": "Obsolete"},
                    {"title": "advanced"},
                ]
                rdme.get_category_docs.side_effect = [[{"slug": "advanced"}], []]
                rdme.create_category_if_not_exists.return_value = (
                    "/branches/0.40/categories/guides/advanced",
                    False,
                )

                if rendered:
                    upload_rendered(hierarchy, rdme)
                else:
                    upload(hierarchy, "/unused", rdme)

                rdme.delete_doc.assert_called_once_with("advanced")
                rdme.delete_category.assert_called_once_with("Obsolete")

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
            ReadMeItem(id="Old Category", type="category"),
            ReadMeItem(id="old-doc", type="doc"),
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
                ReadMeItem(id="Old Category", type="category"),
                ReadMeItem(id="old-doc", type="doc"),
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

        mock_get_all_categories_docs.assert_called_once_with(rdme)

    def test_set_do_not_delete_removes_document_by_id(self):
        to_be_deleted = [
            ReadMeItem(id="test-doc", type="doc"),
            ReadMeItem(id="Test Category", type="category"),
        ]

        set_do_not_delete(to_be_deleted, ReadMeItem(id="test-doc", type="doc"))

        self.assertEqual(
            to_be_deleted,
            [ReadMeItem(id="Test Category", type="category")],
        )

    def test_set_do_not_delete_removes_category_by_id(self):
        to_be_deleted = [
            ReadMeItem(id="test-doc", type="doc"),
            ReadMeItem(id="Test Category", type="category"),
        ]

        set_do_not_delete(
            to_be_deleted, ReadMeItem(id="Test Category", type="category")
        )

        self.assertEqual(
            to_be_deleted,
            [ReadMeItem(id="test-doc", type="doc")],
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

        self.assertEqual(
            result,
            [
                ReadMeItem(id="Category 1", type="category"),
                ReadMeItem(id="parent-doc", type="doc"),
                ReadMeItem(id="child-doc", type="doc"),
                ReadMeItem(id="Category 2", type="category"),
                ReadMeItem(id="other-doc", type="doc"),
            ],
        )
        self.assertEqual(
            rdme.get_category_docs.call_args_list,
            [
                call({"title": "Category 1"}),
                call({"title": "Category 2"}),
            ],
        )
