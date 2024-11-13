# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest
from unittest.mock import MagicMock, patch

from tools.github_readme_sync.upload import (
    get_all_categories_docs,
    process_children,
    set_do_not_delete,
    upload,
)


class TestUpload(unittest.TestCase):
    @patch("tools.github_readme_sync.upload.get_all_categories_docs")
    @patch("tools.github_readme_sync.upload.process_children")
    def test_upload(self, mock_process_children, mock_get_all_categories_docs):
        mock_rdme_instance = MagicMock()
        mock_get_all_categories_docs.return_value = [{"slug": "test", "type": "doc"}]

        new_hierarchy = [{"slug": "cat1", "title": "Category 1", "children": []}]
        file_path = "/path/to/files"

        mock_rdme_instance.create_category_if_not_exists.return_value = ("cat_id", True)

        upload(new_hierarchy, file_path, mock_rdme_instance)

        mock_rdme_instance.create_version_if_not_exists.assert_called_once()
        mock_rdme_instance.create_category_if_not_exists.assert_called_once_with(
            "cat1", "Category 1"
        )
        mock_process_children.assert_called_once()

    @patch("tools.github_readme_sync.upload.load_doc")
    def test_process_children(self, mock_load_doc):
        mock_rdme_instance = MagicMock()
        mock_load_doc.return_value = {"title": "Document", "slug": "doc1"}

        # Mock create_or_update_doc to return a tuple
        mock_rdme_instance.create_or_update_doc.return_value = ("doc_id", True)

        parent = {"slug": "parent", "children": [{"slug": "child1", "children": []}]}
        to_be_deleted = []

        process_children(
            parent=parent,
            cat_id="cat_id",
            file_path="/path/to/files",
            rdme=mock_rdme_instance,
            to_be_deleted=to_be_deleted,
        )

        mock_load_doc.assert_called_once()
        mock_rdme_instance.create_or_update_doc.assert_called_once()

    def test_set_do_not_delete(self):
        to_be_deleted = [
            {"slug": "test-doc", "type": "doc"},
            {"slug": "test-cat", "type": "category"},
        ]
        set_do_not_delete(to_be_deleted, "test-doc")
        self.assertEqual(len(to_be_deleted), 1)
        self.assertEqual(to_be_deleted[0]["slug"], "test-cat")

    def test_get_all_categories_docs(self):
        mock_rdme_instance = MagicMock()
        mock_rdme_instance.get_categories.return_value = [{"slug": "cat1"}]
        mock_rdme_instance.get_category_docs.return_value = [
            {"slug": "doc1", "children": []}
        ]

        result = get_all_categories_docs(mock_rdme_instance)
        expected = [
            {"slug": "cat1", "type": "category"},
            {"slug": "doc1", "type": "doc"},
        ]

        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
