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
from unittest.mock import MagicMock, call

from tools.github_readme_sync.export import export
from tools.github_readme_sync.readme import ReadMe


class TestExport(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.test_dir.name) / "exported-docs"

    def tearDown(self):
        self.test_dir.cleanup()

    def test_export_rebuilds_nested_v2_pages_and_preserves_page_slugs(self):
        rdme = MagicMock(spec=ReadMe)

        # API v2 categories have titles but no category slug. export.py creates
        # a local folder slug from the title.
        category = {"title": "How to Use Monty"}
        rdme.get_categories.return_value = [category]

        # get_category_doc_tree() has already reconstructed the flat v2 page
        # collection using parent URIs. Deliberately use titles that do not
        # match the slugs to ensure export preserves server-provided slugs.
        rdme.get_category_doc_tree.return_value = [
            {
                "title": "Getting Started with Monty",
                "slug": "getting-started",
                "uri": "/branches/0.40/guides/getting-started",
                "children": [
                    {
                        "title": "Windows Setup via WSL",
                        "slug": "getting-started-on-windows-via-wsl",
                        "uri": (
                            "/branches/0.40/guides/getting-started-on-windows-via-wsl"
                        ),
                        "parent": {"uri": "/branches/0.40/guides/getting-started"},
                        "children": [],
                    }
                ],
            }
        ]

        rdme.get_doc_by_slug.side_effect = lambda slug: {
            "getting-started": "---\ntitle: Getting Started with Monty\n---\nRoot",
            "getting-started-on-windows-via-wsl": (
                "---\ntitle: Windows Setup via WSL\n---\nChild"
            ),
        }[slug]

        hierarchy = export(self.output_dir, rdme)

        self.assertEqual(
            hierarchy,
            [
                {
                    "title": "How to Use Monty",
                    "slug": "how-to-use-monty",
                    "children": [
                        {
                            "title": "Getting Started with Monty",
                            "slug": "getting-started",
                            "children": [
                                {
                                    "title": "Windows Setup via WSL",
                                    "slug": ("getting-started-on-windows-via-wsl"),
                                    "children": [],
                                }
                            ],
                        }
                    ],
                }
            ],
        )

        root_file = self.output_dir / "how-to-use-monty" / "getting-started.md"
        child_file = (
            self.output_dir
            / "how-to-use-monty"
            / "getting-started"
            / "getting-started-on-windows-via-wsl.md"
        )

        self.assertTrue(root_file.is_file())
        self.assertTrue(child_file.is_file())
        self.assertEqual(
            root_file.read_text(encoding="utf-8"),
            "---\ntitle: Getting Started with Monty\n---\nRoot",
        )
        self.assertEqual(
            child_file.read_text(encoding="utf-8"),
            "---\ntitle: Windows Setup via WSL\n---\nChild",
        )

        rdme.get_category_doc_tree.assert_called_once_with(category)
        self.assertEqual(
            rdme.get_doc_by_slug.call_args_list,
            [
                call("getting-started"),
                call("getting-started-on-windows-via-wsl"),
            ],
        )

    def test_export_replaces_an_existing_output_directory(self):
        rdme = MagicMock(spec=ReadMe)
        rdme.get_categories.return_value = []

        self.output_dir.mkdir(parents=True)
        stale_file = self.output_dir / "stale.md"
        stale_file.write_text("old export", encoding="utf-8")

        hierarchy = export(self.output_dir, rdme)

        self.assertEqual(hierarchy, [])
        self.assertTrue(self.output_dir.is_dir())
        self.assertFalse(stale_file.exists())


if __name__ == "__main__":
    unittest.main()
