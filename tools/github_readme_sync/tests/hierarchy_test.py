# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import contextlib
import http.server
import os
import socketserver
import tempfile
import threading
import unittest
from pathlib import Path

from tools.github_readme_sync.hierarchy import (
    CATEGORY_PREFIX,
    DOCUMENT_PREFIX,
    HIERARCHY_FILE,
    INDENTATION_UNIT,
    check_external,
    check_hierarchy_file,
    check_links,
    create_hierarchy_file,
    extract_external_links,
)
from tools.github_readme_sync.readme import ReadMe


class TestHierarchyFile(unittest.TestCase):
    def setUp(self):
        self.test_dir_context = tempfile.TemporaryDirectory()
        self.test_dir = self.test_dir_context.name
        self.server = None
        self.server_thread = None

    def tearDown(self):
        self.test_dir_context.cleanup()
        if self.server:
            self.server.shutdown()
        if self.server_thread:
            self.server_thread.join()

    def test_create_hierarchy_file(self):
        hierarchy_structure = [
            {
                "slug": "category-1",
                "title": "Category 1",
                "children": [
                    {
                        "slug": "doc-1",
                        "children": [
                            {"slug": "child-1", "children": [{"slug": "grandchild-1"}]}
                        ],
                    }
                ],
            }
        ]
        create_hierarchy_file(self.test_dir, hierarchy_structure)

        hierarchy_file_path = os.path.join(self.test_dir, HIERARCHY_FILE)
        self.assertTrue(os.path.exists(hierarchy_file_path))

        with open(hierarchy_file_path) as f:
            content = f.read()
            self.assertIn(f"{CATEGORY_PREFIX}category-1: Category 1\n", content)
            self.assertIn(f"{DOCUMENT_PREFIX}[doc-1](category-1/doc-1.md)\n", content)
            self.assertIn(
                f"{INDENTATION_UNIT}{DOCUMENT_PREFIX}"
                f"[child-1](category-1/doc-1/child-1.md)\n",
                content,
            )
            self.assertIn(
                f"{INDENTATION_UNIT}{INDENTATION_UNIT}{DOCUMENT_PREFIX}"
                f"[grandchild-1](category-1/doc-1/child-1/grandchild-1.md)\n",
                content,
            )

    def test_check_hierarchy_file_success(self):
        hierarchy_file = os.path.join(self.test_dir, HIERARCHY_FILE)
        with open(hierarchy_file, "w") as f:
            f.write(
                f"{CATEGORY_PREFIX}category-1: Category 1\n"
                f"{DOCUMENT_PREFIX}[doc-1](category-1/doc-1.md)\n"
            )

        os.makedirs(os.path.join(self.test_dir, "category-1"))
        doc_file = os.path.join(self.test_dir, "category-1", "doc-1.md")
        with open(doc_file, "w") as f:
            f.write("---\ntitle: Doc 1\n---\nContent")

        check_hierarchy_file(self.test_dir)

    def test_check_hierarchy_file_duplicate_slugs(self):
        hierarchy_file = os.path.join(self.test_dir, HIERARCHY_FILE)
        with open(hierarchy_file, "w") as f:
            f.write(
                f"{CATEGORY_PREFIX}category-1: Category 1\n"
                f"{DOCUMENT_PREFIX}[doc-1](category-1/doc-1.md)\n"
                f"{DOCUMENT_PREFIX}[doc-1](doc-1.md)\n"
            )

        os.makedirs(os.path.join(self.test_dir, "category-1"))
        doc_file1 = os.path.join(self.test_dir, "category-1", "doc-1.md")
        with open(doc_file1, "w") as f:
            f.write("---\ntitle: Doc 1\n---\nContent")

        doc_file2 = os.path.join(self.test_dir, "doc-1.md")
        with open(doc_file2, "w") as f:
            f.write("---\ntitle: Doc 1\n---\nContent")

        with self.assertLogs(level="ERROR") as log:
            with self.assertRaises(SystemExit):
                check_hierarchy_file(self.test_dir)

        self.assertTrue(any("Duplicate" in message for message in log.output))

    def test_check_hierarchy_broken_link_in_file(self):
        hierarchy_file = os.path.join(self.test_dir, HIERARCHY_FILE)
        with open(hierarchy_file, "w") as f:
            f.write(
                f"{CATEGORY_PREFIX}category-1: Category 1\n"
                f"{DOCUMENT_PREFIX}[doc-1](category-1/doc-1.md)\n"
            )

        os.makedirs(os.path.join(self.test_dir, "category-1"))
        doc_file = os.path.join(self.test_dir, "category-1", "doc-1.md")
        with open(doc_file, "w") as f:
            f.write(
                "---\ntitle: Doc 1\n---\nContent\n"
                "[missing](category-1/missing.md)\n"
                "[fragment](category-1/missing.md#fragment)"
            )

        existing_file = os.path.join(self.test_dir, "category-1", "existing.md")
        with open(existing_file, "w") as f:
            f.write("---\ntitle: Existing Doc\n---\nContent")

        with self.assertLogs(level="ERROR") as log:
            with self.assertRaises(SystemExit):
                check_hierarchy_file(self.test_dir)

        self.assertTrue(any("missing.md" in message for message in log.output))
        self.assertTrue(any("fragment" in message for message in log.output))

    def test_check_external_links(self):
        # Set up a mock server
        class MockHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if self.path == "/missing":
                    self.send_response(404)
                    self.end_headers()
                else:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"OK")

        def run_server():
            with socketserver.TCPServer(("", 0), MockHandler) as httpd:
                self.server = httpd
                port = httpd.server_address[1]
                self.server_url = f"http://localhost:{port}"
                httpd.serve_forever()

        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.start()

        while not hasattr(self, "server_url"):
            pass

        test_file = os.path.join(self.test_dir, "test_external_links.md")
        with open(test_file, "w") as f:
            f.write(f"[Valid Link]({self.server_url}/valid)\n")
            f.write(f"[Missing Link]({self.server_url}/missing)\n")
            f.write(f"[Fragment]({self.server_url}/valid#fragment)\n")

        with self.assertLogs(level="ERROR") as log:
            with contextlib.suppress(SystemExit):
                check_external(self.test_dir, [], ReadMe("0.0"))

        self.assertTrue(
            any(
                f"broken link: {self.server_url}/missing" in message
                for message in log.output
            )
        )
        self.assertFalse(
            any(f"{self.server_url}/valid" in message for message in log.output)
        )

    def test_extract_links_happy_path(self):
        def extract(input_string, expected_output):
            self.assertEqual(extract_external_links(input_string), expected_output)

        extract("[Link 1](https://a.com)", ["https://a.com"])
        extract(
            "[Link 2 with spaces](https://a.com/path with spaces)",
            ["https://a.com/path with spaces"],
        )
        extract(
            "![Image 1](https://a.com/image.jpg)",
            ["https://a.com/image.jpg"],
        )
        extract(
            "![](https://a.com/image-no-alt.jpg)",
            ["https://a.com/image-no-alt.jpg"],
        )
        extract(
            '<a href="https://a.com/html-link">HTML Link</a>',
            ["https://a.com/html-link"],
        )
        extract(
            "<a href='https://a.com/single-quote'>HTML Link</a>",
            ["https://a.com/single-quote"],
        )
        extract(
            '<img src="https://a.com/html-image.png" alt="HTML Image">',
            ["https://a.com/html-image.png"],
        )
        extract(
            "<img src='https://a.com/html-image-no-alt.png' alt=''>",
            ["https://a.com/html-image-no-alt.png"],
        )
        extract(
            "[Link with (parentheses) inside](https://a.com/parentheses)",
            ["https://a.com/parentheses"],
        )
        extract(
            '[Link with "quotes"](https://a.com/quotes)',
            ["https://a.com/quotes"],
        )

    def test_check_links_ignores_external(self):
        """Test that check_links ignores external links (http/https/mailto)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("""
            [External Link](https://example.com/page.md)
            [Another External](http://test.com/doc.md)
            [Email Link](mailto:test@example.md)
            """)
            temp_path = f.name

        try:
            self.assertEqual(check_links(temp_path), [])
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    unittest.main()
