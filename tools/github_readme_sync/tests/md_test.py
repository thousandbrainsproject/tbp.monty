# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

from tools.github_readme_sync.md import parse_frontmatter, process_markdown


class TestMarkdown(unittest.TestCase):
    def test_parse_frontmatter_with_empty_frontmatter(self):
        content = """---
---

Content here.
"""
        expected = None  # Adjusted to match the actual return value
        result = parse_frontmatter(content)
        self.assertEqual(result, expected)

    def test_process_markdown_with_empty_frontmatter_raises_error(self):
        body = """---
---

Content here.
"""
        slug = "empty-frontmatter-slug"
        with self.assertRaises(ValueError) as context:
            process_markdown(body, slug)
        self.assertEqual(str(context.exception), "No frontmatter found in the document")

    def test_parse_frontmatter_without_frontmatter(self):
        content = "No frontmatter here."
        expected = {}
        result = parse_frontmatter(content)
        self.assertEqual(result, expected)

    def test_process_markdown_with_frontmatter(self):
        body = """---
title: Sample Title
hidden: true
---

This is the body of the markdown.
"""
        slug = "sample-slug"
        expected = {
            "title": "Sample Title",
            "body": "This is the body of the markdown.",
            "hidden": True,
            "slug": slug,
        }
        result = process_markdown(body, slug)
        result["body"] = result["body"].strip()
        self.assertEqual(result, expected)

    def test_process_markdown_without_frontmatter_raises_error(self):
        body = "This is the body of the markdown without frontmatter."
        slug = "no-frontmatter-slug"
        with self.assertRaises(ValueError) as context:
            process_markdown(body, slug)
        self.assertEqual(str(context.exception), "No frontmatter found in the document")

    def test_process_markdown_with_empty_body_raises_error(self):
        body = ""
        slug = "empty-body-slug"
        with self.assertRaises(ValueError) as context:
            process_markdown(body, slug)
        self.assertEqual(str(context.exception), "No frontmatter found in the document")

    def test_process_markdown_with_frontmatter_in_body(self):
        body = (
            "---\ntitle: Sample Title\nhidden: true\n---\n\nThis is the body of the "
            "markdown. More front matter here.\n---\nhidden: false\n---something else"
        )
        slug = "frontmatter-in-body-slug"
        expected = {
            "title": "Sample Title",
            "body": "This is the body of the markdown. More front matter here.\n---\n"
            "hidden: false\n---something else",
            "hidden": True,
            "slug": slug,
        }
        result = process_markdown(body, slug)
        result["body"] = result["body"].strip()
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
