# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import unittest

from tools.github_readme_sync.future_work_metadata import (
    is_future_work_doc_path,
    render_future_work_metadata,
)

EXPECTED_ALL_FIELDS_RENDERED = (
    '<div style="border:1px solid #ddd;border-radius:8px;padding:12px 16px;'
    'margin-bottom:16px;overflow-wrap:break-word;">'
    '<div style="display:grid;grid-template-columns:repeat(auto-fit,'
    'minmax(min(300px,100%),1fr));gap:16px;">'
    '<div style="min-width:0;overflow-wrap:break-word;"><strong>Status:</strong> open '
    '<img src="https://github.com/vkakerbeck.png" alt="vkakerbeck" title="vkakerbeck" '
    'style="width:24px;height:24px;border-radius:50%;vertical-align:middle;'
    'margin-right:4px;" /></div>'
    '<div style="min-width:0;overflow-wrap:break-word;">'
    "<strong>Scope:</strong> large</div>"
    '<div style="min-width:0;overflow-wrap:break-word;">'
    "<strong>Output:</strong> prototype, monty-feature, PR</div>"
    '<div style="min-width:0;overflow-wrap:break-word;">'
    "<strong>Skills:</strong> python, research, monty</div>"
    '<div style="min-width:0;overflow-wrap:break-word;">'
    "<strong>Metric:</strong> learning</div>"
    '<div style="min-width:0;overflow-wrap:break-word;"><strong>RFC:</strong> '
    '<a href="https://github.com/thousandbrainsproject/tbp.monty/blob/main/rfcs/'
    '0015_future_work.md" target="_blank" rel="noopener noreferrer">RFC</a></div>'
    "</div>"
    '<div style="margin-top:8px;font-size:0.9em;overflow-wrap:break-word;">'
    "For details on what these values mean, see "
    '<a href="/docs/future-work-widget-metadata">here</a>.</div></div>'
)


class TestFutureWorkMetadata(unittest.TestCase):
    def test_is_future_work_doc_path(self):
        self.assertTrue(
            is_future_work_doc_path("docs/future-work/learning-module-improvements")
        )
        self.assertTrue(is_future_work_doc_path("docs/future-work"))
        self.assertFalse(is_future_work_doc_path("docs/contributing/documentation"))
        self.assertFalse(is_future_work_doc_path("docs/glossary.md"))

    def test_render_future_work_metadata_returns_empty_without_fields(self):
        result = render_future_work_metadata(
            {"title": "Example", "body": "Body text", "slug": "example"}
        )
        self.assertEqual(result, "")

    def test_render_future_work_metadata_includes_all_fields(self):
        doc = {
            "slug": "deal-with-incomplete-models",
            "estimated-scope": "large",
            "improved-metric": "learning",
            "output-type": "prototype, monty-feature, PR",
            "skills": "python, research, monty",
            "contributor": "vkakerbeck",
            "status": "open",
            "rfc": (
                "https://github.com/thousandbrainsproject/tbp.monty/blob/main/rfcs/"
                "0015_future_work.md"
            ),
        }

        self.assertEqual(render_future_work_metadata(doc), EXPECTED_ALL_FIELDS_RENDERED)
