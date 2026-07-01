# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import re
import unittest
from pathlib import Path

from tools.github_readme_sync.future_work_metadata import (
    DEFAULT_BADGE_STYLE,
    METADATA_DOC_URL,
    SCOPE_STYLES,
    STATUS_STYLES,
    is_future_work_doc_path,
    render_future_work_metadata,
)

WIDGET_CSS_PATH = (
    Path(__file__).resolve().parents[2]
    / "future_work_widget"
    / "app"
    / "css"
    / "future-work-widget.css"
)


def _normalize_hex(color: str) -> str:
    color = color.strip().lower()
    if re.fullmatch(r"#[0-9a-f]{3}", color):
        color = "#" + "".join(channel * 2 for channel in color[1:])
    return color


def _extract_colors(style: str) -> dict[str, str]:
    colors = {}
    for prop in ("background-color", "color"):
        match = re.search(rf"(?<![-\w]){re.escape(prop)}\s*:\s*([^;}}]+)", style)
        if match:
            colors[prop] = _normalize_hex(match.group(1))
    return colors


def _parse_css_rules(css_text: str) -> dict[str, dict[str, str]]:
    rules: dict[str, dict[str, str]] = {}
    for match in re.finditer(r"([^{}]+)\{([^}]*)\}", css_text):
        declarations = _extract_colors(match.group(2))
        if not declarations:
            continue
        for selector in match.group(1).split(","):
            rules[selector.strip()] = declarations
    return rules


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
            "rfc": "required",
        }

        result = render_future_work_metadata(doc)

        self.assertIn("Scope", result)
        self.assertIn("large", result)
        self.assertIn("Metric", result)
        self.assertIn("learning", result)
        self.assertIn("Output", result)
        self.assertIn("prototype", result)
        self.assertIn("Skills", result)
        self.assertIn("python", result)
        self.assertIn("Status", result)
        self.assertIn("open", result)
        self.assertIn("#cce5ff", result)
        self.assertIn("RFC", result)
        self.assertIn("required", result)
        self.assertIn("vkakerbeck.png", result)
        self.assertNotIn(">open</span><br><img", result)
        self.assertIn("display:flex", result)
        self.assertNotIn("<table", result)
        self.assertLess(result.index("Status"), result.index("Scope"))
        self.assertLess(result.index("Scope"), result.index("Output"))
        self.assertLess(result.index("Skills"), result.index("Metric"))
        self.assertLess(result.index("Metric"), result.index("RFC"))
        self.assertLess(result.index("Status"), result.index("Skills"))
        self.assertLess(result.index("Scope"), result.index("Metric"))
        self.assertLess(result.index("Output"), result.index("RFC"))
        self.assertIn(METADATA_DOC_URL, result)
        self.assertIn("[block:html]", result)
        self.assertIn("[/block]", result)

    def test_render_future_work_metadata_wraps_multiple_badges_inline(self):
        doc = {"output-type": "prototype, monty-feature, PR"}
        result = render_future_work_metadata(doc)

        self.assertIn(">prototype</span> <span", result)
        self.assertIn(">monty-feature</span> <span", result)
        self.assertNotIn(">prototype</span><br>", result)

    def test_render_future_work_metadata_status_colors(self):
        doc = {"status": "in-progress"}
        result = render_future_work_metadata(doc)

        self.assertIn("#2f2b5c", result)
        self.assertIn("in-progress", result)

    def test_render_future_work_metadata_links_http_rfc(self):
        doc = {
            "slug": "example",
            "rfc": "https://github.com/thousandbrainsproject/tbp.monty/blob/main/rfcs/0015_future_work.md",
        }

        result = render_future_work_metadata(doc)

        self.assertIn("0015_future_work.md", result)
        self.assertIn(">RFC</a>", result)

    def test_render_future_work_metadata_escapes_html(self):
        doc = {
            "slug": "example",
            "status": "<script>alert(1)</script>",
        }

        result = render_future_work_metadata(doc)

        self.assertNotIn("<script>", result)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", result)

    def test_badge_colors_match_widget_css(self):
        css_rules = _parse_css_rules(WIDGET_CSS_PATH.read_text(encoding="utf-8"))

        expected = [(".badge", DEFAULT_BADGE_STYLE)]
        expected += [
            (f".badge.badge-status-{status}", style)
            for status, style in STATUS_STYLES.items()
        ]
        expected += [
            (f".badge-size-{scope}", style)
            for scope, style in SCOPE_STYLES.items()
            if scope != "unknown"
        ]

        for selector, style in expected:
            self.assertIn(
                selector, css_rules, f"No CSS rule found for selector {selector}"
            )
            self.assertEqual(
                _extract_colors(style),
                css_rules[selector],
                f"Badge colors for {selector} are out of sync with the widget CSS",
            )


if __name__ == "__main__":
    unittest.main()
