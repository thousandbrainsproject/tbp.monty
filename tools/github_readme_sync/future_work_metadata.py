# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

FUTURE_WORK_SECTION = "future-work"
METADATA_DOC_URL = "/docs/future-work-widget-metadata"

FUTURE_WORK_METADATA_KEYS = frozenset(
    {
        "estimated-scope",
        "improved-metric",
        "output-type",
        "skills",
        "contributor",
        "status",
        "rfc",
    }
)

GITHUB_AVATAR_URL = "https://github.com"
GITHUB_USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9-]{0,38}$")

BADGE_STYLE = (
    "padding:2px 4px;border-radius:4px;font-size:0.85em;"
    "display:inline-block;margin:2px 4px 2px 0;"
)
DEFAULT_BADGE_STYLE = f"{BADGE_STYLE}background-color:#e8e8f0;color:#2f2b5c;"
SKILLS_BADGE_STYLE = DEFAULT_BADGE_STYLE
SCOPE_STYLES = {
    "small": f"{BADGE_STYLE}background-color:#f0f0f0;color:#666666;",
    "medium": f"{BADGE_STYLE}background-color:#00a0df;color:#ffffff;",
    "large": f"{BADGE_STYLE}background-color:#2f2b5c;color:#ffffff;",
    "unknown": DEFAULT_BADGE_STYLE,
}
STATUS_STYLES = {
    "open": f"{BADGE_STYLE}background-color:#cce5ff;color:#004085;",
    "scoping": f"{BADGE_STYLE}background-color:#9bd4f5;color:#004a70;",
    "scoped": f"{BADGE_STYLE}background-color:#00a0df;color:#ffffff;",
    "in-progress": f"{BADGE_STYLE}background-color:#2f2b5c;color:#ffffff;",
    "paused": f"{BADGE_STYLE}background-color:#e8d5f5;color:#5a3d7a;",
    "completed": f"{BADGE_STYLE}background-color:#d4edda;color:#155724;",
    "evergreen": f"{BADGE_STYLE}background-color:#004085;color:#ffffff;",
}
METADATA_CONTAINER_STYLE = (
    "border:1px solid #ddd;border-radius:8px;padding:12px 16px;"
    "margin-bottom:16px;background-color:#fafafa;"
)
METADATA_COLUMNS_STYLE = "display:flex;gap:32px;"
METADATA_COLUMN_STYLE = "flex:1;min-width:0;"
METADATA_FIELD_STYLE = "margin-bottom:8px;"

LEFT_COLUMN_KEYS = ("status", "estimated-scope", "output-type")
RIGHT_COLUMN_KEYS = ("skills", "improved-metric", "rfc")


def is_future_work_doc_path(file_path: str) -> bool:
    """Check whether a document path is within the future work docs section.

    Returns:
        True when the path is under the future work section.
    """
    return FUTURE_WORK_SECTION in Path(file_path).as_posix().split("/")


def _has_metadata_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(str(item).strip() for item in value)
    return True


def _split_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _badge(text: str, style: str = DEFAULT_BADGE_STYLE) -> str:
    return f'<span style="{style}">{html.escape(text)}</span>'


def _scope_badge(scope: str) -> str:
    scope_key = scope.strip().lower()
    return _badge(scope, SCOPE_STYLES.get(scope_key, DEFAULT_BADGE_STYLE))


def _status_badge(status: str) -> str:
    status_key = status.strip().lower()
    return _badge(status, STATUS_STYLES.get(status_key, DEFAULT_BADGE_STYLE))


def _label_cell(label: str, content: str) -> str:
    if not content:
        return ""
    return (
        f'<div style="{METADATA_FIELD_STYLE}">'
        f"<strong>{html.escape(label)}:</strong> {content}"
        f"</div>"
    )


def _badges_html(values: list[str], style: str = DEFAULT_BADGE_STYLE) -> str:
    return " ".join(_badge(value, style) for value in values)


def _status_cell_content(fields: dict[str, Any]) -> str:
    parts: list[str] = []
    if "status" in fields:
        parts.append(_status_badge(str(fields["status"])))

    avatars: list[str] = []
    if "contributor" in fields:
        for username in _split_values(fields["contributor"]):
            if GITHUB_USERNAME_PATTERN.match(username):
                avatar_url = f"{GITHUB_AVATAR_URL}/{html.escape(username)}.png"
                avatars.append(
                    f'<img src="{avatar_url}" alt="{html.escape(username)}" '
                    f'title="{html.escape(username)}" '
                    f'style="width:24px;height:24px;border-radius:50%;'
                    f'vertical-align:middle;margin-right:4px;" />'
                )

    if avatars:
        parts.append(" ".join(avatars))

    return " ".join(parts)


def _wrap_readme_html_block(html_content: str) -> str:
    return f"[block:html]\n{json.dumps({'html': html_content}, indent=2)}\n[/block]"


def _field_cell(key: str, fields: dict[str, Any]) -> str:
    if key == "status":
        content = _status_cell_content(fields)
        if not content:
            return ""
        return _label_cell("Status", content)

    if key not in fields:
        return ""

    if key == "estimated-scope":
        return _label_cell("Scope", _scope_badge(str(fields[key])))

    if key == "output-type":
        return _label_cell("Output", _badges_html(_split_values(fields[key])))

    if key == "skills":
        return _label_cell(
            "Skills",
            _badges_html(_split_values(fields[key]), SKILLS_BADGE_STYLE),
        )

    if key == "improved-metric":
        return _label_cell("Metric", _badges_html(_split_values(fields[key])))

    if key == "rfc":
        rfc = str(fields[key]).strip()
        if rfc.lower().startswith(("http://", "https://")):
            rfc_content = (
                f'<a href="{html.escape(rfc)}" target="_blank" '
                f'rel="noopener noreferrer">RFC</a>'
            )
        else:
            rfc_content = html.escape(rfc)
        return _label_cell("RFC", rfc_content)

    return ""


def _column_html(keys: tuple[str, ...], fields: dict[str, Any]) -> str:
    cells = "".join(_field_cell(key, fields) for key in keys)
    if not cells:
        return ""
    return f'<div style="{METADATA_COLUMN_STYLE}">{cells}</div>'


def render_future_work_metadata(doc: dict[str, Any]) -> str:
    """Render future work frontmatter fields as an HTML metadata block.

    Args:
        doc: Processed markdown document containing optional metadata fields.

    Returns:
        HTML metadata block, or an empty string when no metadata is present.
    """
    fields = {
        key: doc[key]
        for key in FUTURE_WORK_METADATA_KEYS
        if key in doc and _has_metadata_value(doc[key])
    }
    if not fields:
        return ""

    left_column = _column_html(LEFT_COLUMN_KEYS, fields)
    right_column = _column_html(RIGHT_COLUMN_KEYS, fields)
    if not left_column and not right_column:
        return ""

    footer = (
        '<div style="width:100%;margin-top:8px;font-size:0.9em;">'
        "For details on what these values mean, see "
        f'<a href="{METADATA_DOC_URL}">here</a>.'
        "</div>"
    )

    metadata_html = (
        f'<div style="{METADATA_CONTAINER_STYLE}">'
        f'<div style="{METADATA_COLUMNS_STYLE}">{left_column}{right_column}</div>'
        f"{footer}"
        f"</div>"
    )
    return _wrap_readme_html_block(metadata_html)
