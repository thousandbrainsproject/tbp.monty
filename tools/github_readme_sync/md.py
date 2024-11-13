# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import yaml


def process_markdown(body: str, slug: str) -> dict:
    doc = {"title": "", "body": "", "hidden": False, "slug": slug}
    frontmatter = parse_frontmatter(body)
    if not frontmatter:
        raise ValueError("No frontmatter found in the document")
    doc["title"] = frontmatter.get("title", "")
    doc["hidden"] = frontmatter.get("hidden", False)

    body = body.split("---\n", maxsplit=2)
    if len(body) > 2:
        doc["body"] = body[2]
    else:
        doc["body"] = body[0]

    return doc


def parse_frontmatter(file_content):
    if file_content.startswith("---"):
        _, frontmatter, _ = file_content.split("---", 2)
        return yaml.safe_load(frontmatter)
    return {}
