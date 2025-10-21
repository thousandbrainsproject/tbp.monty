# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import csv
import html
import json
import logging
import os
import re
from collections import OrderedDict
from copy import deepcopy
from typing import Any, List, Tuple
from urllib.parse import parse_qs

import nh3
import yaml

from tools.github_readme_sync.colors import GRAY, GREEN, RESET
from tools.github_readme_sync.constants import (
    IGNORE_DOCS,
    IGNORE_IMAGES,
    IGNORE_TABLES,
    REGEX_CSV_TABLE,
)
from tools.github_readme_sync.req import delete, get, post, put

PREFIX = "https://dash.readme.com/api/v1"
GITHUB_RAW = "https://raw.githubusercontent.com"

regex_images = re.compile(r"!\[(.*?)\]\((.*?)\)")
regex_image_path = re.compile(
    r"(\.\./){1,5}figures/((.+)\.(png|jpg|jpeg|gif|svg|webp))"
)
regex_markdown_path = re.compile(r"\(([\./]*)([\w\-/]+)\.md(#.*?)?\)")
regex_cloudinary_video = re.compile(
    r"\[(.*?)\]\((https://res\.cloudinary\.com/([^/]+)/video/upload/v(\d+)/([^/]+\.mp4))\)",
    re.IGNORECASE,
)
regex_markdown_snippet = re.compile(r"!snippet\[(.*?)\]")

# Allowlist of supported CSS properties
ALLOWED_CSS_PROPERTIES = {"width", "height"}


class OrderedDumper(yaml.SafeDumper):
    pass


def _dict_representer(dumper, data):
    return dumper.represent_dict(data.items())


OrderedDumper.add_representer(OrderedDict, _dict_representer)


class ReadMe:
    def __init__(self, version: str):
        self.version = version

    def get_categories(self) -> List[Any]:
        categories = get(f"{PREFIX}/categories", {"x-readme-version": self.version})
        if not categories:
            return []
        return sorted(categories, key=lambda x: x["order"])

    def get_category_docs(self, category: Any) -> List[Any]:
        response = get(
            f"{PREFIX}/categories/{category['slug']}/docs",
            {"x-readme-version": self.version},
        )
        if not response:
            return []
        return sorted(response, key=lambda x: x["order"])

    def get_doc_by_slug(self, slug: str) -> str:
        response = get(f"{PREFIX}/docs/{slug}", {"x-readme-version": self.version})

        if not response:
            raise DocumentNotFound(f"Document {slug} not found")

        front_matter = OrderedDict()
        front_matter["title"] = response.get("title")

        if response.get("hidden"):
            front_matter["hidden"] = True

        if response.get("excerpt"):
            front_matter["description"] = response.get("excerpt")

        front_matter_str = (
            f"---\n"
            f"""{
                yaml.dump(
                    front_matter,
                    Dumper=OrderedDumper,
                    default_flow_style=False,
                    width=float("inf"),
                ).strip()
            }\n"""
            f"---\n"
        )

        doc_body = response.get("body")

        return front_matter_str + doc_body

    def get_doc_id(self, slug: str) -> str:
        response = get(f"{PREFIX}/docs/{slug}", {"x-readme-version": self.version})
        if response:
            return response["_id"]
        return None

    def make_version_stable(self):
        if self.version_has_suffix():
            return
        logging.info(f"{GREEN}Setting version {self.version} to stable{RESET}")
        if not put(
            f"{PREFIX}/version/{self.version}", {"is_stable": True, "is_hidden": False}
        ):
            raise ValueError("Failed to make version stable")

    def version_has_suffix(self) -> bool:
        return "-" in self.version

    def create_version_if_not_exists(self) -> bool:
        if get(f"{PREFIX}/version/{self.version}") is None:
            stable_version = self.get_stable_version()
            logging.info(
                f"{GRAY}Creating version: {self.version} "
                f"forked from {stable_version}{RESET}"
            )
            if not post(
                f"{PREFIX}/version",
                {
                    "version": self.version,
                    "from": stable_version,
                    "is_stable": False,
                    "is_hidden": True,
                },
            ):
                raise ValueError("Failed to create a new version")
            return True
        return False

    def delete_categories(self):
        logging.info(f"{GRAY}Deleting categories for version {self.version}{RESET}")
        categories = self.get_categories()
        for category in categories:
            self.delete_category(category["slug"])

    def delete_category(self, slug: str):
        logging.info(f"{GRAY}Deleting category {slug}{RESET}")
        delete(f"{PREFIX}/categories/{slug}", {"x-readme-version": self.version})

    def delete_doc(self, slug: str):
        logging.info(f"{GRAY}Deleting doc {slug}{RESET}")
        delete(f"{PREFIX}/docs/{slug}", {"x-readme-version": self.version})

    def validate_csv_align_param(self, align_value: str) -> None:
        if align_value not in ["left", "right"]:
            raise ValueError(
                f"Invalid alignment value: {align_value}. Must be 'left' or 'right'"
            )

    def create_category_if_not_exists(self, slug: str, title: str) -> Tuple[str, bool]:
        category = get(
            f"{PREFIX}/categories/{slug}", {"x-readme-version": self.version}
        )
        if category is None:
            response = post(
                f"{PREFIX}/categories",
                {"title": title, "type": "guide"},
                {"x-readme-version": self.version},
            )
            if response is None:
                raise ValueError(f"Failed to create category {title}")

            category = json.loads(response)
            return category["_id"], True

        return category["_id"], False

    def convert_csv_to_html_table(self, body: str, file_path: str) -> str:
        """Convert CSV table references to HTML tables.

        Args:
            body: The document body containing CSV table references
            file_path: The path to the current document being processed

        Returns:
            The document body with CSV tables converted to HTML format.
        """

        def replace_match(match):
            csv_path = match.group(1)
            table_name = os.path.basename(csv_path)
            if table_name in IGNORE_TABLES:
                return match.group(0)

            # Get absolute path of CSV relative to current document
            csv_path = os.path.join(file_path, csv_path)
            csv_path = os.path.normpath(csv_path)

            try:
                with open(csv_path) as f:
                    reader = csv.reader(f)
                    headers = next(reader)
                    rows = list(reader)

                    # Build unsafe HTML table
                    unsafe_html = "<div class='data-table'><table>\n<thead>\n<tr>"

                    # Process headers and build alignment lookup
                    alignments = {}
                    for i, unparsed_header in enumerate(headers):
                        title_attr = ""
                        align_style = ""
                        parts = [p.strip() for p in unparsed_header.split("|")]
                        header = parts[0]

                        # Process additional attributes in any order
                        for part in parts[1:]:
                            if part.startswith("hover "):
                                hover_text = html.escape(part[6:])
                                title_attr = f" title='{hover_text}'"
                            elif part.startswith("align "):
                                align_value = part[6:]
                                self.validate_csv_align_param(align_value)
                                alignments[i] = (
                                    f" style='text-align:{html.escape(align_value)}'"
                                )
                        unsafe_html += f"<th{title_attr}>{header}</th>"
                    unsafe_html += "</tr>\n</thead>\n<tbody>\n"

                    # Add rows using stored alignments
                    for row in rows:
                        unsafe_html += "<tr>"
                        for i, cell in enumerate(row):
                            align_style = alignments.get(i, "")
                            unsafe_html += f"<td{align_style}>{cell}</td>"
                        unsafe_html += "</tr>\n"

                    unsafe_html += "</tbody>\n</table></div>"

                    # Clean and return the HTML
                    return nh3.clean(
                        unsafe_html,
                        attributes={
                            "div": {"class"},
                            "th": {"title", "style"},
                            "td": {"style"},
                        },
                    )

            except Exception as e:  # noqa: BLE001
                return f"[Failed to load table from {csv_path} - {e}]"

        return REGEX_CSV_TABLE.sub(replace_match, body)

    def create_or_update_doc(
        self, order: int, category_id: str, doc: dict, parent_id: str, file_path: str
    ) -> Tuple[str, bool]:
        markdown = self.process_markdown(doc["body"], file_path, doc["slug"])

        create_doc_request = {
            "title": doc["title"],
            "type": "basic",
            "body": markdown,
            "category": category_id,
            "hidden": doc.get("hidden", False),
            "order": order,
            "parentDoc": parent_id,
        }

        # Include description field as excerpt if it exists in the document
        if "description" in doc:
            create_doc_request["excerpt"] = doc["description"]

        doc_id = self.get_doc_id(doc["slug"])
        created = doc_id is None
        if doc_id:
            if not put(
                f"{PREFIX}/docs/{doc['slug']}",
                create_doc_request,
                {"x-readme-version": self.version},
            ):
                raise ValueError(f"Failed to update doc {doc['title']}")
        else:
            response = post(
                f"{PREFIX}/docs", create_doc_request, {"x-readme-version": self.version}
            )
            if response is None:
                raise ValueError(f"Failed to create doc {doc['title']}")
            doc_id = json.loads(response)["_id"]

        return doc_id, created

    def process_markdown(self, body: str, file_path: str, slug: str) -> str:
        body = self.insert_edit_this_page(body, slug, file_path)
        body = self.insert_markdown_snippet(body, file_path)
        body = self.convert_csv_to_html_table(body, file_path)
        body = self.correct_image_locations(body)
        body = self.correct_file_locations(body)
        body = self.convert_note_tags(body)
        body = self.parse_images(body)
        body = self.convert_cloudinary_videos(body)
        return body

    def sanitize_html(self, body: str) -> str:
        allowed_attributes = deepcopy(nh3.ALLOWED_ATTRIBUTES)
        allowed_tags = deepcopy(nh3.ALLOWED_TAGS)

        allowed_tags.add("style")
        allowed_tags.add("a")
        allowed_tags.add("label")
        for tag in allowed_attributes:
            allowed_attributes[tag].add("width")
            allowed_attributes[tag].add("style")
            allowed_attributes[tag].add("target")
            allowed_attributes[tag].add("class")

        return nh3.clean(
            body,
            tags=allowed_tags,
            attributes=allowed_attributes,
            link_rel=None,
            strip_comments=False,
            generic_attribute_prefixes={"data-"},
            clean_content_tags={"script"},
        )

    def insert_edit_this_page(self, body: str, filename: str, file_path: str) -> str:
        depth = len(file_path.split("/")) - 1
        relative_path = "../" * depth
        relative_path = relative_path + "snippets/edit-this-page.md"
        body = body + f"\n\n!snippet[{relative_path}]"
        body = self.insert_markdown_snippet(body, file_path)
        body = body.replace(
            "!!LINK!!",
            f"https://github.com/thousandbrainsproject/tbp.monty/edit/main/{file_path}/{filename}.md",
        )
        return body

    def correct_image_locations(self, body: str) -> str:
        repo = os.getenv("IMAGE_PATH")
        if not repo:
            raise ValueError("IMAGE_PATH environment variable not set")
        new_body = body

        def replace_image_path(match):
            image_filename = match.group(2)
            # Ignore images that are in the ignore list
            if image_filename in IGNORE_IMAGES:
                return match.group(0)
            return f"{GITHUB_RAW}/{repo}/{image_filename}"

        # Find all image tags in the body
        img_tags = re.finditer(r'<img\s+[^>]*src="([^"]*)"[^>]*>', new_body)
        for match in img_tags:
            img_tag = match.group(0)
            src = match.group(1)
            # Only process if it's a relative path to figures
            if "../figures/" in src:
                image_path = re.search(regex_image_path, src)
                if image_path:
                    image_filename = image_path.group(2)
                    if image_filename not in IGNORE_IMAGES:
                        new_src = f"{GITHUB_RAW}/{repo}/{image_filename}"
                        new_img_tag = img_tag.replace(src, new_src)
                        new_body = new_body.replace(img_tag, new_img_tag)

        # Process regular markdown images
        new_body = re.sub(regex_image_path, replace_image_path, new_body)
        return new_body

    def correct_file_locations(self, body: str) -> str:
        def replace_path(match):
            matched_text = match.group(0)
            if any(placeholder in matched_text for placeholder in IGNORE_DOCS):
                return matched_text

            slug = match.group(2).split("/")[-1]
            fragment = match.group(3) or ""
            return f"(/docs/{slug}{fragment})"

        return re.sub(regex_markdown_path, replace_path, body)

    def convert_note_tags(self, body: str) -> str:
        conversions = {
            r"\[!NOTE\]": "📘",
            r"\[!TIP\]": "👍",
            r"\[!IMPORTANT\]": "📘",
            r"\[!WARNING\]": "🚧",
            r"\[!CAUTION\]": "❗️",
        }

        for old, new in conversions.items():
            body = re.sub(old, new, body)

        return body

    def get_stable_version(self) -> str:
        versions = get(f"{PREFIX}/version")
        if versions is None or not versions:
            raise ValueError("Failed to retrieve versions or no versions found")

        for version in versions:
            if version["is_stable"]:
                return version["version_clean"]

        raise ValueError("No stable version found")

    def parse_images(self, markdown_text: str) -> str:
        def replace_image(match):
            if any(ignore_image in match.groups()[1] for ignore_image in IGNORE_IMAGES):
                return match.group(0)
            alt_text, image_src = match.groups()

            # Split image source and fragment
            src_parts = image_src.split("#")
            clean_src = nh3.clean(src_parts[0])
            style = "border-radius: 8px;"

            # Parse and filter style parameters using allowlist
            if len(src_parts) > 1:
                try:
                    params = parse_qs(src_parts[1])
                    allowed_styles = []
                    for key, values in params.items():
                        if key in ALLOWED_CSS_PROPERTIES:
                            # Sanitize both key and value
                            safe_key = nh3.clean(key)
                            safe_value = nh3.clean(values[0])
                            allowed_styles.append(f"{safe_key}: {safe_value}")
                        else:
                            logging.warning(f"Ignoring disallowed CSS property '{key}'")
                    if allowed_styles:
                        style = f"{style} " + "; ".join(allowed_styles)
                except (ValueError, ImportError):
                    pass

            # Construct HTML with sanitized values
            if alt_text:
                unsafe_html = (
                    f'<figure><img src="{clean_src}" align="center"'
                    f' style="{style}" />'
                    f"<figcaption>{nh3.clean(alt_text)}</figcaption></figure>"
                )
            else:
                unsafe_html = (
                    f'<figure><img src="{clean_src}" align="center"'
                    f' style="{style}" /></figure>'
                )

            return nh3.clean(unsafe_html, attributes={"img": {"src", "align", "style"}})

        return regex_images.sub(replace_image, markdown_text)

    def delete_version(self):
        delete(f"{PREFIX}/version/v{self.version}")
        logging.info(f"{GREEN}Successfully deleted version {self.version}{RESET}")

    def convert_cloudinary_videos(self, markdown_text: str) -> str:
        def replace_video(match):
            title, full_url, cloud_id, version, filename = match.groups()
            # Replace the cloud ID with the environment variable
            new_url = f"https://res.cloudinary.com/{cloud_id}/video/upload/v{version}/{filename}"
            block = {
                "html": (
                    f'<div style="display: flex;justify-content: center;">'
                    f'<video width="640" height="360" '
                    f'style="border-radius: 10px;" controls '
                    f'poster="{new_url.replace(".mp4", ".jpg")}">'
                    f'<source src="{new_url}" type="video/mp4">'
                    f"Your browser does not support the video tag.</video></div>"
                )
            }
            return f"[block:html]\n{json.dumps(block, indent=2)}\n[/block]"

        return regex_cloudinary_video.sub(replace_video, markdown_text)

    def insert_markdown_snippet(self, body: str, file_path: str) -> str:
        """Insert markdown snippets from referenced files.

        Args:
            body: The document body containing snippet references
            file_path: The path to the current document being processed

        Returns:
            The document body with snippets inserted.
        """

        def replace_match(match):
            snippet_path = os.path.join(file_path, match.group(1))
            snippet_path = os.path.normpath(snippet_path)

            try:
                with open(snippet_path) as f:
                    unsafe_content = f.read()
                    return self.sanitize_html(unsafe_content)

            except Exception:  # noqa: BLE001
                return f"[File not found or could not be read: {snippet_path}]"

        return regex_markdown_snippet.sub(replace_match, body)


class DocumentNotFound(RuntimeError):
    """Raised when a document is not found."""

    pass
