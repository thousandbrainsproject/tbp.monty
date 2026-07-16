# Copyright 2025-2026 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import csv
import html
import json
import logging
import os
import re
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote

import nh3
import yaml

from tools.github_readme_sync.colors import GRAY, GREEN, RESET
from tools.github_readme_sync.constants import (
    IGNORE_CLOUDINARY,
    IGNORE_DOCS,
    IGNORE_IMAGES,
    IGNORE_TABLES,
    IGNORE_YOUTUBE,
    REGEX_CSV_TABLE,
)
from tools.github_readme_sync.req import delete, get, get_collection, patch, post

logger = logging.getLogger(__name__)

API_PREFIX = "https://api.readme.com/v2"
GUIDES_SECTION = "guides"  # v2 category paths are /categories/{section}/…
GUIDES_SECTION_BODY = "guide"  # v2 POST body wants singular guide.
GITHUB_RAW = "https://raw.githubusercontent.com"

regex_images = re.compile(r"!\[(.*?)\]\((.*?)\)")
regex_image_path = re.compile(
    r"(\.\./){1,5}figures/((.+)\.(png|jpg|jpeg|gif|svg|webp))"
)
regex_markdown_path = re.compile(r"\(([\./]*)([\w\-/]+)\.md(#.*?)?\)")
regex_cloudinary_video = re.compile(
    r"^\s*\[(.*?)\]\((https://res\.cloudinary\.com/([^/]+)/video/upload/v(\d+)/([^/]+\.mp4))\)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
regex_youtube_link = re.compile(
    r"^\s*\[(.*?)\]\((?:https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})(?:[&?][^\)]*)?)\)\s*$",
    re.IGNORECASE | re.MULTILINE,
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

    def _is_hidden(self, resource: dict) -> bool:
        # v1 bool hidden → v2 privacy.view == anyone_with_link
        return (resource.get("privacy") or {}).get("view") == "anyone_with_link"

    def branch_url(self, suffix: str = "") -> str:
        # Every version-scoped call becomes …/branches/{self.version}{suffix}
        return f"{API_PREFIX}/branches/{self.version}{suffix}"

    def get_categories(self) -> list[Any]:
        """Return guide categories in the order supplied by ReadMe."""
        categories = get_collection(self.branch_url(f"/categories/{GUIDES_SECTION}"))
        if not categories:
            return []

        return categories

    def get_category_docs(self, category: Any) -> list[Any]:
        """Return a category's pages as a flat v2 collection."""
        # API v2 identifies categories by title rather than by the
        # category slug or hexadecimal ID used in API v1.
        title = quote(category["title"], safe="")

        response = get_collection(
            self.branch_url(f"/categories/{GUIDES_SECTION}/{title}/pages")
        )

        if not response:
            return []

        return response

    def get_category_doc_tree(self, category: Any) -> list[Any]:
        """Rebuild a nested page tree from API v2's flat page collection.

        Args:
            category: The ReadMe category whose pages should be retrieved.

        Returns:
            The category's pages organized as a nested tree. Each page contains
            a ``children`` list containing its direct child pages.

        Raises:
            ValueError: If a page has no URI, a duplicate URI is returned, or a
                page references a parent URI that is not present in the category.
        """
        pages = self.get_category_docs(category)

        if not pages:
            return []

        pages_by_uri = {}

        # First create independent page dictionaries with empty children lists.
        # We use resource URIs because v2 uses URIs as resource identifiers.
        for raw_page in pages:
            page = dict(raw_page)
            page["children"] = []

            uri = page.get("uri")
            if not uri:
                raise ValueError(f"ReadMe page {page.get('slug')!r} has no uri")

            if uri in pages_by_uri:
                raise ValueError(f"ReadMe returned duplicate page URI {uri!r}")

            pages_by_uri[uri] = page

        roots = []

        # Iterate over the original API order so sibling ordering remains
        # the same as the order returned by ReadMe.
        for original_page in pages:
            page = pages_by_uri[original_page["uri"]]
            parent_uri = (page.get("parent") or {}).get("uri")

            if not parent_uri:
                # A page without a parent is a category-level root page.
                roots.append(page)
                continue

            parent = pages_by_uri.get(parent_uri)
            if parent is None:
                # Continuing would silently flatten or corrupt the hierarchy.
                raise ValueError(
                    f"ReadMe page {page['slug']!r} refers to missing "
                    f"parent URI {parent_uri!r}"
                )

            parent["children"].append(page)

        return roots

    def get_doc_by_slug(self, slug: str) -> str:
        doc = self.get_doc(slug)

        if doc is None:
            raise DocumentNotFound(f"Document {slug} not found")

        front_matter = OrderedDict()
        front_matter["title"] = doc.get("title")

        if self._is_hidden(doc):
            front_matter["hidden"] = True

        # Include a fallback for the excerpt in case the content is None
        # which can happen if the document has no content.
        excerpt = (doc.get("content") or {}).get("excerpt")
        if excerpt:
            front_matter["description"] = excerpt

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

        # ReadMe may return null for a page that has no body.
        # Use an empty string so front matter can still be exported.
        body = (doc.get("content") or {}).get("body") or ""

        return front_matter_str + body

    def get_doc(self, slug: str) -> dict | None:
        """Return one guide and verify its slug and URI.

        Args:
            slug: The guide slug to retrieve.

        Returns:
            The guide returned by ReadMe, or ``None`` if ReadMe returns a 404.

        Raises:
            ValueError: If ReadMe resolves the request to a different slug or
                returns a guide without a URI.
        """
        doc = get(self.branch_url(f"/guides/{slug}"))

        if doc is None:
            # None can only mean an actual 404.
            return None

        returned_slug = doc.get("slug")
        uri = doc.get("uri")

        # Do not allow ReadMe to resolve an alias or another canonical slug
        # without the sync tool noticing.
        if returned_slug != slug:
            raise ValueError(
                f"ReadMe resolved requested slug {slug!r} to "
                f"{returned_slug!r} ({uri!r})"
            )

        if not uri:
            raise ValueError(f"ReadMe guide {slug!r} has no uri")

        return doc

    def make_version_stable(self):
        """Make a release version the project's stable/default version."""
        # Versions containing a suffix are preview versions, such as
        # 0.40-brothman-newtest3. Do not make those stable.
        if self.version_has_suffix():
            return

        logger.info(f"{GREEN}Setting version {self.version} to stable{RESET}")

        patch(
            f"{API_PREFIX}/branches/{self.version}",
            {
                # The live ReadMe v2 API and tested project behavior use
                # privacy.view="default" to make this version stable/default.
                # This differs from the public migration guide.
                "privacy": {"view": "default"},
            },
        )

    def version_has_suffix(self) -> bool:
        return "-" in self.version

    def create_version_if_not_exists(self) -> bool:
        if get(f"{API_PREFIX}/branches/{self.version}") is None:
            stable_version = self.get_stable_version()
            logger.info(
                f"{GRAY}Creating version: {self.version} "
                f"forked from {stable_version}{RESET}"
            )
            if not post(
                f"{API_PREFIX}/branches",
                {
                    "name": self.version,
                    "base": stable_version,
                    "privacy": {"view": "hidden"},
                },
            ):
                raise ValueError("Failed to create a new version")
            return True
        return False

    def delete_categories(self):
        logger.info(f"{GRAY}Deleting categories for version {self.version}{RESET}")
        categories = self.get_categories()
        for category in categories:
            self.delete_category(category["title"])

    def delete_category(self, title: str):
        logger.info(f"{GRAY}Deleting category {title}{RESET}")
        delete(self.branch_url(f"/categories/{GUIDES_SECTION}/{quote(title, safe='')}"))

    def delete_doc(self, slug: str):
        logger.info(f"{GRAY}Deleting doc {slug}{RESET}")
        delete(self.branch_url(f"/guides/{slug}"))

    def validate_csv_align_param(self, align_value: str) -> None:
        if align_value not in ["left", "right"]:
            raise ValueError(
                f"Invalid alignment value: {align_value}. Must be 'left' or 'right'"
            )

    def create_category_if_not_exists(self, title: str) -> tuple[str, bool]:
        category = get(
            self.branch_url(f"/categories/{GUIDES_SECTION}/{quote(title, safe='')}")
        )
        if category is None:
            created = post(
                self.branch_url("/categories"),
                {"title": title, "section": GUIDES_SECTION_BODY},
            )
            if created is None:
                raise ValueError(f"Failed to create category {title}")

            return created["uri"], True
        return category["uri"], False

    def convert_csv_to_html_table(self, body: str, file_path: str) -> str:
        """Convert CSV table references to HTML tables.

        Args:
            body: The document body containing CSV table references
            file_path: The path to the current document being processed

        Returns:
            The document body with CSV tables converted to HTML format.
        """

        def replace_match(match):
            csv_path = Path(match.group(1))
            table_name = csv_path.name
            if table_name in IGNORE_TABLES:
                return match.group(0)

            # Get absolute path of CSV relative to current document
            csv_path = Path(file_path) / csv_path
            csv_path = csv_path.resolve()

            try:
                with csv_path.open() as f:
                    reader = csv.reader(f)
                    headers = next(reader)
                    rows = list(reader)

                    # Build unsafe HTML table
                    unsafe_html = "<div class='data-table'><table>\n<thead>\n<tr>"

                    # Process headers and build alignment lookup
                    alignments = {}
                    hidden_columns = set()
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
                            elif part == "hidden":
                                hidden_columns.add(i)
                        if i not in hidden_columns:
                            unsafe_html += f"<th{title_attr}>{header}</th>"
                    unsafe_html += "</tr>\n</thead>\n<tbody>\n"

                    # Add rows using stored alignments
                    for row in rows:
                        unsafe_html += "<tr>"
                        for i, cell in enumerate(row):
                            if i in hidden_columns:
                                continue
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
        self,
        order: int,
        category_id: str,
        doc: dict,
        parent_id: str,
        file_path: str,
    ) -> tuple[str, bool]:
        # Convert the document body into the format expected by ReadMe.
        markdown = self.process_markdown(
            doc["body"],
            file_path,
            doc["slug"],
        )

        # This payload is used when updating an existing guide.
        #
        # We not include "slug" in this request as updating a doc
        # uses its slug in the patch URL.
        update_doc_request = {
            "title": doc["title"],
            "type": "basic",
            "content": {"body": markdown},
            "category": {"uri": category_id},
            "privacy": {
                "view": ("anyone_with_link" if doc.get("hidden", False) else "public")
            },
            "position": order,
        }

        # Include the parent URI when this guide is nested under another guide.
        if parent_id:
            update_doc_request["parent"] = {"uri": parent_id}

        # ReadMe v2 stores the document description as the content excerpt.
        if "description" in doc:
            update_doc_request["content"]["excerpt"] = doc["description"]

        existing_doc = self.get_doc(doc["slug"])

        if existing_doc is not None:
            # The guide already exists, so update it without sending its slug.
            patch(
                self.branch_url(f"/guides/{doc['slug']}"),
                update_doc_request,
            )

            # Return the existing guide URI for use as the parent of nested pages.
            return existing_doc["uri"], False

        # The guide does not exist, so create it using the requested slug.
        create_doc_request = dict(update_doc_request)
        create_doc_request["slug"] = doc["slug"]

        created = post(
            self.branch_url("/guides"),
            create_doc_request,
            # Prevent ReadMe from silently changing a duplicate slug into slug-1.
            # A slug collision will instead cause the request to fail.
            headers={"prefer": "handling=strict"},
        )

        actual_slug = created.get("slug")
        created_uri = created.get("uri")

        # Keep this protection even though strict handling should prevent
        # ReadMe from assigning a different slug.
        if actual_slug != doc["slug"]:
            raise ValueError(
                f"ReadMe created {doc['title']!r} with slug "
                f"{actual_slug!r}; expected {doc['slug']!r}"
            )

        if not created_uri:
            raise ValueError(f"Created doc {doc['title']!r} has no uri")

        return created_uri, True

    def process_markdown(self, body: str, file_path: str, slug: str) -> str:
        body = self.insert_edit_this_page(body, slug, file_path)
        body = self.insert_markdown_snippet(body, file_path)
        body = self.convert_csv_to_html_table(body, file_path)
        body = self.correct_image_locations(body)
        body = self.correct_file_locations(body)
        body = self.convert_note_tags(body)
        body = self.parse_images(body)
        body = self.convert_cloudinary_videos(body)
        return self.convert_youtube_videos(body)

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
        return body.replace(
            "!!LINK!!",
            f"https://github.com/thousandbrainsproject/tbp.monty/edit/main/{file_path}/{filename}.md",
        )

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
        return re.sub(regex_image_path, replace_image_path, new_body)

    def correct_file_locations(self, body: str) -> str:
        def replace_path(match):
            matched_text = match.group(0)

            if "check-links-ignore" in matched_text:
                return matched_text
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
        """Return the name of the project's stable/default version.

        Returns:
            The name of the branch currently identified by ReadMe as stable.

        Raises:
            ValueError: If no stable branch exists or the stable branch response
                does not contain a name.
        """
        # ReadMe v2 supports "stable" as an alias for the current
        # stable/default version.
        stable = get(f"{API_PREFIX}/branches/stable")
        if stable is None:
            raise ValueError("No stable version found")

        name = stable.get("name")
        if not name:
            raise ValueError("Stable version response did not contain a name")

        return name

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
                            logger.warning(f"Ignoring disallowed CSS property '{key}'")
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
        delete(f"{API_PREFIX}/branches/{self.version}")
        logger.info(f"{GREEN}Successfully deleted version {self.version}{RESET}")

    def _should_ignore_video(self, identifier: str, ignore_list: list[str]) -> bool:
        return identifier in ignore_list

    def _create_video_block(self, block_type: str, block_data: dict[str, Any]) -> str:
        return f"[block:{block_type}]\n{json.dumps(block_data, indent=2)}\n[/block]"

    def convert_cloudinary_videos(self, markdown_text: str) -> str:
        def replace_video(match):
            _, _, cloud_id, version, filename = match.groups()
            if self._should_ignore_video(filename, IGNORE_CLOUDINARY):
                return match.group(0)
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
            return self._create_video_block("html", block)

        return regex_cloudinary_video.sub(replace_video, markdown_text)

    def convert_youtube_videos(self, markdown_text: str) -> str:
        def replace_youtube(match):
            title, video_id = match.groups()
            if self._should_ignore_video(video_id, IGNORE_YOUTUBE):
                return match.group(0)
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
            embed_url = f"https://www.youtube.com/embed/{video_id}?feature=oembed"
            thumbnail_url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
            block = {
                "html": (
                    f'<iframe class="embedly-embed" '
                    f'src="//cdn.embedly.com/widgets/media.html?'
                    f"src={quote(embed_url, safe='')}&"
                    f"display_name=YouTube&"
                    f"url={quote(youtube_url, safe='')}&"
                    f"image={quote(thumbnail_url, safe='')}&"
                    f'type=text%2Fhtml&schema=youtube" '
                    f'width="854" height="480" scrolling="no" '
                    f'title="YouTube embed" frameborder="0" '
                    f'allow="autoplay; fullscreen; encrypted-media; '
                    f'picture-in-picture;" '
                    f'allowfullscreen="true"></iframe>'
                ),
                "url": youtube_url,
                "title": title,
                "favicon": "https://www.youtube.com/favicon.ico",
                "image": thumbnail_url,
                "provider": "https://www.youtube.com/",
                "href": youtube_url,
                "typeOfEmbed": "youtube",
            }
            return self._create_video_block("embed", block)

        return regex_youtube_link.sub(replace_youtube, markdown_text)

    def insert_markdown_snippet(self, body: str, file_path: str) -> str:
        """Insert markdown snippets from referenced files.

        Args:
            body: The document body containing snippet references
            file_path: The path to the current document being processed

        Returns:
            The document body with snippets inserted.
        """

        def replace_match(match):
            snippet_path = Path(file_path) / match.group(1)
            snippet_path = snippet_path.resolve()

            try:
                with snippet_path.open() as f:
                    unsafe_content = f.read()
                    return self.sanitize_html(unsafe_content)

            except Exception:  # noqa: BLE001
                return f"[File not found or could not be read: {snippet_path}]"

        return regex_markdown_snippet.sub(replace_match, body)


class DocumentNotFound(RuntimeError):
    """Raised when a document is not found."""

    pass
