# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import concurrent.futures
import logging
import os
import re
import sys
import timeit

import requests

from tools.github_readme_sync.colors import CYAN, GREEN, RED, RESET, WHITE, YELLOW
from tools.github_readme_sync.constants import (
    IGNORE_DOCS,
    IGNORE_EXTERNAL_URLS,
    IGNORE_IMAGES,
    IGNORE_TABLES,
    REGEX_CSV_TABLE,
)
from tools.github_readme_sync.file import find_markdown_files, read_file_content

HIERARCHY_FILE = "hierarchy.md"
CATEGORY_PREFIX = "# "
DOCUMENT_PREFIX = "- "
INDENTATION_UNIT = "  "  # Single indentation level

# URLs that are checked
README_URL = "https://thousandbrainsproject.readme.io"


def create_hierarchy_file(output_dir, hierarchy):
    with open(os.path.join(output_dir, HIERARCHY_FILE), "w") as f:
        for category in hierarchy:
            write_category(f, category, 0)
    logging.info(f"{GREEN}Export complete{RESET}")


def write_category(file, category, indent_level):
    """Write the category title with appropriate prefix."""
    category_slug = category["slug"]
    title = category["title"]
    file.write(f"{CATEGORY_PREFIX}{category_slug}: {title}\n")

    for doc in category.get("children", []):
        write_document(file, category_slug, doc, indent_level)

    # newline
    file.write("\n")


def write_document(file, parent_slug, doc, indent_level):
    """Write the document entry with appropriate prefix and indentation."""
    doc_slug = doc["slug"]
    doc_path = f"{parent_slug}/{doc_slug}.md"
    indentation = INDENTATION_UNIT * indent_level
    file.write(f"{indentation}{DOCUMENT_PREFIX}[{doc_slug}]({doc_path})\n")

    for child in doc.get("children", []):
        write_document(file, f"{parent_slug}/{doc_slug}", child, indent_level + 1)


def check_hierarchy_file(folder: str):
    hierarchy = []

    if not os.path.exists(os.path.join(folder, HIERARCHY_FILE)):
        logging.error(f"File {os.path.join(folder, HIERARCHY_FILE)} does not exist")
        sys.exit(1)

    with open(os.path.join(folder, HIERARCHY_FILE)) as f:
        content = f.read()
        content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
        lines = content.splitlines()

    parent_stack = []
    current_category = None
    unique_slugs = {}
    link_check_errors = []

    for line in lines:
        if line.startswith(CATEGORY_PREFIX):
            slug, title = map(str.strip, line[len(CATEGORY_PREFIX) :].split(":"))
            current_category = {"title": title, "slug": slug, "children": []}
            hierarchy.append(current_category)
            parent_stack = [current_category]

        elif DOCUMENT_PREFIX in line:
            indent_level = (len(line) - len(line.lstrip(INDENTATION_UNIT))) // len(
                INDENTATION_UNIT
            )
            slug = extract_slug(line.strip())

            if slug in unique_slugs:
                logging.error(
                    f"Duplicate slug found: {slug}"
                    f"\n{unique_slugs[slug].strip()}\n{line.strip()}"
                )
                sys.exit(1)
            unique_slugs[slug] = line

            new_doc = {"slug": slug, "children": []}
            parent_stack = parent_stack[: indent_level + 1]
            parent_stack[-1]["children"].append(new_doc)
            parent_stack.append(new_doc)

            full_path = (
                os.path.join(folder, *(el["slug"] for el in parent_stack)) + ".md"
            )
            errors = sanity_check(full_path)
            if errors:
                link_check_errors.extend(errors)

    if link_check_errors:
        for error in link_check_errors:
            logging.error(error)
        sys.exit(1)

    logging.info(f"{GREEN}Hierarchy check complete{RESET}")
    return hierarchy


def extract_slug(line: str):
    regex = r"\[(.*)\]\(.*\)"
    match = re.search(regex, line)
    return match.group(1)


def sanity_check(path):
    if not os.path.exists(path):
        return [f"File {path} does not exist"]

    return check_links(path)


def check_links(path):
    with open(path) as f:
        content = f.read()
    file_name = path.split("/")[-1]

    regex_md_links = r"\[([^\]]*)\]\(([^)]+\.md(?:#[^)]*)?)\)"
    md_link_matches = re.findall(regex_md_links, content)

    table_matches = re.findall(REGEX_CSV_TABLE, content)

    regex_figures = (
        r"(?:\.\./)*figures/[^\s\)\"\']+(?:\.png|\.jpg|\.jpeg|\.gif|\.svg|\.webp|\s)"
    )
    image_link_matches = re.findall(regex_figures, content)
    logging.debug(
        f"{WHITE}{file_name}"
        f"{GREEN} {len(md_link_matches)} links"
        f"{CYAN} {len(image_link_matches)} images"
        f"{YELLOW} {len(table_matches)} tables{RESET}"
    )

    current_dir = os.path.dirname(path)
    errors = []

    for match in table_matches:
        table_name = os.path.basename(match)
        if table_name in IGNORE_TABLES:
            continue

        path_to_check = os.path.join(current_dir, match)
        path_to_check = os.path.normpath(path_to_check)
        if not os.path.exists(path_to_check):
            errors.append(f"  CSV {match} does not exist")

    for match in md_link_matches:
        if match[1].startswith(("http://", "https://", "mailto:")):
            continue

        path_to_check = os.path.join(current_dir, match[1].split("#")[0])
        path_to_check = os.path.normpath(path_to_check)
        if any(placeholder in match[1] for placeholder in IGNORE_DOCS):
            continue
        logging.debug(f"{GREEN}  {path_to_check.split('/')[-1]}{RESET}")
        if not os.path.exists(path_to_check):
            errors.append(f"  Linked {match[1]} does not exist")

    for match in image_link_matches:
        path_to_check = os.path.join(current_dir, match.split("#")[0])
        path_to_check = os.path.normpath(path_to_check)
        if any(placeholder in match for placeholder in IGNORE_IMAGES):
            continue
        logging.debug(f"{CYAN}  {path_to_check.split('/')[-1]}{RESET}")
        if not os.path.exists(path_to_check):
            errors.append(f"  Image {path_to_check} does not exist")

    if errors:
        path_error = f"Errors in file {RED}{path}{RESET}:"
        errors.insert(0, path_error)
        return errors

    return []


def check_external(folder, ignore_dirs, rdme):
    errors = {}
    total_links_checked = 0
    url_cache = {}  # Cache to store URL check results

    md_files = find_markdown_files(folder, ignore_dirs=ignore_dirs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_file = {
            executor.submit(process_file, file_path, rdme, url_cache): file_path
            for file_path in md_files
        }
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                file_errors, links_checked = future.result()
                total_links_checked += links_checked
                if file_errors:
                    errors[file_path] = file_errors
            except Exception:
                logging.exception(f"{RED}Error processing {file_path}: {RESET}")

    report_errors(errors, total_links_checked)


def process_file(file_path, rdme, url_cache):
    logging.debug(f"{WHITE}{file_path}{RESET}")
    file_errors = []
    content = read_file_content(file_path)
    all_links = extract_external_links(content)
    links_checked = 0

    for url in all_links:
        if url in url_cache:
            logging.debug("cache hit %s", url)
            if url_cache[url]:
                file_errors.extend(url_cache[url])
            links_checked += 1
        elif is_readme_url(url):
            result = check_readme_link(url, rdme)
            url_cache[url] = result
            file_errors.extend(result)
            links_checked += 1
        elif is_external_url(url):
            result = check_external_link(url)
            url_cache[url] = result
            file_errors.extend(result)
            links_checked += 1
        else:
            # Do nothing as it's an ignored link
            pass

    return file_errors, links_checked


def extract_external_links(content):
    return (
        re.findall(r"\[[^\]]*\]\(([^)]+)\)", content)
        + re.findall(r'<a[^>]+href="([^"]+)"', content)
        + re.findall(r"<a[^>]+href=\'([^\']+)\'", content)
        + re.findall(r'<img[^>]+src="([^"]+)"', content)
        + re.findall(r"<img[^>]+src=\'([^\']+)\'", content)
    )


def is_readme_url(url):
    return url.startswith(README_URL)


def is_external_url(url):
    return url.startswith(("http://", "https://"))


def check_readme_link(url, rdme):
    if url == f"{README_URL}/":
        return []

    try:
        doc_slug = url.split("/")[-1]
        time = timeit.default_timer()
        response = rdme.get_doc_by_slug(doc_slug)
        time = timeit.default_timer() - time
        status_color = GREEN if response else RED
        log_msg = f"{CYAN}{url} {status_color}"
        log_msg += f"[{200 if response else 404}]{RESET}"
        if time > 1:
            log_msg += f" ({YELLOW}{time:.2f}s{RESET})"
        logging.info(log_msg)
        if not response:
            return [f"  broken link: {url} (Not found)"]
    except Exception as e:  # noqa: BLE001
        return [f"  {url}: {str(e)}"]

    return []


def check_external_link(url):
    if any(ignored_url in url for ignored_url in IGNORE_EXTERNAL_URLS):
        logging.info(f"{WHITE}{url} {GREEN}[IGNORED]{RESET}")
        return []

    try:
        time = timeit.default_timer()
        response = check_url(url)
        time = timeit.default_timer() - time

        status_color = GREEN if 200 <= response.status_code <= 299 else RED
        log_msg = f"{WHITE}{url} {status_color}[{response.status_code}]{RESET}"
        if time > 1:
            log_msg += f" ({YELLOW}{time:.2f}s{RESET})"
        logging.info(log_msg)
        if response.status_code < 200 or response.status_code > 299:
            return [f"  broken link: {url} ({response.status_code})"]
    except requests.RequestException as e:
        return [f"  {url}: {str(e)}"]

    return []


def check_url(url) -> requests.Response:
    """Check if the URL exists.

    The cache-control was just in-case.
    The User Agent was needed to stop a 406 from
    ycbbenchmarks.com. I think it will work with
    any user-agent, but I figured a realistic one
    was a bit more future proof.

    Returns:
        The response from the URL request.
    """
    headers = request_headers()

    try:
        response = requests.head(url, timeout=5, headers=headers)
    except requests.RequestException:
        # If HEAD fails, try GET instead
        response = requests.get(url, timeout=5, headers=headers)
    else:
        # If HEAD succeeds but returns non-2xx, try GET
        if not (200 <= response.status_code <= 299):
            response = requests.get(url, timeout=5, headers=headers)

    return response


def request_headers() -> dict:
    """Populate the headers for the request.

    The cache-control was just in-case.
    The User Agent was needed to stop a 406 from
    ycbbenchmarks.com. I think it will work with
    any user-agent, but I figured a realistic one
    was a bit more future proof.

    Returns:
        A dictionary containing the request headers.
    """
    return {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/129.0.0.0 Safari/537.36",
        "Cache-Control": "no-cache",  # just in-case
        "Pragma": "no-cache",
    }


def report_errors(errors, total_links_checked):
    logging.info("")
    if errors:
        for file_path, file_errors in errors.items():
            logging.error(f"{RED}{file_path}{RESET}")
            for error in file_errors:
                logging.error(error)
        sys.exit(1)
    else:
        logging.info(
            f"{GREEN}No external link errors found. "
            f"Total links checked: {total_links_checked}{RESET}"
        )
