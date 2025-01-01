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

import requests

from tools.github_readme_sync.colors import CYAN, GREEN, RED, RESET, WHITE
from tools.github_readme_sync.excluded_items import IGNORE_DOCS, IGNORE_IMAGES

HIERARCHY_FILE = "hierarchy.md"
CATEGORY_PREFIX = "# "
DOCUMENT_PREFIX = "- "
INDENTATION_UNIT = "  "  # Single indentation level


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

    with open(os.path.join(folder, HIERARCHY_FILE), "r") as f:
        lines = f.readlines()

    parent_stack = []
    current_category = None
    # unique_slugs a set of all slugs and corresponding line in a dict
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

            # Check for duplicate slugs
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

            # Verify file existence and sanity checks
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
    with open(path, "r") as f:
        content = f.read()
    file_name = path.split("/")[-1]

    regex_md_links = r"\[([^\]]*)\]\(([^)]+\.md(?:#[^)]*)?)\)"
    md_link_matches = re.findall(regex_md_links, content)

    regex_figures = (
        r"(?:\.\./)*figures/[^\s\)\"\']+(?:\.png|\.jpg|\.jpeg|\.gif|\.svg|\.webp|\s)"
    )
    image_link_matches = re.findall(regex_figures, content)
    logging.debug(
        f"{WHITE}{file_name}"
        f"{GREEN} {len(md_link_matches)} links"
        f"{CYAN} {len(image_link_matches)} images{RESET}"
    )

    current_dir = os.path.dirname(path)
    errors = []

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
        # Remove any #hash fragments from the path
        path = match.split("#")[0]
        path_to_check = os.path.join(current_dir, path)
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
    ignore_dirs.extend([".pytest_cache", ".github", ".git"])
    total_links_checked = 0
    url_cache = {}  # Cache to store URL check results

    md_files = []
    for root, _, files in os.walk(folder):
        if any(ignore_dir in root for ignore_dir in ignore_dirs):
            continue
        md_files.extend(
            [os.path.join(root, file) for file in files if file.endswith(".md")]
        )

    with concurrent.futures.ThreadPoolExecutor() as executor:
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
            except Exception as exc:
                logging.error(f"{RED}Error processing {file_path}: {exc}{RESET}")

    report_errors(errors, total_links_checked)


def process_file(file_path, rdme, url_cache):
    logging.debug(f"{WHITE}{file_path}{RESET}")
    file_errors = []
    content = read_file_content(file_path)
    all_links = extract_links(content)
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


def read_file_content(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_links(content):
    return (
        re.findall(r"\[[^\]]*\]\(([^)]+)\)", content)
        + re.findall(r'<a[^>]+href="([^"]+)"', content)
        + re.findall(r"<a[^>]+href=\'([^\']+)\'", content)
        + re.findall(r'<img[^>]+src="([^"]+)"', content)
        + re.findall(r"<img[^>]+src=\'([^\']+)\'", content)
    )


def is_readme_url(url):
    return url.startswith("https://thousandbrainsproject.readme.io")


def is_external_url(url):
    return url.startswith(("http://", "https://"))


def check_readme_link(url, rdme):
    if url == "https://thousandbrainsproject.readme.io/":
        return []

    try:
        doc_slug = url.split("/")[-1]
        response = rdme.get_doc_by_slug(doc_slug)
        if not response:
            logging.debug(f"{WHITE}  {url} (Not found){RESET}")
            return [f"  broken link: {url} (Not found)"]
    except Exception as e:
        return [f"  {url}: {str(e)}"]

    return []


def check_external_link(url):
    try:
        headers = request_headers()
        response = requests.get(url, timeout=5, headers=headers)
        if response.status_code < 200 or response.status_code > 299:
            logging.debug(f"{WHITE}  {url} ({response.status_code}){RESET}")
            return [f"  broken link: {url} ({response.status_code})"]
    except requests.RequestException as e:
        return [f"  {url}: {str(e)}"]

    return []


def request_headers():
    """Populate the headers for the request.

    The cache-control was just in-case.
    The User Agent was needed to stop a 406 from
    ycbbenchmarks.com. I think it will work with
    any user-agent, but I figured a realistic one
    was a bit more future proof.

    Returns:
        dict: A dictionary containing the request headers.
    """
    return {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/129.0.0.0 Safari/537.36",
        "Cache-Control": "no-cache",  # just in-case
        "Pragma": "no-cache",
    }


def report_errors(errors, total_links_checked):
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
