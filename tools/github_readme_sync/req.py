# Copyright 2025-2026 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging
import os
from urllib.parse import urljoin

import requests

REQUEST_TIMEOUT_SECONDS = 60
logger = logging.getLogger(__name__)


def _auth_headers(headers=None) -> dict:
    # v2 uses Bearer tokens.
    # Copy caller-provided headers so adding Authorization does not mutate them.
    headers = dict(headers or {})
    headers["Authorization"] = f"Bearer {os.getenv('README_API_KEY')}"
    return headers


def _unwrap_data(payload):
    # Single-resource and collection responses nest the payload under "data" in v2.
    # Return the inner value so callers never see the wrapper.
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def get(url: str, headers=None):
    headers = _auth_headers(headers)
    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    logger.debug("get %s %s", url, response.status_code)
    if response.status_code == 404:
        return None
    if response.status_code < 200 or response.status_code >= 300:
        # Only a 404 means "the resource does not exist."
        #
        # A 401, 403, 429, or 500 must stop the upload. Otherwise,
        # create_or_update_doc() interprets the failure as a missing page
        # and incorrectly creates a duplicate.
        raise RuntimeError(
            f"GET {url} failed with {response.status_code}: {response.text}"
        )
    return _unwrap_data(response.json())


def get_collection(url: str, headers=None) -> list:
    """Retrieve every page from a paginated v2 collection endpoint."""

    headers = _auth_headers(headers)
    items = []
    next_url = url

    while next_url:
        response = requests.get(
            next_url,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        logger.debug(
            "get_collection %s %s",
            next_url,
            response.status_code,
        )

        if response.status_code == 404:
            return items

        if response.status_code < 200 or response.status_code >= 300:
            # Do not return a partial collection. Some callers (cleanup code) use this
            # inventory to determine which documents should be deleted.
            raise RuntimeError(
                f"GET {next_url} failed with {response.status_code}: {response.text}"
            )

        payload = response.json()
        data = _unwrap_data(payload)

        # ReadMe v2 collection responses should always contain a list
        # under "data".
        if not isinstance(data, list):
            raise RuntimeError(
                f"Expected collection data from {next_url} to be a list, "
                f"received {type(data).__name__}"
            )

        items.extend(data)

        paging = payload.get("paging") if isinstance(payload, dict) else None
        next_path = paging.get("next") if isinstance(paging, dict) else None

        # Resolve the next-page link relative to the current response URL.
        # This works whether ReadMe returns an absolute or relative URL.
        next_url = urljoin(response.url, next_path) if next_path else None

    return items


def post(url: str, data: dict, headers=None):
    headers = _auth_headers(headers)
    response = requests.post(
        url, json=data, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS
    )
    logger.debug("post %s %s", url, response.status_code)
    if response.status_code < 200 or response.status_code >= 300:
        # Preserve the API response, especially when strict slug handling
        # produces a 409 Conflict.
        raise RuntimeError(
            f"POST {url} failed with {response.status_code}: {response.text}"
        )

    if not response.content:
        return {}
    return _unwrap_data(response.json())


def patch(url: str, data: dict, headers=None) -> bool:
    # v2 updates use PATCH (replaces put for guides/branches).
    headers = _auth_headers(headers)
    response = requests.patch(
        url, json=data, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS
    )
    logger.debug("patch %s %s", url, response.status_code)
    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError(
            f"PATCH {url} failed with {response.status_code}: {response.text}"
        )
    return True


def delete(url: str, headers=None) -> None:
    headers = _auth_headers(headers)

    response = requests.delete(
        url,
        headers=headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    logger.debug("delete %s %s", url, response.status_code)

    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError(
            f"DELETE {url} failed with {response.status_code}: {response.text}"
        )
