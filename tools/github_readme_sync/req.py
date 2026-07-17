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

import logging
import os
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

import requests

if TYPE_CHECKING:
    from collections.abc import Mapping

REQUEST_TIMEOUT_SECONDS = 60
logger = logging.getLogger(__name__)


def _auth_headers(
    headers: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build authenticated request headers without modifying the input.

    Args:
        headers: Optional request headers supplied by the caller.

    Returns:
        A new dictionary containing the caller's headers and authorization.
    """
    request_headers = dict(headers or {})
    request_headers["Authorization"] = f"Bearer {os.getenv('README_API_KEY')}"
    return request_headers


def _unwrap_data(payload: Mapping[str, Any]) -> Any:
    """Return the resource stored in a response envelope.

    Args:
        payload: The decoded JSON response object.

    Returns:
        The value stored under the response's ``data`` field.

    Raises:
        ValueError: If the response does not contain a ``data`` field.
    """
    if "data" not in payload:
        raise ValueError("ReadMe response is missing the required 'data' field")

    return payload["data"]


def get(
    url: str,
    headers: Mapping[str, str] | None = None,
) -> Any | None:
    headers = _auth_headers(headers)
    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    logger.debug("get %s %s", url, response.status_code)
    if response.status_code == 404:
        return None

    if response.status_code >= 400:
        # Only a 404 means "the resource does not exist."
        #
        # Other failures must stop the upload. Otherwise,
        # create_or_update_doc() interprets the failure as a missing page
        # and incorrectly creates a duplicate.
        raise RuntimeError(
            f"GET {url} failed with {response.status_code}: {response.text}"
        )

    return _unwrap_data(response.json())


def get_collection(
    url: str,
    headers: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve every page from a paginated collection endpoint.

    Args:
        url: The initial collection endpoint URL.
        headers: Optional additional HTTP request headers.

    Returns:
        A flat list containing the resources from every response page.

    Raises:
        RuntimeError: If ReadMe returns an unsuccessful HTTP response.
        TypeError: If the response's ``data`` field is not a list.
    """
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

        if response.status_code >= 400:
            # Do not return a partial collection. Some callers (cleanup code) use this
            # inventory to determine which documents should be deleted.
            raise RuntimeError(
                f"GET {next_url} failed with {response.status_code}: {response.text}"
            )

        payload = response.json()
        data = _unwrap_data(payload)

        # ReadMe collection responses should always contain a list
        # under "data".
        if not isinstance(data, list):
            raise TypeError(
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


def post(
    url: str,
    data: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
) -> Any:
    headers = _auth_headers(headers)
    response = requests.post(
        url, json=data, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS
    )
    logger.debug("post %s %s", url, response.status_code)
    if response.status_code >= 400:
        # Preserve the API response, especially when strict slug handling
        # produces a 409 Conflict.
        raise RuntimeError(
            f"POST {url} failed with {response.status_code}: {response.text}"
        )

    if not response.content:
        return {}
    return _unwrap_data(response.json())


def patch(
    url: str,
    data: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
) -> bool:
    """Update a resource.

    Args:
        url: The URL of the resource to update.
        data: The request body to send as JSON.
        headers: Optional additional request headers.

    Returns:
        ``True`` when the resource is updated successfully.

    Raises:
        RuntimeError: If the request returns a client or server error.
    """
    headers = _auth_headers(headers)
    response = requests.patch(
        url, json=data, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS
    )
    logger.debug("patch %s %s", url, response.status_code)
    if response.status_code >= 400:
        raise RuntimeError(
            f"PATCH {url} failed with {response.status_code}: {response.text}"
        )
    return True


def delete(
    url: str,
    headers: Mapping[str, str] | None = None,
) -> None:
    headers = _auth_headers(headers)

    response = requests.delete(
        url,
        headers=headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    logger.debug("delete %s %s", url, response.status_code)

    if response.status_code >= 400:
        raise RuntimeError(
            f"DELETE {url} failed with {response.status_code}: {response.text}"
        )
