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
from typing import TYPE_CHECKING, Any, TypedDict
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

if TYPE_CHECKING:
    from collections.abc import Mapping

REQUEST_TIMEOUT_SECONDS = 60

# Retry a request up to three times after the initial attempt.
MAX_RETRIES = 3

# Wait progressively longer between retries so temporary API or network
# problems have time to recover.
RETRY_BACKOFF_FACTOR = 3

# Retry rate limiting and temporary server failures.
#
# Other client errors, such as 400, 401, 403, 404, and 409, normally mean
# the request must be corrected and should not be automatically retried.
RETRY_STATUS_CODES = (429, 500, 502, 503, 504)

logger = logging.getLogger(__name__)


class ReadMeResponse(TypedDict):
    """Expected structure shared by ReadMe API responses."""

    data: object


def _create_retry_session() -> requests.Session:
    """Create an HTTP session configured to retry failures.

    Returns:
        A session that retries requests after failures.
    """
    retry = Retry(
        # Retry up to three times after the initial request.
        total=MAX_RETRIES,
        # Increase the delay between retries instead of repeatedly sending
        # requests to an API that may be temporarily unavailable.
        backoff_factor=RETRY_BACKOFF_FACTOR,
        # Retry rate limiting and temporary server failures.
        status_forcelist=RETRY_STATUS_CODES,
        # Retry every HTTP method used by this module.
        allowed_methods=frozenset({"DELETE", "GET", "PATCH", "POST"}),
        # Follow ReadMe's Retry-After header when it tells us how long to
        # wait before sending another request.
        respect_retry_after_header=True,
        # Return the final unsuccessful response after all retries fail.
        #
        # This lets the existing code below raise its detailed RuntimeError
        # containing the response status code and body.
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()

    # Install the retry behavior for both HTTP and HTTPS URLs.
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


# Reuse one session for every request so all request functions have the same
# retry behavior. A session also reuses its underlying HTTP connections.
_SESSION = _create_retry_session()


def _auth_headers(
    headers: dict[str, str] | None = None,
) -> dict[str, str]:
    """Add authorization to the supplied request headers.

    Args:
        headers: Optional mutable request headers supplied by the caller.

    Returns:
        The supplied headers with authorization added, or a new dictionary
        when no headers are supplied.
    """
    if headers is None:
        headers = {}

    headers["Authorization"] = f"Bearer {os.getenv('README_API_KEY')}"
    return headers


def _unwrap_object(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return an object stored in a response envelope.

    Args:
        payload: The decoded JSON response object.

    Returns:
        The object stored under the response's ``data`` field.

    Raises:
        ValueError: If the response does not contain a ``data`` field.
        TypeError: If the response's ``data`` field is not an object.
    """
    if "data" not in payload:
        raise ValueError("ReadMe response is missing the required 'data' field")

    data = payload["data"]

    # JSON objects are decoded into Python dictionaries.
    if not isinstance(data, dict):
        raise TypeError(
            "Expected ReadMe response data to be an object, "
            f"received {type(data).__name__}"
        )

    return data


def _unwrap_list(payload: Mapping[str, Any]) -> list[Any]:
    """Return a list stored in a response envelope.

    Args:
        payload: The decoded JSON response object.

    Returns:
        The list stored under the response's ``data`` field.

    Raises:
        ValueError: If the response does not contain a ``data`` field.
        TypeError: If the response's ``data`` field is not a list.
    """
    if "data" not in payload:
        raise ValueError("ReadMe response is missing the required 'data' field")

    data = payload["data"]

    # JSON arrays are decoded into Python lists.
    if not isinstance(data, list):
        raise TypeError(
            "Expected ReadMe response data to be a list, "
            f"received {type(data).__name__}"
        )

    return data


def get(
    url: str,
    headers: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    headers = _auth_headers(headers)
    response = _SESSION.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
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
    payload: ReadMeResponse = response.json()
    return _unwrap_object(payload)


def get_collection(
    url: str,
    headers: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve every page from a paginated collection endpoint.

    Args:
        url: The initial collection endpoint URL.
        headers: Optional additional HTTP request headers.

    Returns:
        A flat list containing the resources from every response page.

    Raises:
        RuntimeError: If ReadMe returns an unsuccessful HTTP response.
    """
    headers = _auth_headers(headers)
    items = []
    next_url = url

    while next_url:
        response = _SESSION.get(
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
        data = _unwrap_list(payload)

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
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    headers = _auth_headers(headers)
    response = _SESSION.post(
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

    payload: ReadMeResponse = response.json()
    return _unwrap_object(payload)


def patch(
    url: str,
    data: Mapping[str, Any],
    headers: dict[str, str] | None = None,
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
    response = _SESSION.patch(
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
    headers: dict[str, str] | None = None,
) -> None:
    headers = _auth_headers(headers)

    response = _SESSION.delete(
        url,
        headers=headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    logger.debug("delete %s %s", url, response.status_code)

    if response.status_code >= 400:
        raise RuntimeError(
            f"DELETE {url} failed with {response.status_code}: {response.text}"
        )
