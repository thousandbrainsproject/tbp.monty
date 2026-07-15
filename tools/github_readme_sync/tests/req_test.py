# Copyright 2025-2026 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
import unittest
from unittest.mock import MagicMock, call, patch

from tools.github_readme_sync.req import (
    REQUEST_TIMEOUT_SECONDS,
    delete,
    get,
    get_collection,
    patch as patch_request,
    post,
)


@patch.dict(os.environ, {"README_API_KEY": "test_api_key"})
class TestReq(unittest.TestCase):
    """Tests for the shared ReadMe API v2 request helpers."""

    @patch("tools.github_readme_sync.req.requests.get")
    def test_get_success_unwraps_v2_data(self, mock_get):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"data": {"slug": "test-doc"}}
        mock_get.return_value = response

        result = get("https://api.example.com/data")

        self.assertEqual(result, {"slug": "test-doc"})
        mock_get.assert_called_once_with(
            "https://api.example.com/data",
            headers={"Authorization": "Bearer test_api_key"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    @patch("tools.github_readme_sync.req.requests.get")
    def test_get_preserves_caller_headers(self, mock_get):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"data": {"name": "0.40"}}
        mock_get.return_value = response

        caller_headers = {"prefer": "handling=strict"}
        result = get("https://api.example.com/data", caller_headers)

        self.assertEqual(result, {"name": "0.40"})
        # The helper adds Bearer authentication without mutating the input.
        self.assertEqual(caller_headers, {"prefer": "handling=strict"})
        mock_get.assert_called_once_with(
            "https://api.example.com/data",
            headers={
                "prefer": "handling=strict",
                "Authorization": "Bearer test_api_key",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    @patch("tools.github_readme_sync.req.requests.get")
    def test_get_404_returns_none(self, mock_get):
        response = MagicMock()
        response.status_code = 404
        mock_get.return_value = response

        result = get("https://api.example.com/missing")

        self.assertIsNone(result)

    @patch("tools.github_readme_sync.req.requests.get")
    def test_get_non_404_failure_raises(self, mock_get):
        response = MagicMock()
        response.status_code = 500
        response.text = "Internal Server Error"
        mock_get.return_value = response

        with self.assertRaisesRegex(
            RuntimeError,
            r"GET https://api\.example\.com/data failed with 500",
        ):
            get("https://api.example.com/data")

    @patch("tools.github_readme_sync.req.requests.get")
    def test_get_collection_follows_v2_pagination(self, mock_get):
        first_response = MagicMock()
        first_response.status_code = 200
        first_response.url = "https://api.readme.com/v2/branches/0.40/guides"
        first_response.json.return_value = {
            "data": [{"slug": "doc-1"}],
            "paging": {"next": "/v2/branches/0.40/guides?page=2"},
        }

        second_response = MagicMock()
        second_response.status_code = 200
        second_response.url = "https://api.readme.com/v2/branches/0.40/guides?page=2"
        second_response.json.return_value = {
            "data": [{"slug": "doc-2"}],
            "paging": {"next": None},
        }

        mock_get.side_effect = [first_response, second_response]

        result = get_collection("https://api.readme.com/v2/branches/0.40/guides")

        self.assertEqual(result, [{"slug": "doc-1"}, {"slug": "doc-2"}])
        self.assertEqual(
            mock_get.call_args_list,
            [
                call(
                    "https://api.readme.com/v2/branches/0.40/guides",
                    headers={"Authorization": "Bearer test_api_key"},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                ),
                call(
                    "https://api.readme.com/v2/branches/0.40/guides?page=2",
                    headers={"Authorization": "Bearer test_api_key"},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                ),
            ],
        )

    @patch("tools.github_readme_sync.req.requests.get")
    def test_get_collection_404_returns_items_already_collected(self, mock_get):
        first_response = MagicMock()
        first_response.status_code = 200
        first_response.url = "https://api.readme.com/v2/items"
        first_response.json.return_value = {
            "data": [{"slug": "doc-1"}],
            "paging": {"next": "/v2/items?page=2"},
        }

        second_response = MagicMock()
        second_response.status_code = 404
        mock_get.side_effect = [first_response, second_response]

        result = get_collection("https://api.readme.com/v2/items")

        self.assertEqual(result, [{"slug": "doc-1"}])

    @patch("tools.github_readme_sync.req.requests.get")
    def test_get_collection_rejects_non_list_data(self, mock_get):
        response = MagicMock()
        response.status_code = 200
        response.url = "https://api.readme.com/v2/items"
        response.json.return_value = {"data": {"slug": "not-a-list"}}
        mock_get.return_value = response

        with self.assertRaisesRegex(
            RuntimeError,
            "Expected collection data .* to be a list",
        ):
            get_collection("https://api.readme.com/v2/items")

    @patch("tools.github_readme_sync.req.requests.get")
    def test_get_collection_failure_raises_instead_of_returning_partial_data(
        self, mock_get
    ):
        response = MagicMock()
        response.status_code = 429
        response.text = "Rate limited"
        mock_get.return_value = response

        with self.assertRaisesRegex(RuntimeError, "failed with 429"):
            get_collection("https://api.readme.com/v2/items")

    @patch("tools.github_readme_sync.req.requests.post")
    def test_post_success_unwraps_v2_data(self, mock_post):
        response = MagicMock()
        response.status_code = 201
        response.content = b'{"data": {"uri": "/guides/doc"}}'
        response.json.return_value = {"data": {"uri": "/guides/doc"}}
        mock_post.return_value = response

        result = post(
            "https://api.example.com/data",
            {"slug": "doc"},
            {"prefer": "handling=strict"},
        )

        self.assertEqual(result, {"uri": "/guides/doc"})
        mock_post.assert_called_once_with(
            "https://api.example.com/data",
            json={"slug": "doc"},
            headers={
                "prefer": "handling=strict",
                "Authorization": "Bearer test_api_key",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    @patch("tools.github_readme_sync.req.requests.post")
    def test_post_success_without_body_returns_empty_dict(self, mock_post):
        response = MagicMock()
        response.status_code = 204
        response.content = b""
        mock_post.return_value = response

        self.assertEqual(post("https://api.example.com/data", {}), {})

    @patch("tools.github_readme_sync.req.requests.post")
    def test_post_failure_raises_with_api_response(self, mock_post):
        response = MagicMock()
        response.status_code = 409
        response.text = "Slug already exists"
        mock_post.return_value = response

        with self.assertRaisesRegex(RuntimeError, "failed with 409"):
            post("https://api.example.com/data", {"slug": "doc"})

    @patch("tools.github_readme_sync.req.requests.patch")
    def test_patch_success(self, mock_patch):
        response = MagicMock()
        response.status_code = 200
        mock_patch.return_value = response

        result = patch_request(
            "https://api.example.com/data",
            {"title": "Updated"},
        )

        self.assertTrue(result)
        mock_patch.assert_called_once_with(
            "https://api.example.com/data",
            json={"title": "Updated"},
            headers={"Authorization": "Bearer test_api_key"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    @patch("tools.github_readme_sync.req.requests.patch")
    def test_patch_failure_raises(self, mock_patch):
        response = MagicMock()
        response.status_code = 400
        response.text = "Bad Request"
        mock_patch.return_value = response

        with self.assertRaisesRegex(RuntimeError, "failed with 400"):
            patch_request("https://api.example.com/data", {})

    @patch("tools.github_readme_sync.req.requests.delete")
    def test_delete_success_returns_none(self, mock_delete):
        response = MagicMock()
        response.status_code = 204
        mock_delete.return_value = response

        result = delete("https://api.example.com/data")

        # Successful DELETE calls now complete normally instead of returning True.
        self.assertIsNone(result)
        mock_delete.assert_called_once_with(
            "https://api.example.com/data",
            headers={"Authorization": "Bearer test_api_key"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    @patch("tools.github_readme_sync.req.requests.delete")
    def test_delete_failure_raises(self, mock_delete):
        response = MagicMock()
        response.status_code = 400
        response.text = "Bad Request"
        mock_delete.return_value = response

        with self.assertRaisesRegex(RuntimeError, "failed with 400"):
            delete("https://api.example.com/data")


if __name__ == "__main__":
    unittest.main()
