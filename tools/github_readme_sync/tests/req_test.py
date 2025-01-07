# Copyright 2025 Thousand Brains Project
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
from unittest.mock import MagicMock, patch

from tools.github_readme_sync.readme import ReadMe
from tools.github_readme_sync.req import delete, get, post, put


@patch.dict(os.environ, {"README_API_KEY": "test_api_key"})
class TestReq(unittest.TestCase):
    @patch("requests.get")
    def test_get_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"key": "value"}
        mock_get.return_value = mock_response

        url = "https://api.example.com/data"
        result = get(url)

        self.assertEqual(result, {"key": "value"})
        mock_get.assert_called_once_with(
            url, headers={"Authorization": "Basic test_api_key"}
        )

    @patch("requests.get")
    def test_get_404(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        url = "https://api.example.com/data"
        result = get(url)

        self.assertIsNone(result)
        mock_get.assert_called_once_with(
            url, headers={"Authorization": "Basic test_api_key"}
        )

    @patch("requests.get")
    def test_get_failure(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response

        url = "https://api.example.com/data"
        with self.assertLogs(level="ERROR") as log:
            result = get(url)

        self.assertIsNone(result)
        self.assertIn("Failed to get https://api.example.com/data", log.output[0])
        mock_get.assert_called_once_with(
            url, headers={"Authorization": "Basic test_api_key"}
        )

    @patch("requests.post")
    def test_post_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.text = "Created"
        mock_post.return_value = mock_response

        url = "https://api.example.com/data"
        data = {"key": "value"}
        result = post(url, data, {"version": "1.0"})

        self.assertEqual(result, "Created")
        mock_post.assert_called_once_with(
            url,
            json=data,
            headers={"Authorization": "Basic test_api_key", "version": "1.0"},
        )

    @patch("requests.post")
    def test_post_failure(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response

        url = "https://api.example.com/data"
        data = {"key": "value"}
        with self.assertLogs(level="ERROR") as log:
            result = post(url, data)

        self.assertIsNone(result)
        self.assertIn("Failed to post https://api.example.com/data", log.output[0])
        mock_post.assert_called_once_with(
            url, json=data, headers={"Authorization": "Basic test_api_key"}
        )

    @patch("requests.put")
    def test_put_success(self, mock_put):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_put.return_value = mock_response

        url = "https://api.example.com/data"
        data = {"key": "value"}
        result = put(url, data)

        self.assertTrue(result)
        mock_put.assert_called_once_with(
            url, json=data, headers={"Authorization": "Basic test_api_key"}
        )

    @patch("requests.put")
    def test_put_failure(self, mock_put):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_put.return_value = mock_response

        url = "https://api.example.com/data"
        data = {"key": "value"}
        with self.assertLogs(level="ERROR") as log:
            result = put(url, data)

        self.assertFalse(result)
        self.assertIn("Failed to put https://api.example.com/data", log.output[0])
        mock_put.assert_called_once_with(
            url, json=data, headers={"Authorization": "Basic test_api_key"}
        )

    @patch("requests.delete")
    def test_delete_success(self, mock_delete):
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_delete.return_value = mock_response

        url = "https://api.example.com/data"
        result = delete(url)

        self.assertTrue(result)
        mock_delete.assert_called_once_with(
            url, headers={"Authorization": "Basic test_api_key"}
        )

    @patch("requests.delete")
    def test_delete_failure(self, mock_delete):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_delete.return_value = mock_response

        url = "https://api.example.com/data"
        result = delete(url)

        self.assertFalse(result)
        mock_delete.assert_called_once_with(
            url, headers={"Authorization": "Basic test_api_key"}
        )

    @patch("requests.delete")
    def test_delete_version(self, mock_delete):
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_delete.return_value = mock_response

        url = "https://dash.readme.com/api/v1/version/v1.0.0"

        rdme = ReadMe("1.0.0")
        with self.assertLogs() as log:
            rdme.delete_version()

        self.assertIn("Successfully deleted version 1.0.0", log.output[0])
        mock_delete.assert_called_once_with(
            url, headers={"Authorization": "Basic test_api_key"}
        )


if __name__ == "__main__":
    unittest.main()
