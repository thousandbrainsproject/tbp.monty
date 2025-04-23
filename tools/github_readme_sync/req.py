# Copyright 2025 Thousand Brains Project
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

import requests

REQUEST_TIMEOUT_SECONDS = 60


def get(url: str, headers=None):
    headers = headers or {}
    headers["Authorization"] = f"Basic {os.getenv('README_API_KEY')}"
    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    logging.debug("get %s %s", url, response.status_code)
    if response.status_code == 404:
        return None
    if response.status_code < 200 or response.status_code >= 300:
        logging.error(f"Failed to get {url} {response.text}")
        return None
    return response.json()


def post(url: str, data: dict, headers=None):
    headers = headers or {}
    headers["Authorization"] = f"Basic {os.getenv('README_API_KEY')}"
    response = requests.post(
        url, json=data, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS
    )
    logging.debug("post %s %s", url, response.status_code)
    if response.status_code < 200 or response.status_code >= 300:
        logging.error(f"Failed to post {url} {response.text}")
        return None
    return response.text


def put(url: str, data: dict, headers=None):
    headers = headers or {}
    headers["Authorization"] = f"Basic {os.getenv('README_API_KEY')}"
    response = requests.put(
        url, json=data, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS
    )
    logging.debug("put %s %s", url, response.status_code)
    if response.status_code < 200 or response.status_code >= 300:
        logging.error(f"Failed to put {url} {response.text}")
        return False
    return True


def delete(url: str, headers=None):
    headers = headers or {}
    headers["Authorization"] = f"Basic {os.getenv('README_API_KEY')}"
    response = requests.delete(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    logging.debug("delete %s %s", url, response.status_code)
    if response.status_code < 200 or response.status_code >= 300:
        return False
    return True
