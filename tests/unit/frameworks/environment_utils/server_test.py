# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from contextlib import contextmanager
from http.server import HTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory as TempDir
from threading import Thread

import pytest
import requests

from tbp.monty.frameworks.environment_utils.server import MontyRequestHandler


@contextmanager
def serve_request(server):
    server_thread = Thread(target=server.handle_request, daemon=True)
    server_thread.start()
    try:
        yield
    finally:
        server_thread.join(timeout=1)


@pytest.mark.parametrize("type_and_suffix", [("rgb", "png"), ("depth", "data")])
@pytest.mark.parametrize("query_params", [None, {"future": "proof"}])
def test_monty_request_handler(type_and_suffix, query_params):
    data_type, suffix = type_and_suffix

    inert_server = HTTPServer(("localhost", 0), MontyRequestHandler)
    port = inert_server.server_address[1]
    inert_server.timeout = 1  # seconds

    monkey_patcher = pytest.MonkeyPatch.context
    # TODO: After upgrading to Py3.9+, use parenthesized with statement for clarity
    with inert_server as server, monkey_patcher() as mp, TempDir() as temp_dir:
        # Create and assign the local directory location to write files to
        data_dir = Path(temp_dir, "worldimages", "world_data_stream")
        data_dir.mkdir(parents=True, exist_ok=True)
        mp.setenv("MONTY_DATA", temp_dir)

        # Confirm blank slate
        assert len(list(data_dir.glob(f"{data_type}_*.{suffix}"))) == 0

        # Send first PUT request
        file_size = 10  # bytes
        request_body = b"\x00" * file_size  # byte matches the file index

        with serve_request(server):
            result = requests.put(
                f"http://localhost:{port}/{data_type}.{suffix}",
                data=request_body,
                timeout=1,
                params=query_params,
            )
        assert result.status_code == 201
        assert result.text == f'Saved "{data_type}_0.{suffix}"\n'

        # Send second PUT request
        request_body = b"\x01" * file_size  # byte matches the file index
        with serve_request(server):
            result = requests.put(
                f"http://localhost:{port}/{data_type}.{suffix}",
                data=request_body,
                timeout=1,
                params=query_params,
            )
        assert result.status_code == 201
        assert result.text == f'Saved "{data_type}_1.{suffix}"\n'

        # Check that the files were created correctly
        assert len(list(data_dir.glob(f"{data_type}_*.{suffix}"))) == 2
        for idx, file in enumerate(sorted(data_dir.glob(f"{data_type}_*.{suffix}"))):
            assert file.name == f"{data_type}_{idx}.{suffix}"
            with file.open("rb") as f:
                content = f.read()
                assert len(content) == file_size
                # The files are full of bytes that equal the file index
                assert all(byte == idx for byte in content)
