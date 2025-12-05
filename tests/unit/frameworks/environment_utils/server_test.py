# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from http.server import HTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory as TempDir
from threading import Thread

import pytest
import requests

from tbp.monty.frameworks.environment_utils.server import MontyRequestHandler


def test_monty_request_handler_put_depth():
    inert_server = HTTPServer(("localhost", 8888), MontyRequestHandler)
    inert_server.timeout = 1  # seconds
    monkey_patcher = pytest.MonkeyPatch.context
    # TODO: After upgrading to Py3.9+, use parenthesized with statement for clarity
    with inert_server as server, monkey_patcher() as mp, TempDir() as temp_dir:
        # Create and assign the local directory location to write files to
        data_dir = Path(temp_dir, "worldimages", "world_data_stream")
        data_dir.mkdir(parents=True, exist_ok=True)
        mp.setenv("MONTY_DATA", temp_dir)

        # Confirm blank slate
        assert len(list(data_dir.glob("depth_*.data"))) == 0

        # Launch server to serve 2 requests
        def serve_n_requests(n):
            for _ in range(n):
                server.handle_request()

        server_thread = Thread(target=serve_n_requests, args=(2,))
        server_thread.start()

        # Send two PUT requests to the server
        file_size = 10  # bytes
        request_body = b"\x00" * file_size  # byte matches the file index
        result = requests.put(
            "http://localhost:8888/depth.data", data=request_body, timeout=1
        )
        assert result.status_code == 201
        assert result.text == 'Saved "depth_0.data"\n'

        request_body = b"\x01" * file_size  # byte matches the file index
        result = requests.put(
            "http://localhost:8888/depth.data", data=request_body, timeout=1
        )
        assert result.status_code == 201
        assert result.text == 'Saved "depth_1.data"\n'

        # Clean up server thread
        server_thread.join()

        # Check that the files were created correctly
        assert len(list(data_dir.glob("depth_*.data"))) == 2
        for idx, file in enumerate(sorted(data_dir.glob("depth_*.data"))):
            assert file.name == f"depth_{idx}.data"
            with file.open("rb") as f:
                content = f.read()
                assert len(content) == file_size
                # The files are full of bytes that equal the file index
                assert all(byte == idx for byte in content)
