# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import http.server
import os
import re

# This class is used for the monty meets world demo to live stream data from the iPad
# camera to a server that Monty can then read from.


class MontyRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_PUT(self):  # noqa: N802
        # Check type of incoming data: depth or rgb
        inc_filename = os.path.basename(self.path)
        data_type = "depth" if inc_filename == "depth.data" else "rgb"

        # check existing filenames in the directory
        data_path = os.path.expanduser("~/tbp/data/worldimages/world_data_stream")
        file_list = [f for f in os.listdir(data_path) if not f.startswith(".")]
        if data_type == "depth":
            match = [re.search("depth_(.*).data", file) for file in file_list]
            match_no_none = [m for m in match if m is not None]
            number_list = [int(m.group(1)) for m in match_no_none]
            next_number = max(number_list) + 1 if number_list else 0
            new_filename = "depth_" + str(next_number) + ".data"
        else:
            match = [re.search("rgb_(.*).png", file) for file in file_list]
            match_no_none = [m for m in match if m is not None]
            number_list = [int(m.group(1)) for m in match_no_none]
            next_number = max(number_list) + 1 if number_list else 0
            new_filename = "rgb_" + str(next_number) + ".png"

        # Write data to file
        file_length = int(self.headers["Content-Length"])
        with open(data_path + "/" + new_filename, "wb") as output_file:
            output_file.write(self.rfile.read(file_length))
            output_file.close()

        self.send_response(201, "Created")
        self.end_headers()

        # Return object id
        reply_body = 'Saved "%s"\n' % new_filename
        self.wfile.write(reply_body.encode("utf-8"))


if __name__ == "__main__":
    server = http.server.HTTPServer(("0.0.0.0", 8080), MontyRequestHandler)
    server.serve_forever()
