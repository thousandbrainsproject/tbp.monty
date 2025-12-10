# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import http.server
import logging
import os
import socketserver
import sys
import tempfile
from pathlib import Path

monty_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(monty_root))

from tools.future_work_widget.build import build  # noqa: E402
from tools.github_readme_sync.index import generate_index  # noqa: E402

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)

    os.chdir(monty_root)

    docs_dir = monty_root / "docs"
    output_dir = monty_root / "tools" / "future_work_widget" / "app"
    docs_snippets_dir = monty_root / "docs" / "snippets"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        index_file = Path(tmp.name)

    generate_index(str(docs_dir), index_file)
    result = build(index_file, output_dir, docs_snippets_dir)
    index_file.unlink()
    logger.info(result.model_dump_json(exclude_none=True, indent=2))

    if not result.success:
        sys.exit(1)

    port = 8080
    os.chdir(output_dir)

    logger.info("\nStarting HTTP server on port %d...", port)
    logger.info("Serving files from: %s", output_dir)
    logger.info("Open http://localhost:%d in your browser", port)
    logger.info("Press Ctrl+C to stop the server\n")

    with socketserver.TCPServer(
        ("", port), http.server.SimpleHTTPRequestHandler
    ) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    main()
