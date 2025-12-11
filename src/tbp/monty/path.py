# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import os
from pathlib import Path


def monty_data_path(custom_data_path: str | Path, default_subpath: str | Path) -> Path:
    """Get data path, using custom path if provided, or return the default.

    Args:
        custom_data_path: Custom data path provided by user.
        default_subpath: Default subpath within MONTY_DATA to use if no custom path.

    Returns:
        Full data path.
    """
    if custom_data_path is None:
        return Path(os.environ["MONTY_DATA"]) / default_subpath

    return Path(custom_data_path)
