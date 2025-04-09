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
from pathlib import Path


def get_folders(file_path: str) -> list:
    return [
        name
        for name in os.listdir(file_path)
        if Path(file_path).joinpath(name).is_dir()
    ]
