# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import pathlib
import pickle

from tbp.monty.frameworks.run_env import setup_env

setup_env()

current_dir = pathlib.Path(__file__).parent

CONFIGS = {}
for path in current_dir.glob("*.pkl"):
    with path.open("rb") as f:
        config = pickle.load(f)
        name = path.stem
        CONFIGS[name] = config
