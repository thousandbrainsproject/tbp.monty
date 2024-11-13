# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
import pathlib
import pickle

current_dir = pathlib.Path(__file__).parent
files = [file for file in os.listdir(current_dir) if file.endswith(".pkl")]
names = [file.split(".")[0] for file in files]

CONFIGS = dict()
for file, name in zip(files, names):
    with open(os.path.join(current_dir, file), "rb") as f:
        config = pickle.load(f)
        CONFIGS[name] = config
