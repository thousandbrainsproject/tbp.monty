# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import tempfile

from tbp.monty.frameworks.run_env import setup_env

setup_env(monty_logs_dir_default=tempfile.mkdtemp())
