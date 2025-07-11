# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import pytest

# Adding this at the test `mujoco` module level means we don't have to remember to
# add it for every module beneath here.
pytest.importorskip(
    "mujoco",
    reason="MuJoCo optional dependency not installed.",
)
