# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Tuple

# TODO: These are type aliases for now, but we should consider combining
#  these together into a unified concept for pose, perhaps using NumPy's
#  custom array containers.
VectorXYZ = Tuple[float, float, float]
EulerAnglesXYZ = Tuple[float, float, float]
QuaternionWXYZ = Tuple[float, float, float, float]

DEFAULT_TOLERANCE = 1e-6
