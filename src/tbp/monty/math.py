# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

# TODO: These are type aliases for now, but we should consider combining
#  these together into a unified concept for pose, perhaps using NumPy's
#  custom array containers.
VectorXYZ = tuple[float, float, float]
EulerAnglesXYZ = tuple[float, float, float]
QuaternionWXYZ = tuple[float, float, float, float]

IDENTITY_QUATERNION: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0)
ZERO_VECTOR: VectorXYZ = (0.0, 0.0, 0.0)
ONES_VECTOR: VectorXYZ = (1.0, 1.0, 1.0)

DEFAULT_TOLERANCE = 1e-6
ROTATION_TOLERANCE_RADIANS = 1e-6
