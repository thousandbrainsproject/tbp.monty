# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from tbp.monty.frameworks.environments.environment import SemanticID
from tbp.monty.math import EulerAnglesXYZ, QuaternionWXYZ, VectorXYZ


@dataclass
class PrimaryTarget:
    euler_rotation: npt.NDArray[np.float64] | EulerAnglesXYZ
    object_name: str
    position: VectorXYZ
    rotation: QuaternionWXYZ
    scale: VectorXYZ
    quat_rotation: npt.NDArray[np.float64] | None = None
    semantic_id: SemanticID | None = None
