# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from tbp.monty.frameworks.models.salience.strategies.protocol import SalienceStrategy


class UniformSalienceStrategy(SalienceStrategy):
    def __call__(
        self,
        rgba: npt.NDArray[np.int_],  # noqa: ARG002
        depth: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return np.ones_like(depth)
