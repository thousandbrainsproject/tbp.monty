# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Protocol

import numpy as np
from typing_extensions import override


class SalienceStrategy(Protocol):
    def __call__(self, rgba: np.ndarray, depth: np.ndarray) -> np.ndarray: ...


class UniformSalienceStrategy(SalienceStrategy):
    @override
    def __call__(self, rgba: np.ndarray, depth: np.ndarray) -> np.ndarray:
        return np.ones_like(depth)
