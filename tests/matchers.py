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


class ArrayEqual:
    def __init__(self, arr: npt.ArrayLike):
        self.arr = arr

    def __eq__(self, other: object):
        return np.array_equal(self.arr, np.asanyarray(other))

    def __hash__(self):
        return hash(np.asarray(self.arr).tobytes())
