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


@dataclass
class RuntimeContext:
    """Monty's runtime context.

    The RuntimeContext carries runtime-scoped values used throughout Monty.
    """

    rng: np.random.RandomState
    """Random number generator."""
