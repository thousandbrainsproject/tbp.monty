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

    Attributes:
        rng: The random number generator.
        suppress_runtime_errors: Whether to suppress runtime errors. Runtime errors
            can be raised when goal is None or invalid. When in an experimental
            mode, we want to raise runtime errors by default. When in a production
            mode, we want to suppress runtime errors by default. Currently, we run
            a lot of experiments, so the current default is to raise runtime errors.
    """

    rng: np.random.RandomState
    suppress_runtime_errors: bool = False
