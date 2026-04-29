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


def total_variation(img: np.ndarray) -> float:
    """Computer the sum of the absolute differences between adjacent pixels.

    Differences are computed across rows and across columns independently and
    then summed together to yield the total variation.

    Args:
        img: The 2D image to compute the total variation of.

    Returns:
        The total variation of the image.
    """
    return np.sum(np.abs(np.diff(img, axis=0))) + np.sum(
        np.abs(np.diff(img, axis=1))
    )
