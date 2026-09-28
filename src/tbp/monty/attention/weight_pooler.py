# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Protocol, Sequence


class WeightPooler(Protocol):
    def __call__(self, values: Sequence[float]) -> float: ...


def negative_priority_max_pool(values: Sequence[float]) -> float:
    """Inhibition-dominating weight pooler.

    Args:
        values: A sequence of float values.

    Returns:
        The maximum value if all are non-negative, otherwise the most negative value.
    """
    return min(values) if any(v < 0 for v in values) else max(values)
