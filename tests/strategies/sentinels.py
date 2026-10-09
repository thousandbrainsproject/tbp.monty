# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Any
from unittest.mock import sentinel

import hypothesis.strategies as st


@st.composite
def distinct(draw: st.DrawFn, prefix: str) -> list[Any]:
    ids = draw(st.lists(st.integers(min_value=0), unique=True))
    return [getattr(sentinel, f"{prefix}_{i}") for i in ids]
