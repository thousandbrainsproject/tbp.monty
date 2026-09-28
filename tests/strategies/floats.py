# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from hypothesis import strategies as st


@st.composite
def with_negative_floats(draw: st.DrawFn) -> list[float]:
    negative_floats = st.lists(
        st.floats(
            max_value=0.0, exclude_max=True, allow_nan=False, allow_infinity=False
        ),
        min_size=1,
    )
    any_floats = st.lists(st.floats(allow_nan=False, allow_infinity=False))
    return draw(negative_floats) + (draw(any_floats))
