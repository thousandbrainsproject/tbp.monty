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
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


@st.composite
def shape_not_n_by_3(draw: st.DrawFn) -> tuple[int, ...]:
    """Returns a shape that is not N by 3."""
    return draw(
        st.one_of(
            # 1D
            st.tuples(st.integers(min_value=0, max_value=10)),
            # 2D, not N by 3
            st.tuples(
                st.integers(min_value=0, max_value=10),
                st.one_of(
                    st.integers(min_value=0, max_value=2),
                    st.integers(min_value=4, max_value=10),
                ),
            ),
            # 3D+
            st.tuples(
                *(
                    st.integers(min_value=0, max_value=10)
                    for _ in range(draw(st.integers(min_value=3, max_value=10)))
                )
            ),
        )
    )


@st.composite
def shape_not_1d(draw: st.DrawFn) -> tuple[int, ...]:
    """Returns a shape that is not 1D."""
    return draw(
        st.tuples(
            *(
                st.integers(min_value=0, max_value=10)
                for _ in range(draw(st.integers(min_value=2, max_value=10)))
            )
        ),
    )


@st.composite
def float_array_not_n_by_3(draw: st.DrawFn) -> npt.NDArray[np.float64]:
    """Returns an array of locations that is not N by 3."""
    return draw(
        arrays(
            dtype=np.float64,
            shape=shape_not_n_by_3(),
            elements=st.just(0.0),
            fill=st.just(0.0),
        )
    )


@st.composite
def float_array_not_1d(draw: st.DrawFn) -> npt.NDArray[np.float64]:
    """Returns an array of weights that is not 1D."""
    return draw(
        arrays(
            dtype=np.float64,
            shape=shape_not_1d(),
            elements=st.just(0.0),
            fill=st.just(0.0),
        )
    )


@st.composite
def float_array_n_by_3(draw: st.DrawFn) -> npt.NDArray[np.float64]:
    """Returns an array of locations that is N by 3."""
    return draw(
        arrays(
            dtype=np.float64,
            shape=st.tuples(st.integers(min_value=0, max_value=10), st.just(3)),
            elements=st.floats(allow_nan=False, allow_infinity=False),
            fill=st.just(0.0),
        )
    )
