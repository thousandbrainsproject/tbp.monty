# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import cast

import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
import quaternion as qt

from tbp.monty.math import DEFAULT_TOLERANCE, VectorXYZ


@st.composite
def position(draw: st.DrawFn) -> VectorXYZ:
    x = draw(st.floats(min_value=-10.0, max_value=10.0))
    y = draw(st.floats(min_value=-10.0, max_value=10.0))
    z = draw(st.floats(min_value=-10.0, max_value=10.0))
    return x, y, z


@st.composite
def rotation(draw: st.DrawFn) -> npt.NDArray[np.double]:
    """Generates a random rotation as a normalized quaternion.

    Args:
        draw (st.DrawFn): The Hypothesis draw function.

    Returns:
        npt.NDArray[np.double]: A normalized quaternion representing a random rotation.
            np.double is used as that is what qt.from_float_array returns.
    """
    return cast(
        "npt.NDArray[np.double]",
        qt.from_float_array(
            np.array([draw(st.floats(-1, 1)) for _ in range(4)], dtype=np.double)
        ).normalized(),
    )


@st.composite
def rotation_non_zero(draw: st.DrawFn) -> npt.NDArray[np.double]:
    """Generates a random non-zero rotation as a normalized quaternion.

    Args:
        draw (st.DrawFn): The Hypothesis draw function.

    Returns:
        npt.NDArray[np.double]: A normalized quaternion representing a random non-zero
            rotation. np.double is used as that is what qt.from_float_array returns.
    """
    return cast(
        "npt.NDArray[np.double]",
        qt.from_float_array(
            np.array(
                [
                    draw(
                        st.one_of(
                            st.floats(-1, -DEFAULT_TOLERANCE),
                            st.floats(DEFAULT_TOLERANCE, 1),
                        )
                    )
                    for _ in range(4)
                ],
                dtype=np.double,
            )
        ).normalized(),
    )
