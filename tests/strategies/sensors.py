# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import hypothesis.strategies as st

from tbp.monty.frameworks.sensors import SensorID


@st.composite
def sensor_id(draw: st.DrawFn) -> SensorID:
    return SensorID(draw(st.text(min_size=1, max_size=10)))
