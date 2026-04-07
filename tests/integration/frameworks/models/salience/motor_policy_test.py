# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import pytest

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

"""
Test I want to write:

Initialize an experiment / episode with cube (or plane) that has a large face that
sits on the X-Y plane. Then supply the policy with goals that are on that plane.
Enact the actions returned by the policy, and verify that the observation after
a goal is attempted is very close to the goal. Note: we must supply goals one after
another -- i.e., not just checking that we can go from having the agent/sensor at
its starting orientation to a single goal orientation.

I believe this is sufficient for verifying that the policy math is working.

Another todo is to check whether we can look at goals that are behind the agent. I
am not 100% confident that the conversion to euler angles that happens within the
policy (which is actually necessary) will work correctly in that case.
"""
