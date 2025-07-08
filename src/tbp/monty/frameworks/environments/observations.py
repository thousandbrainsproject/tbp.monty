# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from typing import Any, Dict, Union

SensorObservation = Dict[str, Any]
"""The observation from a single sensor."""

AgentObservation = Dict[str, SensorObservation]
"""The observations from all sensors of an agent."""

Observations = Dict[str, Union[Any, AgentObservation]]
"""The observations from all agents in the environment."""
