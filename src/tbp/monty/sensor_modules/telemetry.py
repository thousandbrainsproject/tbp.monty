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

import numpy as np
import quaternion as qt
from typing_extensions import Self

from tbp.monty.observations import SensorObservation
from tbp.monty.sensor_modules.sensor_module import Payload, Transform, TransformContext


class Snapshot(Transform):
    def __init__(self) -> None:
        pass

    def __call__(self: Self, ctx: TransformContext, payload: Payload) -> Payload:
        raw_observations: list[SensorObservation] = payload.telemetry.setdefault(
            "raw_observations", []
        )
        raw_observations.append(payload.observation)

        sm_properties: list[dict[str, Any]] = payload.telemetry.setdefault(
            "sm_properties", []
        )
        sm_rotation = qt.as_float_array(ctx.sensor_state.rotation)
        sm_location = np.array(ctx.sensor_state.position)
        sm_properties.append({"sm_rotation": sm_rotation, "sm_location": sm_location})

        return payload
