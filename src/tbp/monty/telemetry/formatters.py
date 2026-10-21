# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json
import logging
from typing import cast

from tbp.monty.telemetry.schemas import TelemetrySchema


class JsonFormatter(logging.Formatter):
    """Serializes a telemetry `LogRecord` instance as a JSON string."""

    def format(self, record: logging.LogRecord) -> str:
        schema = cast("TelemetrySchema", record.msg)

        if not isinstance(schema, TelemetrySchema):
            raise TypeError("This formatter only accepts telemetry schemas")

        return json.dumps(
            {
                "level": record.levelname,
                "name": record.name,
                "funcName": record.funcName,
                "lineno": record.lineno,
                **schema.model_dump(mode="json"),
            }
        )
