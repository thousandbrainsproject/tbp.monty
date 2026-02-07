# Copyright 2025-2026 Thousand Brains Project
# Copyright 2021-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json
import sys
from datetime import datetime

from tbp.monty.libs.telemetry import Event

__all__ = ["StdoutHandler"]

LOCAL_TZ = datetime.now().astimezone().tzinfo


class StdoutHandler:
    def _format(self, event: Event) -> str:
        # basic structure
        timestamp = datetime.fromtimestamp(event.ts, tz=LOCAL_TZ)
        log_line = f"{timestamp} | {event.name}"

        # appending a message, if there is one
        if event.payload:
            if "message" in event.payload:
                log_line += f" | {event.payload['message']}"
                event.payload.pop("message")

        # appending attributes, if there are any
        if event.attrs:
            atts_formatted = ",".join([f'{k}:"{v}"' for k, v in event.attrs])
            log_line += f" | {atts_formatted}"

        # appending the rest of the payload, if there is any
        if event.payload:
            log_line += f" | {json.dumps(event.payload)}"

        return log_line

    async def handle(self, event: Event) -> None:
        log_line = self._format(event)
        print(log_line, file=sys.stdout, flush=True)
