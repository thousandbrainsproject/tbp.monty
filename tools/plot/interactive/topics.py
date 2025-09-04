# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


from dataclasses import dataclass
from typing import Any


@dataclass
class TopicMessage:
    """Message passed on the pubsub bus.

    Attributes:
        name: Topic name.
        value: Value for the topic.
    """

    name: str
    value: Any


@dataclass
class TopicSpec:
    """Specification for a topic tracked by a widget updater.

    Attributes:
        name: Topic name to track.
        required: Whether this topic is required for the callback trigger. If
            True, the updater will not call the callback until a message for this
            topic arrives.
    """

    name: str
    required: bool = True
