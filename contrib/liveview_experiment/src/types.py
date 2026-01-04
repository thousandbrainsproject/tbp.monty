"""Type aliases for better code readability and type safety."""

from __future__ import annotations

from typing import Any, Callable

# Configuration types
ConfigDict = dict[str, Any]
"""Type alias for experiment configuration dictionaries."""

# State types
StateDict = dict[str, Any]
"""Type alias for state update dictionaries."""

# Message types
MessagePayload = dict[str, Any]
"""Type alias for pub/sub message payloads."""

MetricData = tuple[str, float, dict[str, Any]]
"""Type alias for metric data: (name, value, metadata)."""

# Callback types
PubSubCallback = Callable[[str, Any], None]
"""Type alias for pub/sub callback functions."""

HubDict = dict[str, list[PubSubCallback]]
"""Type alias for pub/sub hub dictionaries."""
