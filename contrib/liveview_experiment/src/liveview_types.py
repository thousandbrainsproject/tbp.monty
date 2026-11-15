"""Type aliases for better code readability and type safety."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

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

# Template types
TemplateAssigns = dict[str, Any]
"""Type alias for template variable assignments."""

# Data stream types
DataStreamValue = dict[str, Any]
"""Type alias for a single data stream value."""

DataStreams = dict[str, DataStreamValue]
"""Type alias for data streams dictionary."""

# Log types
LogEntry = dict[str, Any]
"""Type alias for a single log entry."""

LogEntries = list[LogEntry]
"""Type alias for list of log entries."""

# Metrics types
MetricMetadata = dict[str, Any]
"""Type alias for metric metadata dictionary."""

MetricsDict = dict[str, float]
"""Type alias for metrics dictionary (name -> value)."""

# Callback types
PubSubCallback = Callable[[str, Any], None]
"""Type alias for pub/sub callback functions."""

AsyncPubSubCallback = Callable[[str, Any], Coroutine[Any, Any, None]]
"""Type alias for async pub/sub callback functions."""

HubDict = dict[str, list[PubSubCallback | AsyncPubSubCallback]]
"""Type alias for pub/sub hub dictionaries."""
