"""Type aliases for better code readability and type safety."""

from __future__ import annotations

from typing import Any, Callable, Coroutine, Dict, List, Tuple, Union

# Configuration types
ConfigDict = Dict[str, Any]
"""Type alias for experiment configuration dictionaries."""

# State types
StateDict = Dict[str, Any]
"""Type alias for state update dictionaries."""

# Message types
MessagePayload = Dict[str, Any]
"""Type alias for pub/sub message payloads."""

MetricData = Tuple[str, float, Dict[str, Any]]
"""Type alias for metric data: (name, value, metadata)."""

# Callback types
PubSubCallback = Callable[[str, Any], None]
"""Type alias for pub/sub callback functions."""

AsyncPubSubCallback = Callable[[str, Any], Coroutine[Any, Any, None]]
"""Type alias for async pub/sub callback functions."""

HubDict = Dict[str, List[Union[PubSubCallback, AsyncPubSubCallback]]]
"""Type alias for pub/sub hub dictionaries."""
