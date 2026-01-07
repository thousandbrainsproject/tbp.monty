"""Visualization state management for live charts and 3D views.

Provides buffered history for time-series data with configurable depth,
designed to work with the existing ZMQ telemetry stream.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChartBufferConfig:
    """Configuration for chart data buffering.

    Attributes:
        max_points: Maximum number of data points to retain.
            None means keep all points (unlimited).
        throttle_ms: Minimum milliseconds between updates.
            Defaults to 1000 (1 update per second).
    """

    max_points: int | None = None  # Keep all points by default
    throttle_ms: int = 1000


@dataclass
class EvidencePoint:
    """A single evidence data point for charting.

    Attributes:
        step: The experiment step number.
        evidences: Mapping of object names to evidence scores.
        target_object: The current target object being sensed.
        hypothesis_count: Number of active hypotheses (optional).
        episode: Current episode number (optional).
        timestamp: Unix timestamp when this point was recorded.
    """

    step: int
    evidences: dict[str, float]
    target_object: str = ""
    hypothesis_count: int | None = None
    episode: int | None = None
    timestamp: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "step": self.step,
            "evidences": self.evidences,
            "target_object": self.target_object,
            "hypothesis_count": self.hypothesis_count,
            "episode": self.episode,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidencePoint:
        """Create from dictionary."""
        return cls(
            step=data.get("step", 0),
            evidences=data.get("evidences", {}),
            target_object=data.get("target_object", ""),
            hypothesis_count=data.get("hypothesis_count"),
            episode=data.get("episode"),
            timestamp=data.get("timestamp"),
        )


@dataclass
class EpisodeMarker:
    """Marks the start of an episode for chart background bands.

    Attributes:
        start_step: Step number where episode begins.
        target_object: Object being sensed in this episode.
        episode: Episode number.
    """

    start_step: int
    target_object: str
    episode: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "start_step": self.start_step,
            "target_object": self.target_object,
            "episode": self.episode,
        }


@dataclass
class SensorImages:
    """Container for sensor image snapshots.

    Holds base64-encoded PNG images from sensors for inline HTML display.

    Attributes:
        camera_image: Base64-encoded camera/view_finder RGBA image.
        depth_image: Base64-encoded depth image (grayscale).
        step: Step number when images were captured.
        timestamp: Unix timestamp when images were captured.
    """

    camera_image: str | None = None
    depth_image: str | None = None
    step: int = 0
    timestamp: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "camera_image": self.camera_image,
            "depth_image": self.depth_image,
            "step": self.step,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SensorImages:
        """Create from dictionary."""
        return cls(
            camera_image=data.get("camera_image"),
            depth_image=data.get("depth_image"),
            step=data.get("step", 0),
            timestamp=data.get("timestamp"),
        )


@dataclass
class VisualizationState:
    """State container for all visualization data.

    Manages buffered history for evidence charts and tracks episode
    transitions for background bands.

    Attributes:
        config: Buffer configuration.
        evidence_history: Buffered evidence data points.
        episode_markers: List of episode start markers.
        current_mesh_url: URL of currently displayed mesh (if any).
        object_names: Set of all object names seen so far.
        sensor_images: Latest sensor image snapshots.
    """

    config: ChartBufferConfig = field(default_factory=ChartBufferConfig)
    evidence_history: deque[EvidencePoint] = field(default_factory=deque)
    episode_markers: list[EpisodeMarker] = field(default_factory=list)
    current_mesh_url: str | None = None
    object_names: set[str] = field(default_factory=set)
    sensor_images: SensorImages = field(default_factory=SensorImages)
    # Track what's been sent via push_event (for incremental updates)
    _last_sent_count: int = field(default=0, repr=False)
    _last_sent_episode_count: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """Initialize deque with maxlen from config."""
        if self.config.max_points is not None:
            # Re-create deque with maxlen
            self.evidence_history = deque(
                self.evidence_history, maxlen=self.config.max_points
            )

    def append_evidence(self, point: EvidencePoint) -> None:
        """Append an evidence point to history.

        Args:
            point: Evidence data point to append.
        """
        self.evidence_history.append(point)
        # Track object names for legend
        self.object_names.update(point.evidences.keys())

    def add_episode_marker(self, marker: EpisodeMarker) -> None:
        """Add an episode marker for chart background bands.

        Args:
            marker: Episode marker to add.
        """
        self.episode_markers.append(marker)

    def clear(self) -> None:
        """Clear all visualization data."""
        self.evidence_history.clear()
        self.episode_markers.clear()
        self.object_names.clear()
        self.current_mesh_url = None
        self.sensor_images = SensorImages()

    def update_sensor_images(self, images: SensorImages) -> None:
        """Update the current sensor images.

        Args:
            images: New sensor images to store.
        """
        self.sensor_images = images

    def get_chart_data_json(self) -> str:
        """Get chart data as JSON string for template embedding.

        Returns:
            JSON string containing evidence history and episode markers.
        """
        return json.dumps(self.get_chart_data())

    def get_chart_data(self) -> dict[str, Any]:
        """Get chart data as dictionary.

        Returns:
            Dictionary with evidence_history, episode_markers, and object_names.
        """
        return {
            "evidence_history": [p.to_dict() for p in self.evidence_history],
            "episode_markers": [m.to_dict() for m in self.episode_markers],
            "object_names": sorted(self.object_names),
            "current_mesh_url": self.current_mesh_url,
        }

    @property
    def point_count(self) -> int:
        """Number of evidence points in history."""
        return len(self.evidence_history)

    @property
    def latest_step(self) -> int | None:
        """Most recent step number, or None if empty."""
        if not self.evidence_history:
            return None
        return self.evidence_history[-1].step

    def get_unsent_data(self) -> dict[str, Any] | None:
        """Get only data points that haven't been sent via push_event yet.

        Returns:
            Dictionary with new points and markers, or None if nothing new.
        """
        current_count = len(self.evidence_history)
        current_episode_count = len(self.episode_markers)

        if (
            current_count <= self._last_sent_count
            and current_episode_count <= self._last_sent_episode_count
        ):
            return None

        # Get new points (handle deque properly - can't slice directly by index
        # if buffer wrapped around, so convert to list for new items)
        new_points_start = max(0, self._last_sent_count)
        history_list = list(self.evidence_history)
        new_points = history_list[new_points_start:]

        # Get new episode markers
        new_markers = self.episode_markers[self._last_sent_episode_count :]

        return {
            "new_points": [p.to_dict() for p in new_points],
            "new_markers": [m.to_dict() for m in new_markers],
            "new_object_names": sorted(self.object_names),
            "total_points": current_count,
        }

    def mark_as_sent(self) -> None:
        """Mark all current data as sent (after push_event)."""
        self._last_sent_count = len(self.evidence_history)
        self._last_sent_episode_count = len(self.episode_markers)

    def reset_sent_tracking(self) -> None:
        """Reset sent tracking (e.g., for new connections that need full data)."""
        self._last_sent_count = 0
        self._last_sent_episode_count = 0


class VisualizationStateManager:
    """Manages visualization state with support for multiple chart types.

    This manager processes visualization-specific data messages and maintains
    buffered history for real-time charting.

    Example usage:
        config = ChartBufferConfig(max_points=1000)
        manager = VisualizationStateManager(config)

        # Process incoming data
        manager.process_evidence_data({
            "step": 42,
            "evidences": {"mug": 0.8, "bowl": 0.2},
            "target_object": "mug"
        })

        # Get data for template
        chart_json = manager.state.get_chart_data_json()
    """

    def __init__(self, config: ChartBufferConfig | None = None) -> None:
        """Initialize the visualization state manager.

        Args:
            config: Chart buffer configuration. Uses defaults if not provided.
        """
        self.config = config or ChartBufferConfig()
        self.state = VisualizationState(config=self.config)
        self._last_episode: int | None = None

    def process_evidence_data(self, data: dict[str, Any]) -> None:
        """Process evidence chart data from telemetry stream.

        Args:
            data: Evidence data dictionary with step, evidences, etc.
        """
        point = EvidencePoint.from_dict(data)
        self.state.append_evidence(point)

        # Check for episode transition
        episode = data.get("episode")
        if episode is not None and episode != self._last_episode:
            marker = EpisodeMarker(
                start_step=point.step,
                target_object=point.target_object,
                episode=episode,
            )
            self.state.add_episode_marker(marker)
            self._last_episode = episode

    def process_mesh_data(self, data: dict[str, Any]) -> None:
        """Process mesh viewer data from telemetry stream.

        Args:
            data: Mesh data dictionary with mesh_url, object_name, etc.
        """
        mesh_url = data.get("mesh_url") or data.get("url")
        if mesh_url:
            self.state.current_mesh_url = mesh_url

    def process_sensor_images(self, data: dict[str, Any]) -> None:
        """Process sensor image data from telemetry stream.

        Args:
            data: Sensor images dictionary with camera_image, depth_image, etc.
        """
        images = SensorImages.from_dict(data)
        self.state.update_sensor_images(images)

    def clear(self) -> None:
        """Clear all visualization data and reset state."""
        self.state.clear()
        self._last_episode = None
