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
        # Reset sent tracking for new experiment
        self.reset_sent_tracking()

    def update_sensor_images(self, images: SensorImages) -> None:
        """Update current sensor images, preserving existing when new data is None.

        Only updates fields that have actual image data (not None) to persist
        images across episode transitions when new episodes don't send images.

        Args:
            images: New sensor images to store (may have None fields).
        """
        # Only update fields that have actual data to preserve existing images
        if images.camera_image is not None:
            self.sensor_images.camera_image = images.camera_image
        if images.depth_image is not None:
            self.sensor_images.depth_image = images.depth_image
        # Always update step and timestamp if present
        if images.step is not None:
            self.sensor_images.step = images.step
        if images.timestamp is not None:
            self.sensor_images.timestamp = images.timestamp

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

    @property
    def current_max_evidence(self) -> tuple[str, float] | None:
        """Current maximum evidence from the latest evidence point.

        Returns:
            Tuple of (object_name, evidence_value) for the object with highest evidence,
            or None if no evidence data is available.
        """
        if not self.evidence_history:
            return None
        latest_point = self.evidence_history[-1]
        if not latest_point.evidences:
            return None
        # Find the object with the maximum evidence
        max_object = max(latest_point.evidences.items(), key=lambda x: x[1])
        return (max_object[0], max_object[1])

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

        # Get current max evidence
        max_evidence = self.current_max_evidence
        max_evidence_data = None
        if max_evidence is not None:
            max_evidence_data = {
                "object": max_evidence[0],
                "value": max_evidence[1],
            }

        return {
            "new_points": [p.to_dict() for p in new_points],
            "new_markers": [m.to_dict() for m in new_markers],
            "new_object_names": sorted(self.object_names),
            "total_points": current_count,
            "current_max_evidence": max_evidence_data,
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
    """Manages visualization state with support for multiple publishers.

    This manager processes visualization-specific data messages and maintains
    buffered history for real-time charting. Supports multiple publishers
    (e.g., parallel experiments) by partitioning data by run_name.

    Example usage:
        config = ChartBufferConfig(max_points=1000)
        manager = VisualizationStateManager(config)

        # Process incoming data with run_name
        manager.process_evidence_data({
            "step": 42,
            "evidences": {"mug": 0.8, "bowl": 0.2},
            "target_object": "mug",
            "run_name": "exp_0"
        })

        # Get data for all publishers
        all_states = manager.get_all_states()
    """

    def __init__(self, config: ChartBufferConfig | None = None) -> None:
        """Initialize the visualization state manager.

        Args:
            config: Chart buffer configuration. Uses defaults if not provided.
        """
        self.config = config or ChartBufferConfig()
        # Dictionary of run_name -> VisualizationState
        self.states: dict[str, VisualizationState] = {}
        # Track last episode per publisher
        self._last_episodes: dict[str, int | None] = {}

    def _get_or_create_state(self, run_name: str) -> VisualizationState:
        """Get existing state for run_name or create new one.

        Args:
            run_name: Publisher identifier.

        Returns:
            VisualizationState instance for this publisher.
        """
        if run_name not in self.states:
            self.states[run_name] = VisualizationState(config=self.config)
            self._last_episodes[run_name] = None
        return self.states[run_name]

    def process_evidence_data(self, data: dict[str, Any]) -> None:
        """Process evidence chart data from telemetry stream.

        Args:
            data: Evidence data dictionary with step, evidences, run_name, etc.
        """
        run_name = data.get("run_name", "default")
        state = self._get_or_create_state(run_name)
        point = EvidencePoint.from_dict(data)
        state.append_evidence(point)

        # Check for episode transition (per-publisher tracking)
        episode = data.get("episode")
        last_episode = self._last_episodes.get(run_name)
        if episode is not None and episode != last_episode:
            marker = EpisodeMarker(
                start_step=point.step,
                target_object=point.target_object,
                episode=episode,
            )
            state.add_episode_marker(marker)
            self._last_episodes[run_name] = episode

    def process_mesh_data(self, data: dict[str, Any]) -> None:
        """Process mesh viewer data from telemetry stream.

        Args:
            data: Mesh data dictionary with mesh_url, object_name, run_name, etc.
        """
        run_name = data.get("run_name", "default")
        state = self._get_or_create_state(run_name)
        mesh_url = data.get("mesh_url") or data.get("url")
        if mesh_url:
            state.current_mesh_url = mesh_url

    def process_sensor_images(self, data: dict[str, Any]) -> None:
        """Process sensor image data from telemetry stream.

        Args:
            data: Sensor images dict with camera_image, depth_image, run_name.
        """
        run_name = data.get("run_name", "default")
        state = self._get_or_create_state(run_name)
        images = SensorImages.from_dict(data)
        state.update_sensor_images(images)

    def get_all_states(self) -> dict[str, VisualizationState]:
        """Get all publisher states.

        Returns:
            Dictionary mapping run_name to VisualizationState.
        """
        return self.states

    def get_state(self, run_name: str) -> VisualizationState | None:
        """Get state for specific publisher.

        Args:
            run_name: Publisher identifier.

        Returns:
            VisualizationState if exists, None otherwise.
        """
        return self.states.get(run_name)

    @property
    def state(self) -> VisualizationState:
        """Backward compatibility: get first/default state.

        Returns:
            The first publisher's state, or creates default if empty.
        """
        if not self.states:
            return self._get_or_create_state("default")
        return next(iter(self.states.values()))

    def clear(self) -> None:
        """Clear all visualization data and reset state."""
        for state in self.states.values():
            state.clear()
        self._last_episodes.clear()
