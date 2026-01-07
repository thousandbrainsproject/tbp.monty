"""Tests for visualization state management."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from contrib.liveview_experiment.src.visualization_state import (
    ChartBufferConfig,
    EpisodeMarker,
    EvidencePoint,
    VisualizationState,
    VisualizationStateManager,
)


class TestEvidencePoint:
    """Tests for EvidencePoint dataclass."""

    def test_creates_with_minimal_data(self) -> None:
        point = EvidencePoint(step=0, evidences={})
        assert point.step == 0
        assert point.evidences == {}
        assert point.target_object == ""

    def test_creates_with_full_data(self) -> None:
        point = EvidencePoint(
            step=42,
            evidences={"mug": 0.8, "bowl": 0.2},
            target_object="mug",
            hypothesis_count=100,
            episode=3,
            timestamp=1234567890.0,
        )
        assert point.step == 42
        assert point.evidences == {"mug": 0.8, "bowl": 0.2}
        assert point.target_object == "mug"
        assert point.hypothesis_count == 100
        assert point.episode == 3

    def test_to_dict_serializes_all_fields(self) -> None:
        point = EvidencePoint(
            step=10,
            evidences={"mug": 0.5},
            target_object="mug",
            episode=1,
        )
        d = point.to_dict()
        assert d["step"] == 10
        assert d["evidences"] == {"mug": 0.5}
        assert d["target_object"] == "mug"
        assert d["episode"] == 1

    def test_from_dict_deserializes(self) -> None:
        data = {
            "step": 20,
            "evidences": {"bowl": 0.9},
            "target_object": "bowl",
            "hypothesis_count": 50,
        }
        point = EvidencePoint.from_dict(data)
        assert point.step == 20
        assert point.evidences == {"bowl": 0.9}
        assert point.target_object == "bowl"
        assert point.hypothesis_count == 50

    def test_from_dict_handles_missing_fields(self) -> None:
        point = EvidencePoint.from_dict({})
        assert point.step == 0
        assert point.evidences == {}
        assert point.target_object == ""


class TestChartBufferConfig:
    """Tests for ChartBufferConfig."""

    def test_default_keeps_all_points(self) -> None:
        config = ChartBufferConfig()
        assert config.max_points is None
        assert config.throttle_ms == 1000

    def test_configures_max_points(self) -> None:
        config = ChartBufferConfig(max_points=500)
        assert config.max_points == 500


class TestVisualizationState:
    """Tests for VisualizationState."""

    def test_appends_evidence_points(self) -> None:
        state = VisualizationState()
        point = EvidencePoint(step=1, evidences={"mug": 0.5})

        state.append_evidence(point)

        assert state.point_count == 1
        assert state.latest_step == 1

    def test_tracks_object_names_from_evidences(self) -> None:
        state = VisualizationState()
        state.append_evidence(EvidencePoint(step=1, evidences={"mug": 0.5}))
        state.append_evidence(EvidencePoint(step=2, evidences={"bowl": 0.3, "mug": 0.7}))

        assert state.object_names == {"mug", "bowl"}

    def test_respects_max_points_limit(self) -> None:
        config = ChartBufferConfig(max_points=3)
        state = VisualizationState(config=config)

        for i in range(5):
            state.append_evidence(EvidencePoint(step=i, evidences={}))

        assert state.point_count == 3
        # Should keep the most recent 3 points
        steps = [p.step for p in state.evidence_history]
        assert steps == [2, 3, 4]

    def test_adds_episode_markers(self) -> None:
        state = VisualizationState()
        marker = EpisodeMarker(start_step=0, target_object="mug", episode=1)

        state.add_episode_marker(marker)

        assert len(state.episode_markers) == 1
        assert state.episode_markers[0].target_object == "mug"

    def test_clear_resets_all_data(self) -> None:
        state = VisualizationState()
        state.append_evidence(EvidencePoint(step=1, evidences={"mug": 0.5}))
        state.add_episode_marker(EpisodeMarker(start_step=0, target_object="mug", episode=1))
        state.current_mesh_url = "/mesh/mug.glb"

        state.clear()

        assert state.point_count == 0
        assert len(state.episode_markers) == 0
        assert len(state.object_names) == 0
        assert state.current_mesh_url is None

    def test_get_chart_data_returns_serializable_dict(self) -> None:
        state = VisualizationState()
        state.append_evidence(EvidencePoint(step=1, evidences={"mug": 0.8}))
        state.add_episode_marker(EpisodeMarker(start_step=0, target_object="mug", episode=1))

        data = state.get_chart_data()

        assert "evidence_history" in data
        assert "episode_markers" in data
        assert "object_names" in data
        assert len(data["evidence_history"]) == 1
        assert data["object_names"] == ["mug"]

    def test_get_chart_data_json_is_valid_json(self) -> None:
        state = VisualizationState()
        state.append_evidence(EvidencePoint(step=1, evidences={"mug": 0.5}))

        json_str = state.get_chart_data_json()

        # Should parse without error
        parsed = json.loads(json_str)
        assert parsed["evidence_history"][0]["step"] == 1

    def test_latest_step_returns_none_when_empty(self) -> None:
        state = VisualizationState()
        assert state.latest_step is None

    def test_get_unsent_data_returns_none_when_nothing_new(self) -> None:
        """get_unsent_data returns None if all data has been sent."""
        state = VisualizationState()
        state.append_evidence(EvidencePoint(step=1, evidences={"mug": 0.5}))
        state.mark_as_sent()

        # No new data since last send
        assert state.get_unsent_data() is None

    def test_get_unsent_data_returns_only_new_points(self) -> None:
        """get_unsent_data returns only points added since last send."""
        state = VisualizationState()

        # Add first batch and mark sent
        state.append_evidence(EvidencePoint(step=1, evidences={"mug": 0.5}))
        state.append_evidence(EvidencePoint(step=2, evidences={"mug": 0.6}))
        state.mark_as_sent()

        # Add second batch
        state.append_evidence(EvidencePoint(step=3, evidences={"mug": 0.7}))
        state.append_evidence(EvidencePoint(step=4, evidences={"mug": 0.8}))

        unsent = state.get_unsent_data()
        assert unsent is not None
        assert len(unsent["new_points"]) == 2
        assert unsent["new_points"][0]["step"] == 3
        assert unsent["new_points"][1]["step"] == 4
        assert unsent["total_points"] == 4

    def test_reset_sent_tracking_allows_resending_all(self) -> None:
        """reset_sent_tracking allows all data to be sent again."""
        state = VisualizationState()
        state.append_evidence(EvidencePoint(step=1, evidences={"mug": 0.5}))
        state.mark_as_sent()

        # Should have nothing to send
        assert state.get_unsent_data() is None

        # Reset tracking
        state.reset_sent_tracking()

        # Now should have all data to send
        unsent = state.get_unsent_data()
        assert unsent is not None
        assert len(unsent["new_points"]) == 1
        assert unsent["new_points"][0]["step"] == 1


class TestVisualizationStateManager:
    """Tests for VisualizationStateManager."""

    def test_processes_evidence_data(self) -> None:
        manager = VisualizationStateManager()

        manager.process_evidence_data({
            "step": 10,
            "evidences": {"mug": 0.6, "bowl": 0.4},
            "target_object": "mug",
        })

        assert manager.state.point_count == 1
        assert manager.state.latest_step == 10

    def test_detects_episode_transitions(self) -> None:
        manager = VisualizationStateManager()

        # First episode
        manager.process_evidence_data({
            "step": 0,
            "evidences": {"mug": 0.5},
            "target_object": "mug",
            "episode": 1,
        })

        # Same episode
        manager.process_evidence_data({
            "step": 1,
            "evidences": {"mug": 0.6},
            "target_object": "mug",
            "episode": 1,
        })

        # New episode
        manager.process_evidence_data({
            "step": 2,
            "evidences": {"bowl": 0.5},
            "target_object": "bowl",
            "episode": 2,
        })

        assert len(manager.state.episode_markers) == 2
        assert manager.state.episode_markers[0].episode == 1
        assert manager.state.episode_markers[1].episode == 2
        assert manager.state.episode_markers[1].start_step == 2

    def test_processes_mesh_data(self) -> None:
        manager = VisualizationStateManager()

        manager.process_mesh_data({"mesh_url": "/static/meshes/mug.glb"})

        assert manager.state.current_mesh_url == "/static/meshes/mug.glb"

    def test_processes_mesh_data_with_url_key(self) -> None:
        manager = VisualizationStateManager()

        manager.process_mesh_data({"url": "/static/meshes/bowl.glb"})

        assert manager.state.current_mesh_url == "/static/meshes/bowl.glb"

    def test_clear_resets_state_and_episode_tracking(self) -> None:
        manager = VisualizationStateManager()
        manager.process_evidence_data({
            "step": 0,
            "evidences": {"mug": 0.5},
            "episode": 1,
        })

        manager.clear()

        assert manager.state.point_count == 0
        assert manager._last_episode is None

    def test_uses_custom_config(self) -> None:
        config = ChartBufferConfig(max_points=10, throttle_ms=500)
        manager = VisualizationStateManager(config)

        assert manager.config.max_points == 10
        assert manager.config.throttle_ms == 500

    def test_buffer_limit_applied_with_config(self) -> None:
        config = ChartBufferConfig(max_points=2)
        manager = VisualizationStateManager(config)

        for i in range(5):
            manager.process_evidence_data({"step": i, "evidences": {}})

        assert manager.state.point_count == 2
        steps = [p.step for p in manager.state.evidence_history]
        assert steps == [3, 4]

