"""Unit tests for liveview_experiment - testing high-level behavior."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from contrib.liveview_experiment.src.experiment_state import ExperimentState
from contrib.liveview_experiment.src.metadata_extractor import (
    ExperimentMetadata,
    MetadataExtractor,
)


class TestExperimentMetadata:
    """Test ExperimentMetadata behavior."""

    def test_metadata_initialization(self) -> None:
        """Test that metadata can be initialized with values."""
        metadata = ExperimentMetadata(
            environment_name="test_env",
            experiment_name="test_experiment",
            config_path="/path/to/config.yaml",
        )
        assert metadata.environment_name == "test_env"
        assert metadata.experiment_name == "test_experiment"
        assert metadata.config_path == "/path/to/config.yaml"

    def test_metadata_to_dict(self) -> None:
        """Test that metadata converts to dictionary correctly."""
        metadata = ExperimentMetadata(
            environment_name="env", experiment_name="exp", config_path="/path"
        )
        result = metadata.to_dict()
        assert result == {
            "environment_name": "env",
            "experiment_name": "exp",
            "config_path": "/path",
        }

    def test_metadata_defaults(self) -> None:
        """Test that metadata has sensible defaults."""
        metadata = ExperimentMetadata()
        assert metadata.environment_name == ""
        assert metadata.experiment_name == ""
        assert metadata.config_path == ""


class TestMetadataExtractor:
    """Test MetadataExtractor high-level behavior."""

    def test_extract_environment_name_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that environment name is extracted from environment variables."""
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "test_environment")
        extractor = MetadataExtractor()
        metadata = extractor.extract()
        assert metadata.environment_name == "test_environment"

    def test_extract_experiment_name_from_run_name(self) -> None:
        """Test that experiment name falls back to run_name."""
        extractor = MetadataExtractor(run_name="my_experiment")
        metadata = extractor.extract()
        assert metadata.experiment_name == "my_experiment"

    def test_extract_experiment_name_from_config(self) -> None:
        """Test that experiment name is extracted from config when available."""
        config = {"experiment": {"_name_": "config_experiment"}}
        extractor = MetadataExtractor(config=config)
        metadata = extractor.extract()
        assert metadata.experiment_name == "config_experiment"

    def test_extract_returns_complete_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that extract returns all available metadata."""
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "test_env")
        extractor = MetadataExtractor(run_name="test_run")
        metadata = extractor.extract()

        assert isinstance(metadata, ExperimentMetadata)
        assert metadata.environment_name == "test_env"
        assert metadata.experiment_name == "test_run"
        # config_path may be empty if Hydra not available, which is fine
        assert isinstance(metadata.config_path, str)


class TestStateNormalization:
    """Test state normalization behavior - high level."""

    def test_state_preserves_zero_values(self) -> None:
        """Test that zero values in state are preserved (not treated as None)."""
        state = ExperimentState(
            current_step=0,
            current_epoch=0,
            total_train_steps=0,
            train_episodes=0,
        )
        # Zero values should be preserved, not converted to defaults
        assert state.current_step == 0
        assert state.current_epoch == 0
        assert state.total_train_steps == 0
        assert state.train_episodes == 0

    def test_state_handles_none_values(self) -> None:
        """Test that None values in state are handled gracefully."""
        state = ExperimentState(
            current_step=None,
            run_name=None,
            do_train=None,
        )
        # None values should be acceptable in the state object
        assert state.current_step is None
        assert state.run_name is None
        assert state.do_train is None

    def test_state_has_sensible_defaults(self) -> None:
        """Test that state has sensible default values."""
        state = ExperimentState()
        # Check that defaults are set
        assert state.status == "initializing"
        assert state.experiment_mode == "train"
        assert state.total_train_steps == 0
        assert state.current_step == 0


def test_normalize_status_allows_abort_states() -> None:
    """Abort-related statuses should be preserved, not mapped to 'initializing'."""
    from contrib.liveview_experiment.src.liveview_experiment import ExperimentLiveView
    from contrib.liveview_experiment.src.state_manager import ExperimentStateManager
    from pyview.live_view import LiveViewSocket  # type: ignore[import-not-found]

    manager = ExperimentStateManager(route_path="/")
    view = ExperimentLiveView(manager)

    # Use a dummy socket with minimal attributes required by _create_context_from_state
    class DummySocket(LiveViewSocket):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__(id="test-socket")

    socket = DummySocket()
    # Initialize context once so _update_context_from_state can run
    view.mount(socket, {})  # type: ignore[arg-type]

    # Simulate state coming from ZMQ with aborting / aborted
    manager.experiment_state.status = "aborting"
    view._update_context_from_state(socket)
    assert socket.context.status == "aborting"

    manager.experiment_state.status = "aborted"
    view._update_context_from_state(socket)
    assert socket.context.status == "aborted"
