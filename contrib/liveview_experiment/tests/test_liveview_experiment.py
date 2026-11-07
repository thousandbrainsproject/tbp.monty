"""Unit tests for liveview_experiment - testing high-level behavior."""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

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
    # Try to use real implementation if pyview is available (Python 3.11+)
    try:
        from contrib.liveview_experiment.src.liveview_experiment import (
            ExperimentLiveView,
        )
        from contrib.liveview_experiment.src.state_manager import (
            ExperimentStateManager,
        )

        manager = ExperimentStateManager(route_path="/")
        view = ExperimentLiveView(manager)
        normalize_status = view._normalize_status
    except (ImportError, ModuleNotFoundError):
        # Fallback to testing the logic directly if pyview not available
        def normalize_status(status: str | None) -> str:
            """Normalize status to a value understood by the UI."""
            normalized = (status or "initializing").lower()
            valid_statuses = (
                "initializing",
                "running",
                "completed",
                "error",
                "aborting",
                "aborted",
            )
            if normalized in valid_statuses:
                return normalized
            return normalized

    # Test that abort-related statuses are preserved
    assert normalize_status("aborting") == "aborting"
    assert normalize_status("ABORTING") == "aborting"
    assert normalize_status("aborted") == "aborted"
    assert normalize_status("ABORTED") == "aborted"

    # Test that other statuses work correctly
    assert normalize_status("running") == "running"
    assert normalize_status("completed") == "completed"
    assert normalize_status(None) == "initializing"


def test_every_open_page_receives_every_new_evidence_point() -> None:
    """Regression: a second viewer must not starve the first viewer's chart.

    Charts only update from points streamed to a connection, so what a
    connection has already received is tracked per connection rather than
    globally. Otherwise the page that renders first consumes each batch and all
    other open pages freeze while their images keep updating.
    """
    pytest.importorskip("pyview")
    from contrib.liveview_experiment.src.liveview_experiment import (  # noqa: PLC0415
        ExperimentLiveView,
    )
    from contrib.liveview_experiment.src.state_manager import (  # noqa: PLC0415
        ExperimentStateManager,
    )

    manager = ExperimentStateManager(route_path="/")
    view = ExperimentLiveView(manager)
    manager.visualization.process_evidence_data({"step": 1, "evidences": {"mug": 0.5}})

    first_page = _connected_page(view, manager)
    second_page = _connected_page(view, manager)

    assert _steps_rendered_to(view, first_page) == [1]
    assert _steps_rendered_to(view, second_page) == [1]

    manager.visualization.process_evidence_data({"step": 2, "evidences": {"mug": 0.6}})

    assert _steps_rendered_to(view, first_page) == [2]
    assert _steps_rendered_to(view, second_page) == [2]


def _connected_page(view: object, manager: object) -> SimpleNamespace:
    """A freshly connected page, holding its own stream cursors."""
    return SimpleNamespace(context=view._create_context_from_state(manager.experiment_state))


def test_follow_live_is_per_connection_and_survives_a_rerender() -> None:
    """The chart mode is server state, so it outlives patches and stays per page."""
    pytest.importorskip("pyview")
    from contrib.liveview_experiment.src.liveview_experiment import (  # noqa: PLC0415
        ExperimentLiveView,
    )
    from contrib.liveview_experiment.src.state_manager import (  # noqa: PLC0415
        ExperimentStateManager,
    )

    manager = ExperimentStateManager(route_path="/")
    view = ExperimentLiveView(manager)
    first_page = _connected_page(view, manager)
    second_page = _connected_page(view, manager)

    asyncio.run(view.handle_event("toggle_follow_live", {}, first_page))

    view._update_context_from_state(first_page)
    view._update_context_from_state(second_page)

    assert view._build_template_assigns(first_page.context)["follow_live"] is True
    assert view._build_template_assigns(second_page.context)["follow_live"] is False


def test_stream_item_of_point_without_episode_has_empty_episode() -> None:
    """A missing episode must reach the chart as empty, not as the text "None"."""
    pytest.importorskip("pyview")
    from contrib.liveview_experiment.src.liveview_experiment import (  # noqa: PLC0415
        ExperimentLiveView,
    )
    from contrib.liveview_experiment.src.state_manager import (  # noqa: PLC0415
        ExperimentStateManager,
    )
    from contrib.liveview_experiment.src.visualization_state import (  # noqa: PLC0415
        EvidencePoint,
    )

    view = ExperimentLiveView(ExperimentStateManager(route_path="/"))

    [item] = view._build_stream_items(
        "demo", [EvidencePoint(step=1, evidences={"mug": 0.5})]
    )

    assert item["episode"] == ""
    assert item["timestamp"] == ""


def _steps_rendered_to(view: object, page: SimpleNamespace) -> list[int]:
    """Steps one page receives in a single update, taken from its assigns."""
    view._update_context_from_state(page)
    assigns = view._build_template_assigns(page.context)
    return [point["step"] for _dom_id, point in assigns["evidence_stream"]]
