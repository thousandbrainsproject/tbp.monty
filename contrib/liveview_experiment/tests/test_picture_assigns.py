"""What the page is told to draw: a URL to a file, or the blob as a fallback."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from contrib.liveview_experiment.src import frame_files
from contrib.liveview_experiment.src.liveview_experiment import ExperimentLiveView, picture
from contrib.liveview_experiment.src.visualization_state import SensorImages

TINY_PNG = base64.b64encode(bytes(range(16))).decode("ascii")


@pytest.fixture(autouse=True)
def frames(tmp_path, monkeypatch):
    monkeypatch.setenv(frame_files.FRAMES_DIR_ENV, str(tmp_path / "frames"))
    return tmp_path / "frames"


class TestPicture:
    def test_a_picture_is_a_url_and_the_file_is_there(self, frames: Path) -> None:
        src = picture("camera", TINY_PNG, 12, "run")

        assert src == "/frames/run_camera-12.png"
        assert (frames / "run_camera-12.png").read_bytes() == bytes(range(16))

    def test_no_picture_is_nothing_to_draw(self) -> None:
        assert picture("camera", None, 12, "run") is None

    def test_an_unwritable_pictures_directory_falls_back_to_the_blob(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        # A read-only home: publishing fails, the page still draws.
        monkeypatch.setenv(frame_files.FRAMES_DIR_ENV, str(tmp_path / "a-file"))
        (tmp_path / "a-file").write_text("not a directory")

        src = picture("camera", TINY_PNG, 12, "run")

        assert src is not None
        assert src.startswith("data:image/png;base64,")


class TestThePageData:
    def test_the_publisher_data_carries_a_src_for_each_picture(
        self, frames: Path
    ) -> None:
        view = object.__new__(ExperimentLiveView)  # only the builder is under test
        state = _state_with(SensorImages(camera_image=TINY_PNG, depth_image=TINY_PNG, step=7))

        data = ExperimentLiveView._build_publisher_data(view, "run", state, 0)

        assert data["camera_image_src"] == "/frames/run_camera-7.png"
        assert data["depth_image_src"] == "/frames/run_depth-7.png"
        assert data["has_sensor_images"] is True
        # The blobs stay for anything that still reads them, and cost nothing
        # here: the template draws the src.
        assert data["camera_image_b64"] == TINY_PNG

    def test_a_publisher_with_no_picture_says_so(self) -> None:
        view = object.__new__(ExperimentLiveView)
        state = _state_with(SensorImages())

        data = ExperimentLiveView._build_publisher_data(view, "run", state, 0)

        assert data["camera_image_src"] is None
        assert data["has_sensor_images"] is False


def _state_with(images: SensorImages):
    """A VisualizationState holding these pictures and no chart history."""

    class State:
        sensor_images = images
        point_count = 0
        current_max_evidence = None

        @staticmethod
        def points_since(_cursor: int) -> list:
            return []

    return State()
