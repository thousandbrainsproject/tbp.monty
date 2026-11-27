"""The pictures a page shows, as files rather than blobs inside every update."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from contrib.liveview_experiment.src import frame_files


@pytest.fixture(autouse=True)
def frames(tmp_path, monkeypatch):
    """A pictures directory of the test's own, so nothing touches the user's."""
    monkeypatch.setenv(frame_files.FRAMES_DIR_ENV, str(tmp_path / "frames"))
    return tmp_path / "frames"


def png(byte: int = 7) -> str:
    """A tiny "PNG" as base64 — the module moves bytes and never decodes pixels."""
    return base64.b64encode(bytes([byte] * 16)).decode("ascii")


class TestPublishing:
    def test_a_picture_becomes_a_file_and_a_url(self, frames: Path) -> None:
        url = frame_files.publish_base64("camera", png(), 42, "run")

        assert url == "/frames/run_camera-42.png"
        assert (frames / "run_camera-42.png").read_bytes() == bytes([7] * 16)

    def test_no_picture_answers_nothing_to_show(self) -> None:
        assert frame_files.publish_base64("camera", None, 1, "run") is None

    def test_a_blob_that_is_not_base64_is_left_alone(self, frames: Path) -> None:
        # The page keeps the blob; one unreadable picture must not crash a step.
        assert frame_files.publish_base64("camera", "not base64 !!", 1, "run") is None
        assert not frames.exists() or not any(frames.iterdir())

    def test_two_runs_do_not_overwrite_each_other(self, frames: Path) -> None:
        assert frame_files.publish_base64("camera", png(1), 5, "alpha") == (
            "/frames/alpha_camera-5.png"
        )
        assert frame_files.publish_base64("camera", png(2), 5, "beta") == (
            "/frames/beta_camera-5.png"
        )

        assert (frames / "alpha_camera-5.png").read_bytes() == bytes([1] * 16)
        assert (frames / "beta_camera-5.png").read_bytes() == bytes([2] * 16)


class TestWhatIsKept:
    def test_only_the_latest_pictures_stay_on_disk(self, frames: Path) -> None:
        steps = frame_files.KEEP_PER_KIND + 12

        for step in range(steps):
            frame_files.publish_base64("camera", png(), step, "run")

        kept = sorted(path.name for path in frames.iterdir())
        assert len(kept) == frame_files.KEEP_PER_KIND
        # The newest, and none of the oldest.
        assert kept[-1] == f"run_camera-{steps - 1}.png"
        assert f"run_camera-0.png" not in kept

    def test_kinds_are_kept_separately(self, frames: Path) -> None:
        for step in range(frame_files.KEEP_PER_KIND + 2):
            frame_files.publish_base64("camera", png(), step, "run")
            frame_files.publish_base64("depth", png(), step, "run")

        assert len(list(frames.glob("run_camera-*.png"))) == frame_files.KEEP_PER_KIND
        assert len(list(frames.glob("run_depth-*.png"))) == frame_files.KEEP_PER_KIND


class TestResolving:
    def test_a_published_name_resolves_to_its_file(self, frames: Path) -> None:
        url = frame_files.publish_base64("camera", png(), 3, "run")

        assert frame_files.resolve(url.rsplit("/", 1)[-1]) is not None

    @pytest.mark.parametrize(
        "name",
        ["../secrets.png", "sub/dir.png", ".hidden.png", "", "missing.png"],
    )
    def test_anything_else_is_not_a_picture(self, name: str) -> None:
        frame_files.publish_base64("camera", png(), 3, "run")

        assert frame_files.resolve(name) is None
