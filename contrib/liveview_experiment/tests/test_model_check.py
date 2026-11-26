"""A missing checkpoint says which model, which release, and what decides where models live.

The failure this replaces was `FileNotFoundError` from inside `torch.load`, five
frames into a traceback: it named a path nobody chose and not the release the
config asks for. These tests hold the message to the three facts a person needs
to act on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root and Monty's own `src` to the path for imports; see
# `test_experiment_factory.py` for why a plain pytest run needs both.
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from contrib.liveview_experiment.src.model_check import (
    DEFAULT_MODELS_DIR,
    checkpoint_of,
    missing_model_message,
)

RELEASE = "pretrained_compositional_objects_v13"
MODEL = "supervised_pre_training_cows_small_compositional"


def named(models: Path, release: str = RELEASE, model: str = MODEL) -> dict[str, str]:
    """A config naming one model the way Monty's configs do."""
    return {"model_name_or_path": str(models / release / model / "pretrained" / "")}


def test_a_config_that_names_the_directory_asks_for_the_checkpoint_inside_it(tmp_path):
    directory = tmp_path / "pretrained"

    assert checkpoint_of(str(directory)) == directory / "model.pt"


def test_a_config_that_names_the_file_is_taken_at_its_word(tmp_path):
    named_file = tmp_path / "pretrained" / "model.pt"

    assert checkpoint_of(str(named_file)) == named_file


def test_a_model_that_is_here_is_not_mentioned(tmp_path):
    config = named(tmp_path)
    checkpoint = checkpoint_of(config["model_name_or_path"])
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"")

    assert missing_model_message(config) is None


def test_a_config_naming_no_model_says_nothing(tmp_path):
    assert missing_model_message({"model_name_or_path": ""}) is None
    assert missing_model_message({}) is None


def test_a_missing_model_is_named_with_its_release_and_the_variable_that_moves_it(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("MONTY_MODELS", raising=False)

    message = missing_model_message(named(tmp_path))

    assert f"`{MODEL}`" in message
    assert f"`{RELEASE}`" in message
    # Where it looked, so the person can tell whether it is the folder they meant.
    assert str(tmp_path / RELEASE / MODEL / "pretrained" / "model.pt") in message
    # And the one variable that decides that folder.
    assert "MONTY_MODELS" in message
    assert DEFAULT_MODELS_DIR in message


def test_the_models_folder_a_person_set_is_the_one_named(tmp_path, monkeypatch):
    models = tmp_path / "somewhere-else"
    monkeypatch.setenv("MONTY_MODELS", str(models))

    message = missing_model_message(named(tmp_path))

    assert f"MONTY_MODELS is set to {models}" in message


def test_a_path_that_is_not_a_releases_layout_is_a_model_a_run_trained(tmp_path):
    # A checkpoint somewhere of a person's own choosing: no download carries it,
    # so naming a release or the models folder would be advice about a file that
    # does not exist.
    config = {"model_name_or_path": str(tmp_path / "my_trained_models" / "run")}

    message = missing_model_message(config)

    assert "release" not in message.lower()
    assert "MONTY_MODELS" not in message
    assert "model_name_or_path" in message
