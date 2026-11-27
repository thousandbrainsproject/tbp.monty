"""Whether the pretrained model a config names is on this machine, and what to say when it is not.

Monty loads a checkpoint before the first episode, and a missing one used to
surface as a `FileNotFoundError` from inside `torch.load`, five frames deep in a
traceback that names neither the release the config asks for nor the variable
that decides where models live. A person reading that has to reconstruct three
facts the run already knew. This says them in one paragraph, before anything
starts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

#: Monty's own default, from `docs/how-to-use-monty/getting-started.md`.
DEFAULT_MODELS_DIR = "~/tbp/results/monty/pretrained_models"


def checkpoint_of(model_name_or_path: str) -> Path:
    """The file Monty loads, from the path a config names.

    A config may name the directory holding `model.pt` or the file itself
    (`MontyExperiment.setup_experiment` asks the same question), so a path whose
    last part is not `model.pt` gets it appended.
    """
    path = Path(str(model_name_or_path)).expanduser()
    if "model.pt" in path.parts:
        return path
    return path / "model.pt"


def release_of(checkpoint: Path) -> str | None:
    """The release a checkpoint sits in, when the layout is Monty's own.

    `…/<release>/<model>/pretrained/model.pt` is how every released archive
    unpacks; a path that does not look like that answers nothing rather than a
    guess.
    """
    parent = checkpoint.parent
    if parent.name != "pretrained":
        return None
    return parent.parent.parent.name


def model_of(checkpoint: Path) -> str | None:
    """The model's own directory name, when the layout is Monty's own."""
    parent = checkpoint.parent
    if parent.name != "pretrained":
        return None
    return parent.parent.name


def missing_model_message(config: Any) -> str | None:
    """One paragraph about the checkpoint that is not here, or `None` when it is.

    Args:
        config: an experiment config, as the composed Hydra config hands it over.

    Returns:
        The sentence to show, or `None` when the model the config names is on disk.
    """
    named = config.get("model_name_or_path") if hasattr(config, "get") else None
    if not named:
        return None

    checkpoint = checkpoint_of(named)
    if checkpoint.is_file():
        return None

    model = model_of(checkpoint) or "the model"
    release = release_of(checkpoint)

    # A released model arrives in an archive, and `MONTY_MODELS` is the folder
    # those archives unpack into — so the release's name is the thing to look up.
    # A path outside that layout is a model some run trained, which no download
    # carries: it has to be trained or pointed at.
    advice = (
        f"It is the `{release}` release. MONTY_MODELS is {_where_models_live()}, so "
        "unpack that release's archive there — Monty's getting-started page lists "
        "the archives — or set MONTY_MODELS to the folder that already holds it."
        if release
        else "A config may name a model a run here trained; point `model_name_or_path` "
        "at one, or run the experiment that writes it."
    )

    return f"Missing pretrained model: `{model}`, looked for at {checkpoint}. {advice}"


def _where_models_live() -> str:
    """Where models are looked for, said as the variable that decides it."""
    configured = os.environ.get("MONTY_MODELS")
    default = Path(DEFAULT_MODELS_DIR).expanduser()
    if configured and Path(configured).expanduser() != default:
        return f"set to {configured}"
    return f"unset, so Monty's default {DEFAULT_MODELS_DIR} is used"
