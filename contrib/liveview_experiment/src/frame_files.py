"""Sensor pictures as files, instead of blobs inside every update.

The page re-renders on each update and every picture it shows is embedded in the
markup as a ``data:`` URL. A view-finder frame is ~1.2 MB of base64, so each
1 Hz update carried megabytes to every connected socket; a tab that stops
reading — backgrounded, laptop asleep, a half-open connection — leaves that much
behind it per update, and a run of a few hours reached gigabytes of retained
markup. A file has a ~60-byte URL, and the browser fetches the picture once.

The files are named for the step they came from, so re-rendering the same step
is the same URL (and the browser may cache it) while a new step is a new name
(nothing stale can be shown). Only the last few of each kind are kept: the page
shows the latest, and the archive is not this module's job.

Where they go is ``$MONTY_LIVEVIEW_FRAMES``, or ``~/.cache/monty-liveview/frames``
— not the source tree, which is checked in.
"""

from __future__ import annotations

import base64
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

FRAMES_DIR_ENV = "MONTY_LIVEVIEW_FRAMES"
URL_PREFIX = "/frames"

#: How many pictures of one kind (and one run) stay on disk.
KEEP_PER_KIND = 8

_SAFE = re.compile(r"[^A-Za-z0-9_-]+")


def frames_dir() -> Path:
    """The directory the page's pictures are written to, created if missing."""
    configured = os.environ.get(FRAMES_DIR_ENV)
    root = (
        Path(configured)
        if configured
        else Path.home() / ".cache" / "monty-liveview" / "frames"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def publish_base64(
    kind: str,
    encoded: str | None,
    step: int,
    run_name: str = "",
) -> str | None:
    """Write one picture and answer the URL a page can use.

    Args:
        kind: ``"camera"`` or ``"depth"`` — part of the file name.
        encoded: The picture as base64 PNG, or None when this update has none.
        step: The step it came from; part of the name, so a new step never
            overwrites what a page has already asked for.
        run_name: The publisher's name, so two runs in one dashboard do not
            overwrite each other's pictures.

    Returns:
        The URL path, or None when the blob should be kept instead — a
        dashboard whose home is read-only, or a picture that is not valid
        base64, must still draw something.
    """
    if not encoded:
        return None

    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        logger.debug("picture for %s at step %s is not base64", kind, step)
        return None

    name = f"{_slug(run_name)}{kind}-{step}.png"
    try:
        _write_atomically(frames_dir() / name, data)
        _keep_latest(kind, run_name)
    except OSError as error:
        logger.debug("could not publish the %s picture: %s", kind, error)
        return None

    return f"{URL_PREFIX}/{name}"


def resolve(name: str) -> Path | None:
    """The file a URL names, or None if it is not one of ours.

    The name is the only thing a request carries, so it is the only thing that
    has to be checked: no separators, no parent directories, and it has to be a
    regular file inside the frames directory.
    """
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None

    candidate = (frames_dir() / name).resolve()
    if candidate.parent != frames_dir().resolve() or not candidate.is_file():
        return None
    return candidate


def _slug(run_name: str) -> str:
    return f"{_SAFE.sub('_', run_name)}_" if run_name else ""


def _write_atomically(target: Path, data: bytes) -> None:
    partial = target.with_suffix(".partial")
    partial.write_bytes(data)
    os.replace(partial, target)


def _keep_latest(kind: str, run_name: str) -> None:
    prefix = f"{_slug(run_name)}{kind}-"
    pictures = sorted(
        (
            (frame_of(path.name, prefix), path)
            for path in frames_dir().glob(f"{prefix}*.png")
        ),
        reverse=True,
    )

    for _step, stale in pictures[KEEP_PER_KIND:]:
        try:
            stale.unlink()
        except OSError:
            logger.debug("could not remove the stale picture %s", stale)


def frame_of(name: str, prefix: str) -> int:
    """The step a picture's name carries, or -1 when it carries none."""
    digits = name[len(prefix) : -len(".png")]
    return int(digits) if digits.isdigit() else -1
