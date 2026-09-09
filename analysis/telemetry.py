# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""One episode's recorded telemetry, addressed by path.

``EpisodeTelemetry.load("debug_3lm")`` opens an episode lazily; members are read
from the file as they are asked for. Data is asked for by path, where a
segment may be a literal key or index, or a placeholder (``*`` or
``{name}``) standing for every key or index there is::

    ep("SM_3/raw_observations/*/rgba")          # (S, H, W, 4): the steps stack
    ep("SM_3/salience_maps/40")                  # (H, W): one value
    ep("LM_2/evidences/{step}/{object}")         # {(step, object): (N_obj,)}
    ep("LM_2/evidences/{step}/{object}", object="023_mug")   # {(step,): (N_obj,)}
    ep("*/lm_processed_steps")                   # {("LM_0",): (S,), ...}

A free placeholder over a list (steps) stacks the values into a leading
axis when they are arrays of one shape; over a dict (names) it keys them.
When stacking is not possible the values come back in a dict keyed by the
placeholders' values. Numbers stored as lists come back as arrays.

Modules are named by block: ``"SM_<n>"`` / ``"LM_<n>"``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from tbp.monty.frameworks.loggers.npz_handler import (
    EXPERIMENT_SUBDIRECTORY,
    load_episode,
)

if TYPE_CHECKING:
    import os
    from collections.abc import Iterator, Mapping

__all__ = [
    "RESULTS_DIR",
    "EpisodeTelemetry",
    "available_episodes",
    "resolve_run_dir",
]

# Where the eval configs write their runs (their ``output_dir``).
RESULTS_DIR = Path("~/tbp/results/comp_benefits_figures").expanduser()

# A placeholder is a yet-to-be resolved component in a data path.
# Example: 'SM_3/raw_observations/*/{dataset}' has both kinds of
# placeholders -- * for the global wildcard, and {dataset} for a named capture.
_PLACEHOLDER = re.compile(r"\{(\w+)\}|\*")


def resolve_run_dir(run: str | os.PathLike) -> Path:
    """Find an experiment's output directory from a path or a run name.

    Args:
        run: An absolute directory, or the name of a run under
            :data:`RESULTS_DIR`.

    Returns:
        The directory.

    Raises:
        FileNotFoundError: If it does not exist.
    """
    # Joining onto an absolute path yields that path, so one join covers both.
    run_dir = RESULTS_DIR / Path(run).expanduser()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"no experiment directory {run_dir}")
    return run_dir


def available_episodes(run_dir: str | os.PathLike) -> list[int]:
    """List the episode numbers recorded in an experiment directory.

    Args:
        run_dir: Experiment output directory.

    Returns:
        Sorted episode numbers; empty if none were recorded.
    """
    telemetry_dir = Path(run_dir) / EXPERIMENT_SUBDIRECTORY
    return sorted(
        int(path.stem.removeprefix("episode_"))
        for path in telemetry_dir.glob("episode_*.npz")
    )


class EpisodeTelemetry:
    """One episode's recorded telemetry (see the module docstring)."""

    def __init__(self, blocks: Mapping[str, Any]) -> None:
        self.blocks = blocks

    @classmethod
    def load(cls, run: str | os.PathLike, episode: int = 0) -> EpisodeTelemetry:
        """Open an episode of a run, reading members from the file on access.

        Args:
            run: The run's directory, or its name under :data:`RESULTS_DIR`.
            episode: The episode number.

        Returns:
            The episode.

        Raises:
            FileNotFoundError: If the run or the episode does not exist.
        """
        telemetry_dir = resolve_run_dir(run) / EXPERIMENT_SUBDIRECTORY
        path = telemetry_dir / f"episode_{episode:06d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"no telemetry for episode {episode} at {path}")
        return cls(load_episode(path, lazy=True)[str(episode)])

    @property
    def sensor_modules(self) -> list[str]:
        """Names of the sensor modules that recorded telemetry, by index."""
        names = [
            k
            for k in self.blocks
            if k.startswith("SM_") and k.removeprefix("SM_").isdigit()
        ]
        return sorted(names, key=lambda name: int(name.removeprefix("SM_")))

    @property
    def learning_modules(self) -> list[str]:
        """Names of the learning modules, by index."""
        names = [
            k
            for k in self.blocks
            if k.startswith("LM_") and k.removeprefix("LM_").isdigit()
        ]
        return sorted(names, key=lambda name: int(name.removeprefix("LM_")))

    def episode_steps(self, path: str) -> np.ndarray:
        """The episode step of each record target dataset.

        This method exists because LMs aren't always stepped at every episode step.
        In general, each LM's telemetry is on its own timeline. This method maps
        its timeline onto the global episode timeline by using the `lm_processed_steps`
        telemetry.

        Args:
            path: The record list's path, e.g. ``"LM_2/evidences"``; its
                first segment names the component.

        Returns:
            The episode steps, one per record, increasing.

        Raises:
            KeyError: If a learning module's block holds no
                ``lm_processed_steps``.
        """
        component = path.split("/", 1)[0]
        n = len(self(path))
        block = self.blocks.get(component, {})
        if component.startswith("LM_"):
            try:
                processed = np.asarray(block["lm_processed_steps"], dtype=bool)
            except KeyError as error:
                raise KeyError(
                    f"cannot derive episode steps for {component} telemetry"
                    " without 'lm_processed_steps'"
                ) from error
            if 0 < n <= np.count_nonzero(processed):
                return np.flatnonzero(processed)[:n]
        return np.arange(n)

    def __call__(self, path: str, **bound: Any) -> Any:
        """The data at a path (see the module docstring).

        Args:
            path: Keys and indices joined with ``/``; ``*`` or ``{name}``
                for every key or index at that point.
            **bound: A key or index for a ``{name}`` placeholder, so it is
                indexed rather than iterated.

        Returns:
            The value when nothing is free; else the values stacked along
            the free placeholders when those index lists and the values are
            arrays (or scalars) of one shape, or a dict keyed by the free
            placeholders' values.

        Example:
            One sensor module's frames, one of them, and each LM's evidence
            for one object at step 3::

                ep("SM_3/raw_observations/*/rgba").shape    # (100, 256, 256, 4)
                ep("SM_3/raw_observations/12/rgba").shape   # (256, 256, 4)
                ep("{lm}/evidences/3/{object}", object="023_mug")
                # {("LM_0",): array([...]), ("LM_2",): array([...])}

            The first stacks because ``*`` walked list indices and every
            frame has the same shape; the last is a dict because ``{lm}``
            walked the dict of blocks. Binding ``object`` indexes it, so it
            does not appear in the keys -- and ``LM_1``, which has no
            evidence for that object, is skipped rather than an error.
        """
        segments = path.split("/")
        found = dict(_walk(self.blocks, segments, bound, ()))
        if not any(_is_free(segment, bound) for segment in segments):
            return _as_array(found[()])
        values = [_as_array(value) for value in found.values()]
        ordinal = all(isinstance(index, int) for key in found for index in key)
        if values and ordinal:
            if all(np.isscalar(value) for value in values):
                return np.array(values)
            arrays = all(isinstance(value, np.ndarray) for value in values)
            if arrays and len({value.shape for value in values}) == 1:
                return np.stack(values)
        return dict(zip(found, values))


def _is_free(segment: str, bound: Mapping[str, Any]) -> bool:
    # i.e., it's a placeholder and it'll still be unresolved after applying `bound`.
    match = _PLACEHOLDER.fullmatch(segment)
    return match is not None and match.group(1) not in bound


def _walk(
    node: Any,
    segments: list[str],
    bound: Mapping[str, Any],
    key: tuple,
) -> Iterator[tuple[tuple, Any]]:
    # Yields (placeholder values, value) for every way of following the
    # segments; a branch missing the rest of the path is skipped.
    if not segments:
        yield key, node
        return
    segment, rest = segments[0], segments[1:]
    if _is_free(segment, bound):
        children = node.items() if isinstance(node, dict) else enumerate(node)
        for child_key, child in children:
            try:
                yield from _walk(child, rest, bound, (*key, child_key))
            except (KeyError, IndexError, TypeError):
                continue
        return
    match = _PLACEHOLDER.fullmatch(segment)
    index = bound[match.group(1)] if match else segment
    child = node[index] if isinstance(node, dict) else node[int(index)]
    yield from _walk(child, rest, bound, key)


def _as_array(value: Any) -> Any:
    # Numbers, booleans and strings stored as (nested) lists become arrays;
    # anything else is returned as is.
    if isinstance(value, np.ndarray) or not isinstance(value, (list, tuple)):
        return value
    try:
        array = np.asarray(value)
    except ValueError:
        return value
    return value if array.dtype.kind == "O" else array
