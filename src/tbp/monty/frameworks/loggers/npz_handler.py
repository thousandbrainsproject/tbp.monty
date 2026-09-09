# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import logging
from collections.abc import Mapping
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import orjson

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.loggers.monty_handlers import MontyHandler
from tbp.monty.frameworks.models.buffer import BufferEncoder
from tbp.monty.frameworks.utils.logging_utils import maybe_rename_existing_dir

if TYPE_CHECKING:
    from collections.abc import Callable, Container, Iterator, Sequence

__all__ = [
    "LazyDict",
    "LazyList",
    "NpzHandler",
    "PathFilter",
    "load_episode",
    "materialize",
]

logger = logging.getLogger(__name__)

# Subdirectory of the output directory holding the per-episode files.
EXPERIMENT_SUBDIRECTORY = "telemetry"

# The npz member holding the episode's JSON.
INDEX_KEY = "__index__"

# In the JSON, an object with only this key stands in for an array stored as
# its own npz member: {"__array__": {"key", "dtype", "shape"}}.
ARRAY_REFERENCE = "__array__"


class NpzHandler(MontyHandler):
    """Writes detailed episode stats to npz files with optional filtering.

    Saves each episode's detailed stats to an npz file. The file contains JSON
    structured like DetailedJSONHandler's output but with every array replaced
    by a reference to it. The referenced arrays live alongside the JSON index
    in the npz file.

    See :class:`PathFilter` for the path and pattern syntax. From the command
    line, keys the logging group does not declare need the append prefix:
    ``+experiment.config.logging.exclude=[attention_system/goals]``.
    """

    def __init__(
        self,
        episodes: Container[int] | None = None,
        include: Sequence[str] = (),
        exclude: Sequence[str] = (),
        float_dtype: str | None = "float32",
        compressed: bool = True,
    ) -> None:
        """Initialize the handler.

        Args:
            episodes: The global episode ids to save; None or empty (default)
                saves every episode.
            include: Path patterns to keep; empty keeps everything.
                See :class:`PathFilter`.
            exclude: Path patterns to drop, applied after include.
            float_dtype: Cast floating-point arrays to this dtype (e.g.
                ``"float32"``) before writing; None keeps them as they are.
            compressed: Whether to deflate the npz members.
        """
        self._episodes = episodes
        self._path_filter = PathFilter(include, exclude)
        self._moved_previous_run = False
        self._float_dtype = float_dtype
        self._compressed = compressed

    @property
    def path_filter(self) -> PathFilter:
        return self._path_filter

    @classmethod
    def log_level(cls) -> str:
        return "DETAILED"

    @staticmethod
    def episode_stats(
        data: dict[str, Any],
        global_episode_id: int,
        local_episode: int,
        mode: ExperimentMode,
    ) -> dict[str, Any]:
        """Merge an episode's DETAILED blocks over its BASIC stats row.

        The same stats ``DetailedJSONHandler`` writes.

        Args:
            data: The logger's data, with ``BASIC`` and ``DETAILED`` pools.
            global_episode_id: Combined train+eval episode id, which keys the
                DETAILED pool.
            local_episode: Episode number within the mode, which keys the
                BASIC pool.
            mode: The experiment mode.

        Returns:
            The episode's stats.
        """
        basic = data["BASIC"][f"{mode}_stats"][local_episode]
        detailed = data["DETAILED"].get(local_episode)
        if detailed is None:
            detailed = data["DETAILED"][global_episode_id]
        return {**basic, **detailed}

    def close(self) -> None:
        pass

    def report_episode(
        self,
        data: Mapping[str, Any],
        output_dir: str,
        local_episode: int,
        mode: ExperimentMode = ExperimentMode.TRAIN,
        **kwargs,
    ) -> None:
        """Filter and write one episode's telemetry."""
        global_episode_id = kwargs[f"{mode}_episodes_to_total"][local_episode]

        if self._episodes and global_episode_id not in self._episodes:
            logger.debug(
                "Skipping telemetry for episode %s (not requested)",
                global_episode_id,
            )
            return

        stats = self.episode_stats(data, global_episode_id, local_episode, mode)
        episode = self._path_filter.prune(stats)

        telemetry_dir = Path(output_dir) / EXPERIMENT_SUBDIRECTORY
        if not self._moved_previous_run:
            maybe_rename_existing_dir(telemetry_dir)
            self._moved_previous_run = True
        telemetry_dir.mkdir(exist_ok=True, parents=True)
        path = telemetry_dir / f"episode_{global_episode_id:06d}.npz"
        self.write(global_episode_id, episode, path)

        logger.debug("Saved telemetry for episode %s to %s", global_episode_id, path)

    def write(self, episode_id: int, episode: Mapping[str, Any], path: Path) -> None:
        """Write one episode's file.

        Args:
            episode_id: The global episode id.
            episode: The episode's stats, holding only dicts, lists, arrays
                and scalars (see :meth:`PathFilter.prune`).
            path: The npz file to write, overwriting whatever is there.
        """
        npz_items: dict[str, np.ndarray] = {}

        def reference(array: np.ndarray, array_path: str) -> dict:
            # Optional floating-point conversion for storage efficiency.
            if self._float_dtype is not None and np.issubdtype(
                array.dtype, np.floating
            ):
                array = array.astype(self._float_dtype)
            # Store the array as its own member and return a reference, so
            # no array data rides as text in the JSON.
            npz_items[array_path] = array
            return {
                ARRAY_REFERENCE: {
                    "key": array_path,
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                }
            }

        index = _map_arrays(episode, reference)
        npz_items[INDEX_KEY] = np.frombuffer(
            _dumps({episode_id: index}), dtype=np.uint8
        )
        save = np.savez_compressed if self._compressed else np.savez
        save(path, **npz_items)


def _map_arrays(
    stats: Any,
    fn: Callable[[np.ndarray, str], Any],
    path: str = "",
) -> Any:
    """Rebuild stats with every array replaced by ``fn(array, path)``.

    The path of an array is its key path with list indices, e.g.
    ``SM_3/raw_observations/12/rgba``.

    Args:
        stats: The stats, or any part of them.
        fn: Maps an array and its path to the value that replaces it.
        path: The path of ``stats``; empty at the root.

    Returns:
        The rebuilt stats.
    """
    if isinstance(stats, np.ndarray):
        return fn(stats, path)
    if isinstance(stats, dict):
        return {
            key: _map_arrays(value, fn, _join_path(path, key))
            for key, value in stats.items()
        }
    if isinstance(stats, (list, tuple)):
        return [
            _map_arrays(item, fn, _join_path(path, index))
            for index, item in enumerate(stats)
        ]
    return stats


# Sentinel value that marks a part of the stats to be dropped by the filter.
_DROP = object()


class PathFilter:
    """Include/exclude filter over the paths of an episode's stats.

    **Paths** join the keys from the episode root with ``/``; a list item's
    key is its index. The root holds the episode's stats (``target``, ...)
    and one block per module: ``LM_<n>``, ``SM_<n>``, ``attention_system``,
    ``motor_system``::

        target
        SM_3/raw_observations/12/rgba
        attention_system/goals/12/locations

    **Patterns** are :func:`fnmatch.fnmatchcase` globs over whole paths:
    ``*`` matches any run of characters *including* ``/`` (so it covers
    list indices at any depth), ``?`` one character, ``[abc]`` a
    one-character class (not a comma list). A pattern that matches a path
    selects everything below it::

        attention_system                     the whole block
        SM_*/raw_observations/*/depth        one field of every observation
        LM_[01]/evidences                    learning modules 0 and 1

    **Decisions.** A node is kept when an include pattern matches it or an
    ancestor (or there are no include patterns) and no exclude pattern
    matches it or an ancestor; exclude wins. Containers emptied by the
    filter are dropped::

        include=["target", "LM_2"]
            -> the target and learning module 2, nothing else
        exclude=["SM_*/raw_observations/*/semantic_3d"]
            -> everything except that field
    """

    def __init__(
        self, include: Sequence[str] = (), exclude: Sequence[str] = ()
    ) -> None:
        """Initialize the filter.

        Args:
            include: Patterns of the paths to keep; empty keeps everything.
            exclude: Patterns of the paths to drop; they win over include.
        """
        self.include = tuple(str(pattern) for pattern in include)
        self.exclude = tuple(str(pattern) for pattern in exclude)

    def matches_include(self, path: str) -> bool:
        return any(fnmatchcase(path, pattern) for pattern in self.include)

    def matches_exclude(self, path: str) -> bool:
        return any(fnmatchcase(path, pattern) for pattern in self.exclude)

    def prune(self, stats: Mapping[str, Any]) -> dict[str, Any]:
        """Expand an episode's stats and keep only the paths this filter allows.

        Objects the stats hold beyond dicts, lists, arrays and scalars (goals,
        voxel grids, dataclasses, quaternions, ...) are expanded through
        :class:`BufferEncoder` so the filter can see inside them and the
        result holds only dicts, lists, arrays and scalars. Containers emptied
        by the filter are dropped; a list keeps its length, with ``None``
        standing in for dropped items, so indices stay meaningful.

        Args:
            stats: An episode's stats.

        Returns:
            The expanded, filtered stats.
        """
        pruned = self._prune(stats, "", included=not self.include)
        return {} if pruned is _DROP else pruned

    def _prune(self, obj: Any, path: str, included: bool) -> Any:
        # The walk visits ancestors before descendants, so an ancestor's
        # include match is carried down and an exclude match ends the walk
        # right here.
        if path:
            if self.matches_exclude(path):
                return _DROP
            included = included or self.matches_include(path)

        if isinstance(obj, Mapping):
            if not obj:
                return obj if included else _DROP
            kept = {}
            for key, value in obj.items():
                pruned = self._prune(value, _join_path(path, key), included)
                if pruned is not _DROP:
                    kept[key] = pruned
            return kept or _DROP

        if isinstance(obj, (list, tuple)):
            if not obj or all(_is_atomic(item) for item in obj):
                return obj if included else _DROP
            pruned = [
                self._prune(item, _join_path(path, index), included)
                for index, item in enumerate(obj)
            ]
            if all(item is _DROP for item in pruned):
                return _DROP
            return [None if item is _DROP else item for item in pruned]

        if _is_atomic(obj):
            return obj if included else _DROP

        try:
            expanded = BufferEncoder().default(obj)
        except TypeError:
            # Nothing knows how to expand it; the serializer gets to complain.
            return obj if included else _DROP
        return self._prune(expanded, path, included)


def _dumps(payload: Any) -> bytes:
    """Serialize episode stats.

    Arrays serialize straight from their buffers; everything else
    (dataclasses, quaternions, rotations, ...) goes through
    :class:`BufferEncoder`. NaN is written as null, and int dict keys (the
    episode id) are written as strings.

    Args:
        payload: The JSON-encodable payload.

    Returns:
        The serialized payload.
    """
    return orjson.dumps(
        payload,
        default=BufferEncoder().default,
        option=orjson.OPT_SERIALIZE_NUMPY
        | orjson.OPT_NON_STR_KEYS
        | orjson.OPT_PASSTHROUGH_DATACLASS,
    )


def _join_path(path: str, key: Any) -> str:
    return str(key) if not path else f"{path}/{key}"


def _is_atomic(obj: Any) -> bool:
    return obj is None or isinstance(
        obj, (bool, int, float, str, bytes, np.ndarray, np.generic)
    )


def load_episode(path: str | Path, lazy: bool = False) -> dict[str, Any]:
    """Load an episode written by :class:`NpzHandler`.

    Args:
        path: The episode's ``.npz`` file.
        lazy: Read array members from the file on first access instead of
            all at once. The stats then hold :class:`LazyDict` /
            :class:`LazyList` nodes (``dict`` / ``list`` subclasses) that
            resolve a reference the first time it is looked up, iterated or
            unpacked, and keep the file open as long as they live; copying
            or pickling one resolves everything below it into plain dicts
            and lists. Only C code that reads a dict's items without
            calling ``__getitem__`` (``orjson.dumps``) sees unresolved
            references.

    Returns:
        ``{"<episode_id>": {<episode stats>}}`` with every array reference
        replaced by the array.
    """
    npz = np.load(path)
    index = orjson.loads(npz[INDEX_KEY].tobytes())
    if lazy:
        return LazyDict(index, npz)
    with npz:
        return resolve(index, npz)


class LazyDict(dict):
    """A dict that resolves array references from an open npz on access.

    Each lookup replaces the looked-up value in place -- a reference by its
    array, a plain child dict or list by a lazy one -- so a value is read
    from the file once and is an ordinary object afterwards. Whole-dict
    access (``items``, ``values``, iteration, ``**`` unpacking, ``dict(...)``)
    resolves every value at this level first.
    """

    __slots__ = ("_npz",)

    def __init__(self, items: Mapping[str, Any], npz: Mapping[str, np.ndarray]) -> None:
        super().__init__(items)
        self._npz = npz

    def __getitem__(self, key: Any) -> Any:
        value = super().__getitem__(key)
        resolved = _resolve_lazy(value, self._npz)
        if resolved is not value:
            super().__setitem__(key, resolved)
        return resolved

    def get(self, key: Any, default: Any = None) -> Any:
        # dict.get bypasses __getitem__, so resolve here.
        if key in self:
            return self[key]
        return default

    def __iter__(self) -> Iterator:
        # Overriding __iter__ also steers dict(...) and {**...} away from
        # the C fast path that would copy raw references.
        return iter(list(super().__iter__()))

    def _resolve_all(self) -> None:
        for key in list(super().__iter__()):
            self[key]

    def items(self) -> Any:
        self._resolve_all()
        return super().items()

    def values(self) -> Any:
        self._resolve_all()
        return super().values()

    def copy(self) -> dict:
        self._resolve_all()
        return dict(super().items())

    def __reduce__(self) -> tuple:
        # Pickles and deep-copies become plain, fully loaded dicts.
        return (dict, (materialize(self),))


class LazyList(list):
    """A list whose dict and list items are lazy; see :class:`LazyDict`.

    Array references directly in the list are resolved when the list is
    created, so C code reading list items directly (``np.asarray``) never
    sees one; nested dicts and lists stay lazy.
    """

    __slots__ = ("_npz",)

    def __init__(self, items: list, npz: Mapping[str, np.ndarray]) -> None:
        super().__init__(
            _resolve_lazy(item, npz) if _is_reference(item) else item for item in items
        )
        self._npz = npz

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return LazyList(super().__getitem__(index), self._npz)
        value = super().__getitem__(index)
        resolved = _resolve_lazy(value, self._npz)
        if resolved is not value:
            super().__setitem__(index, resolved)
        return resolved

    def __iter__(self) -> Iterator:
        return (self[index] for index in range(len(self)))

    def __reduce__(self) -> tuple:
        return (list, (materialize(self),))


def materialize(stats: Any) -> Any:
    """A fully loaded, plain-dict-and-list copy of (possibly lazy) stats.

    Args:
        stats: Stats from :func:`load_episode`, lazy or not.

    Returns:
        The same tree with every reference resolved and no lazy nodes.
    """
    if isinstance(stats, dict):
        return {key: materialize(value) for key, value in stats.items()}
    if isinstance(stats, list):
        return [materialize(item) for item in stats]
    return stats


def _is_reference(value: Any) -> bool:
    return isinstance(value, dict) and value.keys() == {ARRAY_REFERENCE}


def _resolve_lazy(value: Any, npz: Mapping[str, np.ndarray]) -> Any:
    if isinstance(value, (LazyDict, LazyList)):
        return value
    if _is_reference(value):
        return npz[value[ARRAY_REFERENCE]["key"]]
    if isinstance(value, dict):
        return LazyDict(value, npz)
    if isinstance(value, list):
        return LazyList(value, npz)
    return value


def resolve(stats: Any, arrays: Mapping[str, np.ndarray]) -> Any:
    """Replace the array references in loaded stats with the arrays themselves.

    Args:
        stats: Loaded JSON stats.
        arrays: The arrays the references point to, by key; typically the
            open npz file.

    Returns:
        The stats with every reference replaced by its array.
    """

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if node.keys() == {ARRAY_REFERENCE}:
                return arrays[node[ARRAY_REFERENCE]["key"]]
            return {key: _resolve(value) for key, value in node.items()}
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    return _resolve(stats)
