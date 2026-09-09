# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Plot-ready views of an episode: pure data, no matplotlib.

Every view takes an :class:`~analysis.telemetry.EpisodeTelemetry` and returns a
:class:`Stream` -- one value per recorded step, each tagged with the episode
step it belongs to. Modules tick at different rates (a learning module
records evidence only on the steps it processed), but everything one
module records rides the same tick. So every stream carries its module's
steps, the animation clock is the episode step, and a
panel shows the ``latest`` tick at the clock -- a record that holds nothing
(no mask, no region) is an empty value on its tick, not a gap.

This is the only module that knows where things live in the stats; the
panels only know how to draw a stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from analysis.telemetry import EpisodeTelemetry

__all__ = [
    "Points",
    "Stream",
    "Tick",
    "attention_grid_stream",
    "attention_region_stream",
    "evidence_stream",
    "fixation_point_stream",
    "goal_point_stream",
    "hypothesis_count",
    "max_evidence",
    "recognized_steps",
    "rgba_stream",
    "salience_map_stream",
    "segmentation_map_stream",
    "segmentation_overlay_stream",
]

T = TypeVar("T")
U = TypeVar("U")


@dataclass(frozen=True)
class Tick(Generic[T]):
    """One step of a stream, as :meth:`Stream.latest` returns it.

    Attributes:
        index: The recording's position in the stream's own coordinates
            (e.g. the LM step for a learning module's stream).
        value: The recorded value.
        fresh: Whether the value was recorded exactly at the queried
            episode step, rather than held over from an earlier one.
    """

    index: int
    value: T
    fresh: bool


@dataclass(frozen=True)
class Stream(Generic[T]):
    """Values recorded at some of an episode's steps.

    Attributes:
        values: One per recording, in order.
        steps: The episode step of each recording, increasing.
    """

    values: Sequence[T]
    steps: np.ndarray

    def __len__(self) -> int:
        """How many recorded values in the data that the stream pulls from.

        Returns:
            The record count.
        """
        return len(self.values)

    def __iter__(self) -> Iterator[T]:
        return iter(self.values)

    @property
    def n_steps(self) -> int:
        """How many episode steps the stream spans."""
        return int(self.steps[-1]) + 1 if len(self.steps) else 0

    def latest(self, step: int) -> Tick[T] | None:
        """The latest tick at or before an episode step.

        Returns:
            The tick, or None before the first recording.
        """
        index = int(np.searchsorted(self.steps, step, side="right")) - 1
        if index < 0:
            return None
        return Tick(index, self.values[index], fresh=bool(self.steps[index] == step))

    def map(self, fn: Callable[[T], U]) -> Stream[U]:
        """A stream of ``fn`` of every value, at the same steps.

        Returns:
            The mapped stream.
        """
        return Stream([fn(value) for value in self], self.steps)


# -- sensor module images --------------------------------------------------


def rgba_stream(ep: EpisodeTelemetry, sm: str) -> Stream[np.ndarray]:
    """A sensor module's ``(H, W, 4)`` frames.

    Returns:
        The frames.
    """
    rgba = ep(f"{sm}/raw_observations/*/rgba")
    return Stream(rgba, ep.episode_steps(f"{sm}/raw_observations"))


def segmentation_map_stream(ep: EpisodeTelemetry, sm: str) -> Stream[np.ndarray | None]:
    """A sensor module's segmentation masks; None on steps it produced none.

    Returns:
        The ``(H, W)`` masks.
    """
    masks = ep(f"{sm}/segmentation_maps/*")
    masks = masks if isinstance(masks, np.ndarray) else list(masks.values())
    return Stream(masks, ep.episode_steps(f"{sm}/segmentation_maps"))


def segmentation_overlay_stream(
    ep: EpisodeTelemetry,
    sm: str,
    color: tuple[int, int, int, int] = (0, 255, 60, 110),
) -> Stream[np.ndarray]:
    """A module's segmentation masks as transparent overlay images.

    One per recorded step, at the module's rate: the ``color`` where that
    step's mask is, fully transparent on steps that produced none -- so a
    held frame carries its own mask, never an earlier one.

    Args:
        ep: The episode.
        sm: The sensor module's name.
        color: The mask's RGBA, translucent green by default; the alpha is
            its translucency over the frame.

    Returns:
        ``(H, W, 4)`` uint8 images; empty when no step recorded a mask.
    """
    masks = segmentation_map_stream(ep, sm)
    shape = next((np.shape(mask) for mask in masks if mask is not None), None)
    if shape is None:
        return Stream([], np.array([], dtype=int))

    def overlay(mask: np.ndarray | None) -> np.ndarray:
        image = np.zeros((*shape, 4), np.uint8)
        if mask is not None:
            image[np.asarray(mask) > 0] = color
        return image

    return masks.map(overlay)


def salience_map_stream(ep: EpisodeTelemetry, sm: str) -> Stream[np.ndarray]:
    """A sensor module's ``(H, W)`` salience maps.

    Returns:
        The maps.
    """
    maps = ep(f"{sm}/salience_maps/*")
    return Stream(maps, ep.episode_steps(f"{sm}/salience_maps"))


def fixation_point_stream(ep: EpisodeTelemetry, sm: str) -> Stream[np.ndarray]:
    """The world location a module fixates: the center pixel of each frame.

    Returns:
        ``(3,)`` locations, from ``semantic_3d`` (world-frame pixel
        locations) at the frame's center -- the sensor patch is centered
        on what it fixates.
    """
    height, width = ep(f"{sm}/raw_observations/0/rgba").shape[:2]
    locations = ep(f"{sm}/raw_observations/*/semantic_3d")[..., :3]
    centers = locations[:, (height // 2) * width + width // 2]
    return Stream(centers, ep.episode_steps(f"{sm}/raw_observations"))


# -- point sets: a cloud of points per recorded step -------------------------


@dataclass(frozen=True)
class Points:
    """A point cloud with named per-point feature columns.

    Attributes:
        locations: The ``(N, 3)`` point locations.
        features: Per feature name, its ``(N,)`` values, indexable as
            ``points["weight"]``.
    """

    locations: np.ndarray
    features: dict[str, np.ndarray]

    @classmethod
    def concat(cls, points: Iterable[Points]) -> Points:
        """Every cloud's points in one, for bounds and shared color scales.

        Returns:
            The concatenated cloud; empty when there are no points.
        """
        populated = [p for p in points if len(p)]
        if not populated:
            return cls(np.empty((0, 3)), {})
        return cls(
            np.vstack([p.locations for p in populated]),
            {
                name: np.concatenate([p[name] for p in populated])
                for name in populated[0].features
            },
        )

    def __len__(self) -> int:
        return len(self.locations)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.features[name]


def _point_stream(
    ep: EpisodeTelemetry, path: str, xyz: str | Callable[[Any], Any], **features: str
) -> Stream[Points]:
    # One cloud per record on a "module/field" path, at the module's rate; a
    # record holding nothing is an empty cloud, not a gap. xyz is the record
    # field holding the (N, 3) locations (or a function of the record), each
    # feature the field holding its values.
    records = ep(path)
    empty = Points(np.empty((0, 3)), {name: np.empty(0) for name in features})
    values = [
        empty
        if record is None
        else Points(
            np.asarray(
                xyz(record) if callable(xyz) else record[xyz], dtype=float
            ).reshape(-1, 3),
            {
                name: np.asarray(record[field], dtype=float)
                for name, field in features.items()
            },
        )
        for record in records
    ]
    return Stream(values, ep.episode_steps(path))


def goal_point_stream(
    ep: EpisodeTelemetry, path: str = "motor_system/goals_in"
) -> Stream[Points]:
    """Goals as clouds with a ``confidence`` feature.

    Args:
        ep: The episode.
        path: The recorded goal columns: the motor system's (what passed
            the attention filter) by default, or a sensor module's
            proposals (``"SM_3/goals"``).

    Returns:
        One cloud per step the module recorded.
    """
    return _point_stream(ep, path, xyz="locations", confidence="confidences")


def attention_region_stream(ep: EpisodeTelemetry, module: str) -> Stream[Points]:
    """A module's proposed attention regions as clouds with a ``weight`` feature.

    Works for sensor and learning modules alike; a learning module's
    proposals land on the episode steps it processed.

    Returns:
        One cloud per step the module recorded; empty on steps it proposed
        nothing.
    """
    return _point_stream(
        ep, f"{module}/attention_regions", xyz="locations", weight="weights"
    )


def recognized_steps(ep: EpisodeTelemetry, lm: str) -> np.ndarray:
    """Episode steps from which a module had recognized its object.

    Read from the buffer's individual terminal state: the module records
    the matching step at which it first reached its terminal state
    (``individual_ts_reached_at_step``) and the object it detected
    (``individual_ts_object`` -- None unless the state was a match).

    Returns:
        The episode steps from recognition onward; empty when the module
        never recognized.
    """
    block = ep.blocks.get(lm, {})
    if block.get("individual_ts_object") is None:
        return np.array([], dtype=int)
    processed = np.asarray(block.get("lm_processed_steps", ()), dtype=bool)
    return np.flatnonzero(processed)[int(block["individual_ts_reached_at_step"]) :]


def attention_grid_stream(
    ep: EpisodeTelemetry, grids: str = "attention_system/grids"
) -> Stream[Points]:
    """The attention system's voxels as clouds with a ``weight`` feature.

    Voxel indices are scaled to world units as they are (lower corners;
    no center offset for now).

    Args:
        ep: The episode.
        grids: The recorded grids: the persistent grid by default, or
            ``"attention_system/proposed_grids"``.

    Returns:
        One cloud per step.
    """
    size = ep("attention_system/voxel_size")
    return _point_stream(
        ep,
        grids,
        xyz=lambda grid: np.asarray(grid["voxels"]) * size,
        weight="weight",
    )


# -- learning module evidence ------------------------------------------------


def evidence_stream(ep: EpisodeTelemetry, lm: str) -> Stream[dict[str, np.ndarray]]:
    """Per processed step, each object's hypothesis evidences.

    Chain with :func:`max_evidence` or :func:`hypothesis_count` for the
    plot-ready streams.

    Returns:
        Per LM step, object name to its hypotheses' evidence; empty when
        the module scored nothing.
    """
    # Per (LM step, object): the evidence of every hypothesis on that object.
    evidences = ep(f"{lm}/evidences/{{step}}/{{object}}")
    n_steps = len(ep(f"{lm}/evidences"))
    ticks: list[dict[str, np.ndarray]] = [{} for _ in range(n_steps)]
    for (step, name), values in evidences.items():
        ticks[step][name] = values
    return Stream(ticks, ep.episode_steps(f"{lm}/evidences"))


def max_evidence(tick: Mapping[str, np.ndarray]) -> dict[str, float]:
    """A tick's strongest evidence per object, for ``Stream.map``.

    Returns:
        Object name to its max evidence; an object without hypotheses
        drops out rather than scoring zero.
    """
    return {name: float(values.max()) for name, values in tick.items() if len(values)}


def hypothesis_count(tick: Mapping[str, np.ndarray]) -> int:
    """How many hypotheses a tick scored across its objects, for ``Stream.map``.

    Returns:
        The count.
    """
    return sum(len(values) for values in tick.values())
