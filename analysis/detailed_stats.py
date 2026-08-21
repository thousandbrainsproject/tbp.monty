# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Loading helpers for detailed run stats.

A trimmed-down, self-contained version of ``monty_utils.detailed_stats``
(from ~/tbp/monty_utils), so the analysis scripts run inside this repo's
environment without an extra dependency.

Detailed stats are written by the DetailedJSONHandler either as one
``detailed_run_stats.json`` with one JSON line per episode, or -- with
``detailed_save_per_episode`` -- as a ``detailed_run_stats/`` directory of
``episode_NNNNNN.json`` files. Either way each episode is wrapped in a
one-item ``{episode_number: episode_data}`` dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import orjson

if TYPE_CHECKING:
    import os

# Voxel coordinates are lower corners; offset to centres for plotting.
VOXEL_CENTRE_OFFSET = 0.5

# Runs from before the attention system exported its geometry hold no
# voxel_size; this is the value MontyBase constructs the system with.
DEFAULT_VOXEL_SIZE = 0.05


def _loads(data: bytes) -> dict:
    """Parse detailed-stats JSON bytes.

    Files written since the DetailedJSONHandler switched to orjson
    (2026-08-19) are strict JSON and parse directly. Stdlib-era files may
    hold bare NaN tokens, which are not valid JSON and orjson refuses; on a
    parse failure those are rewritten to null and parsing is retried.
    Stdlib only ever emitted the token after "[" or a separator space, so
    only those two byte contexts are rewritten -- quoted strings merely
    containing "NaN" cannot match either. NaN entries load as None rather
    than float nan (matching what orjson-written files hold).

    Returns:
        The parsed document.

    """
    try:
        return orjson.loads(data)
    except orjson.JSONDecodeError:
        data = data.replace(b"[NaN", b"[null").replace(b" NaN", b" null")
        return orjson.loads(data)


def load_episode_stats(exp_dir: os.PathLike, episode: int = 0) -> DetailedStats:
    """Load one episode's detailed stats from an experiment directory.

    Args:
        exp_dir: Experiment output directory (e.g. ``.../monty_runs/potato``).
        episode: Episode number to load.

    Returns:
        The episode's stats, keyed by module ("SM_1", "LM_0",
        "attention_system", "motor_system", "target").

    Raises:
        FileNotFoundError: If the directory holds no detailed stats.
    """
    exp_dir = Path(exp_dir)

    per_episode_dir = exp_dir / "detailed_run_stats"
    if per_episode_dir.is_dir():
        path = per_episode_dir / f"episode_{episode:06d}.json"
        data = _loads(path.read_bytes())
        return DetailedStats(data[str(episode)])

    single_file = exp_dir / "detailed_run_stats.json"
    if single_file.is_file():
        with single_file.open("rb") as f:
            for line_number, line in enumerate(f):
                if line_number == episode:
                    return DetailedStats(_loads(line)[str(episode)])
        raise FileNotFoundError(f"Episode {episode} not found in {single_file}")

    raise FileNotFoundError(f"No detailed stats found under {exp_dir}")


def available_episodes(exp_dir: os.PathLike) -> list[int]:
    """List the episode numbers recorded in an experiment directory.

    Args:
        exp_dir: Experiment output directory.

    Returns:
        Sorted episode numbers; empty if no detailed stats exist.
    """
    exp_dir = Path(exp_dir)

    per_episode_dir = exp_dir / "detailed_run_stats"
    if per_episode_dir.is_dir():
        return sorted(
            int(p.stem.removeprefix("episode_"))
            for p in per_episode_dir.glob("episode_*.json")
            if "_old" not in p.stem
        )

    single_file = exp_dir / "detailed_run_stats.json"
    if single_file.is_file():
        with single_file.open() as f:
            return list(range(sum(1 for _ in f)))

    return []


class DetailedStats(dict):
    """One episode's detailed stats, with extraction helpers attached.

    Subclasses dict, so it behaves exactly like the plain episode dict
    (keyed by module: "SM_3", "LM_0", "attention_system", "motor_system",
    "target"); the methods extract plot-ready arrays from the module blocks.
    ``load_episode_stats`` returns this class.
    """

    @classmethod
    def load(cls, exp_dir: os.PathLike, episode: int = 0) -> DetailedStats:
        """Load one episode's detailed stats from an experiment directory.

        Args:
            exp_dir: Experiment output directory.
            episode: Episode number to load.

        Returns:
            The loaded episode's stats.
        """
        return load_episode_stats(exp_dir, episode)

    @staticmethod
    def _sm_key(sensor_module_id: int | str) -> str:
        """Normalize a sensor module id to its stats key.

        Returns:
            The stats key ("SM_<n>").

        """
        if isinstance(sensor_module_id, int):
            return f"SM_{sensor_module_id}"
        return sensor_module_id

    def _sm_key_or_default(self, sensor_module_id: int | str | None) -> str:
        """Normalize a sensor module id, defaulting to the last recorded one.

        Returns:
            The stats key ("SM_<n>").

        """
        if sensor_module_id is None:
            sensor_module_id = self.sm_ids[-1]
        return self._sm_key(sensor_module_id)

    def _module_ids(self, prefix: str) -> list[str]:
        """List a module type's recorded stats keys, sorted numerically.

        Returns:
            The matching keys, e.g. ["SM_2", "SM_10"] for prefix "SM_".

        """
        return sorted(
            (key for key in self if key.startswith(prefix)),
            key=lambda key: int(key.removeprefix(prefix)),
        )

    def _raw_observations(self, sensor_module_id: int | str) -> list[dict]:
        """Return a sensor module's raw observations.

        Returns:
            One raw observation dict per recorded step.

        """
        return self[self._sm_key(sensor_module_id)]["raw_observations"]

    @property
    def sm_ids(self) -> list[str]:
        """The sensor module ids recorded in this episode, sorted."""
        return self._module_ids("SM_")

    @property
    def lm_ids(self) -> list[str]:
        """The learning module ids recorded in this episode, sorted."""
        return self._module_ids("LM_")

    def rgba(self, sensor_module_id: int | str) -> np.ndarray:
        """Stack a sensor module's raw rgba frames into one array.

        Args:
            sensor_module_id: Which sensor module to read.

        Returns:
            A ``(num_frames, H, W, 4)`` uint8 array.
        """
        return np.stack(
            [np.array(obs["rgba"]) for obs in self._raw_observations(sensor_module_id)]
        ).astype(np.uint8)

    def depth(self, sensor_module_id: int | str) -> np.ndarray:
        """Stack a sensor module's raw depth frames into one array.

        Args:
            sensor_module_id: Which sensor module to read.

        Returns:
            A ``(num_frames, H, W)`` float array.
        """
        return np.stack(
            [
                np.asarray(obs["depth"], dtype=float)
                for obs in self._raw_observations(sensor_module_id)
            ]
        )

    def sensor_shape(self, sensor_module_id: int | str) -> tuple[int, int]:
        """Extract the shape of a sensor module's raw images.

        Args:
            sensor_module_id: Which sensor module to read.

        Returns:
            A ``(H, W)`` tuple representing the height and width of the
            images.
        """
        raw_observations = self._raw_observations(sensor_module_id)
        return np.asarray(raw_observations[0]["rgba"]).shape[:2]

    def semantic_3d(self, sensor_module_id: int | str) -> np.ndarray:
        """Stack a sensor module's raw semantic 3D frames into one array.

        Args:
            sensor_module_id: Which sensor module to read.

        Returns:
            A ``(num_frames, n_pixels, 4)`` float array.
        """
        return np.stack(
            [
                np.asarray(obs["semantic_3d"], dtype=float)
                for obs in self._raw_observations(sensor_module_id)
            ]
        )

    def pixel_locations(
        self, sensor_module_id: int | str, flat: bool = False
    ) -> np.ndarray:
        """Stack a sensor module's raw pixel location frames into one array.

        Args:
            sensor_module_id: Which sensor module to read.
            flat: Whether to return the pixel locations in a flat
                list-of-points with shape ``(num_frames, n_pixels, 3)`` or a
                reshaped ``(num_frames, H, W, 3)`` array when ``flat`` is
                False.

        Returns:
            A ``(num_frames, n_pixels, 3)`` float array when ``flat`` is
            True, or a ``(num_frames, H, W, 3)`` array when ``flat`` is
            False.
        """
        semantic_3d = self.semantic_3d(sensor_module_id)
        pixel_locations = semantic_3d[..., :3]  # Extract the x, y, z coordinates
        if flat:
            return pixel_locations
        sensor_shape = self.sensor_shape(sensor_module_id)
        return pixel_locations.reshape(pixel_locations.shape[0], *sensor_shape, 3)

    def salience_maps(
        self, sensor_module_id: int | str | None = None
    ) -> list[np.ndarray]:
        """Extract a sensor module's per-step salience maps.

        Args:
            sensor_module_id: Which sensor module to read. Defaults to the
                last one.

        Returns:
            One ``(H, W)`` float array per recorded step; empty if none were
            recorded.
        """
        sm_key = self._sm_key_or_default(sensor_module_id)
        maps = self[sm_key].get("salience_maps") or []
        return [np.asarray(m, dtype=float) for m in maps]

    def segmentation_maps(
        self, sensor_module_id: int | str | None = None
    ) -> np.ndarray | None:
        """Stack a sensor module's per-step segmentation masks.

        Args:
            sensor_module_id: Which sensor module to read. Defaults to the
                last one.

        Returns:
            A ``(num_frames, H, W)`` array, or None when none were recorded
            (or any step recorded no mask).
        """
        sm_key = self._sm_key_or_default(sensor_module_id)
        maps = self[sm_key].get("segmentation_maps") or []
        if maps and all(m is not None for m in maps):
            return np.stack([np.asarray(m) for m in maps])
        return None

    def sm_regions(
        self, sensor_module_id: int | str | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Extract a sensor module's per-step region proposals.

        Args:
            sensor_module_id: Which sensor module to read. Defaults to the
                last one.

        Returns:
            Per step, an ``(N, 3)`` location array and an ``(N,)`` weight
            array.
        """
        sm_key = self._sm_key_or_default(sensor_module_id)
        locations: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for region in self[sm_key].get("regions") or []:
            locations.append(
                np.asarray([aw["location"] for aw in region], dtype=float).reshape(
                    -1, 3
                )
            )
            weights.append(
                np.asarray([aw["weight"] for aw in region], dtype=float).ravel()
            )
        return locations, weights

    def goals(self, stage: str = "post") -> list[tuple[np.ndarray, np.ndarray]]:
        """Extract the attention system's per-step goals.

        Args:
            stage: "post" for the goals passed on to the motor system after
                filtering, "pre" for the goals collected before filtering.

        Returns:
            Per step, an ``(N, 3)`` location array and an ``(N,)`` confidence
            array. Goals without a location are dropped; a missing confidence
            is 0.

        Raises:
            ValueError: If stage is not "pre" or "post".
        """
        if stage not in ("pre", "post"):
            raise ValueError(f"stage must be 'pre' or 'post', got {stage!r}")

        steps = (self.get("attention_system") or {}).get(f"{stage}_filter_goals") or []
        extracted: list[tuple[np.ndarray, np.ndarray]] = []
        for goals in steps:
            located = [
                g
                for g in goals
                if isinstance(g, dict) and g.get("location") is not None
            ]
            extracted.append(
                (
                    np.asarray([g["location"] for g in located], dtype=float).reshape(
                        -1, 3
                    ),
                    np.asarray(
                        [g.get("confidence") or 0.0 for g in located], dtype=float
                    ),
                )
            )
        return extracted

    def fov_centres(self, sensor_module_id: int | str) -> list[np.ndarray | None]:
        """Extract the 3D world location at the centre of the sensor's FOV.

        Reads the central pixel of each step's DepthTo3DLocations output.

        Args:
            sensor_module_id: Which sensor module to read.

        Returns:
            One ``(3,)`` location per step, or None for steps without
            ``semantic_3d`` data.
        """
        centres: list[np.ndarray | None] = []
        for obs in self[self._sm_key(sensor_module_id)].get("raw_observations") or []:
            semantic_3d = obs.get("semantic_3d") if isinstance(obs, dict) else None
            if semantic_3d is None:
                centres.append(None)
                continue
            points = np.asarray(semantic_3d, dtype=float)
            height, width = np.asarray(obs["depth"]).shape[:2]
            centres.append(points[(height // 2) * width + width // 2, :3])
        return centres

    def voxel_grids(self) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
        """Extract the attention system's per-step voxel grids.

        Returns:
            The voxel size and -- per step -- a ``(V, 3)`` array of
            world-frame voxel centres and a ``(V,)`` weight array. A default
            fills in the size for runs from before the attention system
            exported it.
        """
        block = self.get("attention_system") or {}
        voxel_size = float(block.get("voxel_size", DEFAULT_VOXEL_SIZE))
        centres: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for grid in block.get("voxel_grids") or []:
            indices = np.asarray(grid["voxels"], dtype=int).reshape(-1, 3)
            centres.append((indices + VOXEL_CENTRE_OFFSET) * voxel_size)
            weights.append(np.asarray(grid["weight"], dtype=float).ravel())
        return voxel_size, centres, weights
