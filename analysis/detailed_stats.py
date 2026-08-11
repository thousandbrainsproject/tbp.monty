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

import json
from pathlib import Path

import numpy as np


def load_episode_stats(exp_dir: Path, episode: int = 0) -> dict:
    """Load one episode's detailed stats from an experiment directory.

    Args:
        exp_dir: Experiment output directory (e.g. ``.../monty_runs/potato``).
        episode: Episode number to load.

    Returns:
        The episode's stats dict, keyed by module ("SM_1", "LM_0",
        "attention_system", "motor_system", "target").

    Raises:
        FileNotFoundError: If the directory holds no detailed stats.
    """
    exp_dir = Path(exp_dir)

    per_episode_dir = exp_dir / "detailed_run_stats"
    if per_episode_dir.is_dir():
        path = per_episode_dir / f"episode_{episode:06d}.json"
        data = json.loads(path.read_text())
        return data[str(episode)]

    single_file = exp_dir / "detailed_run_stats.json"
    if single_file.is_file():
        with single_file.open() as f:
            for line_number, line in enumerate(f):
                if line_number == episode:
                    return json.loads(line)[str(episode)]
        raise FileNotFoundError(f"Episode {episode} not found in {single_file}")

    raise FileNotFoundError(f"No detailed stats found under {exp_dir}")


def available_episodes(exp_dir: Path) -> list[int]:
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


def extract_rgba(stats: dict, sensor_module_id: int | str) -> np.ndarray:
    """Stack a sensor module's raw rgba frames into one array.

    Args:
        stats: Loaded episode stats.
        sensor_module_id: Which sensor module to read.

    Returns:
        A ``(num_frames, H, W, 4)`` uint8 array.
    """
    if isinstance(sensor_module_id, int):
        sensor_module_id = f"SM_{sensor_module_id}"
    raw_observations = stats[sensor_module_id]["raw_observations"]
    return np.stack([np.array(obs["rgba"]) for obs in raw_observations]).astype(
        np.uint8
    )
