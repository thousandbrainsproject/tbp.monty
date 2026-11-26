# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json
from pathlib import Path

import pytest

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.loggers.monty_handlers import DetailedJSONHandler
from tbp.monty.frameworks.utils.logging_utils import (
    deserialize_json_chunks,
    load_stats,
)


@pytest.fixture
def combined_log(tmp_path: Path) -> Path:
    file = tmp_path / "detailed_run_stats.json"
    # Independent parallel workers can all write an episode under the same key.
    records = [{"0": {"object_name": name}} for name in ("cube", "capsule", "sphere")]
    file.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )
    return file


@pytest.mark.parametrize("use_string_path", [True, False])
def test_load_stats_accepts_the_documented_run_path(
    use_string_path: bool, combined_log: Path
) -> None:
    directory = combined_log.parent
    (directory / "train_stats.csv").write_text("object_name\ncube\n", encoding="utf-8")
    (directory / "eval_stats.csv").write_text("object_name\nsphere\n", encoding="utf-8")
    path = str(directory) if use_string_path else directory

    train, evaluation, detailed, models = load_stats(path, load_models=False)

    assert train["object_name"].tolist() == ["cube"]
    assert evaluation["object_name"].tolist() == ["sphere"]
    assert detailed == {
        "0": {"object_name": "cube"},
        "1": {"object_name": "capsule"},
        "2": {"object_name": "sphere"},
    }
    assert models is None


def test_chunk_range_has_exclusive_stop(combined_log: Path) -> None:
    assert deserialize_json_chunks(combined_log, start=1, stop=2) == {
        "1": {"object_name": "capsule"}
    }
    assert deserialize_json_chunks(combined_log, stop=0) == {}


def test_chunk_selection_accepts_one_shot_iterables(
    combined_log: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    detailed = deserialize_json_chunks(combined_log, episodes=iter([2, 1, 2]))
    assert detailed == {
        "1": {"object_name": "capsule"},
        "2": {"object_name": "sphere"},
    }
    assert list(detailed) == ["1", "2"]
    assert capsys.readouterr().out == ""


def test_chunk_selection_reports_missing_episodes(
    combined_log: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert deserialize_json_chunks(combined_log, episodes=[0, 9]) == {
        "0": {"object_name": "cube"}
    }
    assert "missing episodes: 9" in capsys.readouterr().out


@pytest.fixture
def episode_log_dir(tmp_path: Path) -> Path:
    handler = DetailedJSONHandler(detailed_save_per_episode=True)
    for episode, name in [(12, "sphere"), (3, "cube"), (3, "capsule")]:
        handler.report_episode(
            {
                "BASIC": {"eval_stats": {0: {"object_name": name}}},
                "DETAILED": {0: {"steps": episode + 1}},
            },
            str(tmp_path),
            0,
            ExperimentMode.EVAL,
            eval_episodes_to_total={0: episode},
        )
    return tmp_path / "detailed_run_stats"


def test_load_stats_reads_sparse_per_episode_logs(episode_log_dir: Path) -> None:
    assert (episode_log_dir / "episode_000003_old.json").is_file()
    results = load_stats(
        episode_log_dir.parent, load_train=False, load_eval=False, load_models=False
    )

    assert results == (
        None,
        None,
        {
            "3": {"object_name": "capsule", "steps": 4},
            "12": {"object_name": "sphere", "steps": 13},
        },
        None,
    )
    assert list(results[2]) == ["3", "12"]
    assert deserialize_json_chunks(episode_log_dir, start=3, stop=12) == {
        "3": {"object_name": "capsule", "steps": 4}
    }
    assert deserialize_json_chunks(episode_log_dir, episodes=iter([12])) == {
        "12": {"object_name": "sphere", "steps": 13}
    }


def test_load_stats_retains_legacy_json_mapping(episode_log_dir: Path) -> None:
    # A combined file at the original path takes precedence when both forms exist.
    combined_file = episode_log_dir.parent / "detailed_run_stats.json"
    old_stats = {"7": {"object_name": "mug"}, "8": {"object_name": "bowl"}}
    combined_file.write_text(json.dumps(old_stats, indent=2), encoding="utf-8")

    *_, detailed, models = load_stats(
        episode_log_dir.parent, load_train=False, load_eval=False, load_models=False
    )
    assert detailed == old_stats
    assert models is None
