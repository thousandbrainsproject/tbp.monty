# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import shlex
import subprocess
import sys
from pathlib import Path

import hydra
import pytest

from tbp.monty.frameworks.actions.actions import TurnLeft, TurnRight
from tbp.monty.hydra import instantiate_experiment
from tests import HYDRA_ROOT

TUTORIAL = (
    Path(__file__).parents[2]
    / "docs/how-to-use-monty/tutorials/writing-a-custom-motor-policy.md"
)


@pytest.fixture
def tutorial(tmp_path: Path) -> str:
    text = TUTORIAL.read_text(encoding="utf-8")
    for filename in ("sweep_policy.py", "try_sweep.py"):
        code = text.split(f"```python\n# {filename}\n", 1)[1].split("\n```", 1)[0]
        (tmp_path / filename).write_text(f"{code}\n", encoding="utf-8")
    return text


def test_documented_sweep_sequence(tmp_path: Path, tutorial: str) -> None:
    expected = tutorial.split("```text\n", 1)[1].split("\n```", 1)[0]
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(tmp_path / "try_sweep.py")],
        capture_output=True,
        check=True,
        text=True,
    )

    assert result.stdout.splitlines() == expected.splitlines()


def test_documented_mujoco_experiment(
    tmp_path: Path, tutorial: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(tmp_path))
    command = tutorial.split("```sh\nuv run python run.py ", 1)[1].split("\n```", 1)[0]
    overrides = shlex.split(command.replace("\\\n", " "))

    with hydra.initialize_config_dir(version_base=None, config_dir=str(HYDRA_ROOT)):
        config = hydra.compose(config_name="experiment", overrides=overrides)

    config.experiment.config.logging.output_dir = str(tmp_path / "results")
    experiment = instantiate_experiment(config.experiment)
    with experiment:
        experiment.run()
        actions = [
            action
            for step_actions, _ in experiment.model.motor_system.action_sequence
            for action in step_actions
        ]

    first, second, third = actions
    assert isinstance(first, TurnLeft)
    assert isinstance(second, TurnRight)
    assert isinstance(third, TurnLeft)
    assert first.rotation_degrees == 5.0
    assert second.rotation_degrees == 10.0
    assert third.rotation_degrees == 5.0
