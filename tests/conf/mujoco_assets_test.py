# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from pathlib import Path

import hydra
import pytest
from omegaconf import OmegaConf

from tests import HYDRA_ROOT


@pytest.mark.parametrize(
    ("experiment", "catalog", "relative_path"),
    [
        ("tutorial/first_experiment", "builtins", None),
        (
            "base_infer_cows_large_compositional_random_locs",
            "cows",
            "compositional_objects_1.4_mujoco",
        ),
    ],
)
def test_select_mujoco_assets_without_changing_agent(
    experiment: str,
    catalog: str,
    relative_path: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MONTY_DATA", str(tmp_path))
    package = "experiment.config.environment.env_init_args"
    base_overrides = [f"experiment={experiment}"]
    catalog_overrides = [
        *base_overrides,
        f"environment/mujoco_assets@{package}={catalog}",
    ]
    with hydra.initialize_config_dir(version_base=None, config_dir=str(HYDRA_ROOT)):
        default = hydra.compose(config_name="experiment", overrides=base_overrides)
        selected = hydra.compose(config_name="experiment", overrides=catalog_overrides)
        custom = hydra.compose(
            config_name="experiment",
            overrides=[*catalog_overrides, f"{package}.data_path=custom-mesh-root"],
        )

    expected = str(tmp_path / relative_path) if relative_path is not None else None
    assert OmegaConf.select(selected, f"{package}.data_path") == expected
    assert OmegaConf.select(default, f"{package}.data_path") != expected
    assert OmegaConf.select(custom, f"{package}.data_path") == "custom-mesh-root"

    default_args = OmegaConf.to_container(OmegaConf.select(default, package))
    selected_args = OmegaConf.to_container(OmegaConf.select(selected, package))
    assert isinstance(default_args, dict)
    assert isinstance(selected_args, dict)
    del default_args["data_path"]
    del selected_args["data_path"]
    assert selected_args == default_args
