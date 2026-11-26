# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from pathlib import Path
from unittest.mock import patch

import hydra
import pytest
from omegaconf import open_dict

from tbp.monty.frameworks import run_parallel
from tests import HYDRA_ROOT


class ParentLoggingStarted(Exception):
    """Stop the run at parent logging initialization, before workers start."""


@pytest.mark.parametrize("group", ["benchmark_experiments", "ablation", None])
def test_parallel_eval_uses_configured_wandb_group(
    group: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in ("MONTY_DATA", "MONTY_LOGS", "MONTY_MODELS"):
        monkeypatch.setenv(key, str(tmp_path))
    with hydra.initialize_config_dir(version_base=None, config_dir=str(HYDRA_ROOT)):
        config = hydra.compose(
            config_name="experiment",
            overrides=[
                "experiment=randrot_noise_10distinctobj_dist_agent",
                "experiment.config.n_eval_epochs=1",
                'episodes="0"',
            ],
        )

    parent_config = config.experiment.config
    wrapper = parent_config.logging.monty_data_logger.handlers[-1]
    if group is None:
        with open_dict(wrapper):
            del wrapper["wandb_group"]
    else:
        wrapper.wandb_group = group

    with (
        patch.object(
            run_parallel.wandb, "init", side_effect=ParentLoggingStarted
        ) as init,
        pytest.raises(ParentLoggingStarted),
    ):
        run_parallel.main(config)

    init.assert_called_once()
    assert init.call_args is not None
    assert init.call_args.kwargs["group"] == group
    generated_config = init.call_args.kwargs["config"]["config"]
    assert generated_config["logging"]["log_parallel_wandb"] is True
    assert not run_parallel.has_wandb_wrapper(generated_config)
    assert run_parallel.has_wandb_wrapper(parent_config)
