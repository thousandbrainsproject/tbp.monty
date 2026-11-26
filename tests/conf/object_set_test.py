# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import hydra
import pytest
from omegaconf import DictConfig, OmegaConf

from tests import HYDRA_ROOT


@pytest.mark.parametrize(
    ("experiment", "mode", "object_set", "object_count", "added_object"),
    [
        ("supervised_pre_training_base", "train", "all_77", 77, "scissors"),
        ("base_config_10distinctobj_dist_agent", "eval", "similar_10", 10, "e_cups"),
    ],
)
def test_select_ycb_object_set(
    experiment: str, mode: str, object_set: str, object_count: int, added_object: str
) -> None:
    interface = f"experiment.config.{mode}_env_interface_args"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(HYDRA_ROOT)):
        defaults = hydra.compose(
            config_name="experiment", overrides=[f"experiment={experiment}"]
        )
        selected = hydra.compose(
            config_name="experiment",
            overrides=[
                f"experiment={experiment}",
                f"object_set@{interface}=ycb/{object_set}",
            ],
        )
        custom = hydra.compose(
            config_name="experiment",
            overrides=[f"experiment={experiment}", f"{interface}.object_names=[bowl]"],
        )

    args = OmegaConf.select(selected, interface)
    assert isinstance(args, DictConfig)
    assert len(args.object_names) == object_count
    assert added_object in args.object_names
    assert added_object not in OmegaConf.select(defaults, f"{interface}.object_names")
    assert OmegaConf.to_container(args.object_init_sampler) == OmegaConf.to_container(
        OmegaConf.select(defaults, f"{interface}.object_init_sampler")
    )
    assert OmegaConf.select(custom, f"{interface}.object_names") == ["bowl"]
