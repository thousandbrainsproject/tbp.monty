# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import hydra
import yaml
from omegaconf import OmegaConf

from tbp.monty.frameworks.run_env import setup_env
from tbp.monty.hydra import register_resolvers

PROJECT_ROOT = Path(__file__).parents[4]

RUNS = [
    "base_config_10distinctobj_dist_agent",
    "base_config_10distinctobj_surf_agent",
    "randrot_noise_10distinctobj_dist_agent",
    "randrot_noise_10distinctobj_surf_agent",
    "randrot_noise_10distinctobj_dist_on_distm",
    "randrot_10distinctobj_surf_agent",
    "randrot_noise_10distinctobj_5lms_dist_agent",
    "base_10simobj_surf_agent",
    "randrot_noise_10simobj_dist_agent",
    "randomrot_rawnoise_10distinctobj_surf_agent",
    "randrot_noise_10simobj_surf_agent",
    "base_77obj_dist_agent",
    "base_10multi_distinctobj_dist_agent",
    "base_77obj_surf_agent",
    "randrot_noise_77obj_dist_agent",
    "randrot_noise_77obj_surf_agent",
    "randrot_noise_77obj_5lms_dist_agent",
    "surf_agent_unsupervised_10distinctobj",
    "surf_agent_unsupervised_10distinctobj_noise",
    "surf_agent_unsupervised_10simobj",
    "unsupervised_inference_distinctobj_dist_agent",
    "unsupervised_inference_distinctobj_surf_agent",
    "infer_comp_lvl1_with_comp_models_and_burst_sampling",
    "infer_comp_lvl1_with_monolithic_models",
    "infer_comp_lvl1_with_comp_models",
    "infer_comp_lvl2_with_comp_models",
    "infer_comp_lvl3_with_comp_models",
    "infer_comp_lvl4_with_comp_models",
    "randrot_noise_sim_on_scan_monty_world",
    "world_image_on_scanned_model",
    "bright_world_image_on_scanned_model",
    "dark_world_image_on_scanned_model",
    "hand_intrusion_world_image_on_scanned_model",
    "multi_object_world_image_on_scanned_model",
    "only_surf_agent_training_10obj",
    "only_surf_agent_training_10simobj",
    "only_surf_agent_training_allobj",
    "only_surf_agent_training_numenta_lab_obj",
]


def compare_snapshots(
    experiment: str,
    experiment_prefix: str = "",
    snapshots_dir: Path = PROJECT_ROOT / "tests" / "conf" / "snapshots",
) -> bool:
    snapshot_path = snapshots_dir / f"{experiment}.yaml"
    print(f"Comparing with snapshot: {snapshot_path}")
    with snapshot_path.open("r") as f:
        snapshot: dict[str, Any] = yaml.safe_load(f)

    with hydra.initialize(version_base=None, config_path="."):
        config = hydra.compose(
            config_name="experiment",
            overrides=[f"experiment={experiment_prefix}{experiment}"],
        )
        # to_object ensures the config is resolved
        experiment_yaml = OmegaConf.to_yaml(config)
        experiment_conf: dict[str, Any] = yaml.safe_load(experiment_yaml)
        first = compare(
            snapshot, experiment_conf, left_label="snapshot", right_label="experiment"
        )
        second = compare(
            experiment_conf, snapshot, left_label="experiment", right_label="snapshot"
        )

        return first and second


def compare(
    left: dict[str, Any] | list[Any] | Any,
    right: dict[str, Any] | list[Any] | Any,
    path: str = "",
    left_label: str = "snapshot",
    right_label: str = "experiment",
) -> bool:
    """Compare two configs hierarchically, ignoring key order at every level.

    Prints to stdout details of the first mismatch and exits immediately.

    Args:
        left: The left configuration to compare.
        right: The right configuration to compare.
        path: The path to the value being compared within the configuration.
        left_label: The label for the left config.
        right_label: The label for the right config.

    Returns:
        True if the configs match, False otherwise.
    """
    if type(left) is not type(right):
        print(
            f"{path} types do not match: {left_label}: {left} != {right_label}: {right}"
        )
        return False
    if isinstance(left, dict):
        if not isinstance(right, dict):
            print(
                f"{path} types do not match: "
                f"{left_label}: {left} != {right_label}: {right}"
            )
            return False
        for k in left:
            if k not in right:
                print(f"Key {path}.{k} not in {right_label}")
                return False
            if not compare(
                left[k],
                right[k],
                path=f"{path}.{k}",
                left_label=left_label,
                right_label=right_label,
            ):
                return False
        return True
    if isinstance(left, list):
        if len(left) != len(right):
            print(
                f"{path} lengths do not match: "
                f"{left_label}: {left} != {right_label}: {right}"
            )
            return False
        return all(
            compare(
                a,
                b,
                path=f"{path}.{i}",
                left_label=left_label,
                right_label=right_label,
            )
            for i, (a, b) in enumerate(zip(left, right))
        )
    if left != right:
        print(
            f"Values do not match {path}: "
            f"{left_label}: {left} != {right_label}: {right}"
        )
        return False
    return True


if __name__ == "__main__":
    setup_env()
    register_resolvers()
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", type=str)
    args = parser.parse_args()

    if args.experiment is None:
        for run in RUNS:
            compare_snapshots(
                experiment=run,
            )
    else:
        compare_snapshots(
            experiment=args.experiment,
        )
