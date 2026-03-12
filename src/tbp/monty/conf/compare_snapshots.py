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
    "supervised_pre_training_base",
    "supervised_pre_training_5lms",
    "supervised_pre_training_5lms_all_objects",
    "supervised_pre_training_flat_objects_wo_logos",
    "supervised_pre_training_logos_after_flat_objects",
    "supervised_pre_training_curved_objects_after_flat_and_logo",
    "supervised_pre_training_objects_with_logos_lvl1_monolithic_models",
    "supervised_pre_training_objects_with_logos_lvl1_comp_models",
    "supervised_pre_training_objects_with_logos_lvl1_comp_models_burst_sampling",
    "supervised_pre_training_objects_with_logos_lvl2_comp_models",
    "supervised_pre_training_objects_with_logos_lvl3_comp_models",
    "supervised_pre_training_objects_with_logos_lvl4_comp_models",
    "world_image_from_stream_on_scanned_model",
    "tutorial/first_experiment",
    "tutorial/omniglot_training",
    "tutorial/omniglot_inference",
    "tutorial/surf_agent_2obj_train",
    "tutorial/surf_agent_2obj_eval",
    "tutorial/surf_agent_2obj_unsupervised",
    "tutorial/monty_meets_world_2dimage_inference",
    "test/base_config/base",
    "test/evidence_lm/base",
]

def compare_snapshots(
    run: str,
    config_name: str = "experiment",
    override_key: str = "experiment",
    snapshots_dir: Path = PROJECT_ROOT / "tests" / "conf" / "snapshots",
) -> bool:
    snapshot_path = snapshots_dir / f"{run}.yaml"
    print(f"Comparing with snapshot: {snapshot_path}")
    with snapshot_path.open("r") as f:
        snapshot: dict[str, Any] = yaml.safe_load(f)

    with hydra.initialize(version_base=None, config_path="."):
        config = hydra.compose(
            config_name=config_name,
            overrides=[f"{override_key}={run}"],
        )
        # to_object ensures the config is resolved
        config_yaml = OmegaConf.to_yaml(config)
        config_conf: dict[str, Any] = yaml.safe_load(config_yaml)
        first = compare(
            snapshot, config_conf, left_label="snapshot", right_label=config_name
        )
        second = compare(
            config_conf, snapshot, left_label=config_name, right_label="snapshot"
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


def compare_snapshot_dirs(
    left_dir: Path,
    right_dir: Path,
) -> bool:
    """Compare all YAML snapshots between two directories.

    Recursively finds all .yaml files in both directories, parses them,
    and compares them semantically (ignoring key order).

    Args:
        left_dir: The first snapshot directory.
        right_dir: The second snapshot directory.

    Returns:
        True if all snapshots match, False otherwise.
    """
    left_files = {f.relative_to(left_dir) for f in left_dir.rglob("*.yaml")}
    right_files = {f.relative_to(right_dir) for f in right_dir.rglob("*.yaml")}

    all_match = True

    for rel in sorted(left_files - right_files):
        print(f"MISSING in {right_dir}: {rel}")
        all_match = False

    for rel in sorted(right_files - left_files):
        print(f"MISSING in {left_dir}: {rel}")
        all_match = False

    for rel in sorted(left_files & right_files):
        left_data = yaml.safe_load((left_dir / rel).read_text())
        right_data = yaml.safe_load((right_dir / rel).read_text())
        result = compare(
            left_data, right_data,
            path=str(rel),
            left_label=str(left_dir),
            right_label=str(right_dir),
        ) and compare(
            right_data, left_data,
            path=str(rel),
            left_label=str(right_dir),
            right_label=str(left_dir),
        )
        if result:
            print(f"OK: {rel}")
        else:
            all_match = False

    return all_match


if __name__ == "__main__":
    setup_env()
    register_resolvers()
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", type=str)
    parser.add_argument(
        "--compare-dirs",
        nargs=2,
        metavar=("LEFT", "RIGHT"),
        help="Compare two snapshot directories semantically.",
    )
    args = parser.parse_args()

    if args.compare_dirs:
        left, right = Path(args.compare_dirs[0]), Path(args.compare_dirs[1])
        match = compare_snapshot_dirs(left, right)
        if match:
            print("\nAll snapshots match.")
        else:
            print("\nSome snapshots differ.")
            exit(1)
    elif args.experiment is None:
        for run in RUNS:
            compare_snapshots(run=run)
    else:
        compare_snapshots(run=args.experiment)
