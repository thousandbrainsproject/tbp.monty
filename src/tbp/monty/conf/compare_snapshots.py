# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import argparse
from pathlib import Path

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
    "infer_comp_lvl1_with_monolithic_models",
    "infer_comp_lvl1_with_comp_models",
    "infer_comp_lvl3_with_comp_models",
    "infer_comp_lvl4_with_comp_models",
]


def compare_snapshots(
    experiment: str,
    experiment_prefix: str = "",
    snapshots_dir: Path = PROJECT_ROOT / "tests" / "conf" / "snapshots",
) -> bool:
    snapshot_path = snapshots_dir / f"{experiment}.yaml"
    print(f"Comparing with snapshot: {snapshot_path}")
    with snapshot_path.open("r") as f:
        snapshot = yaml.safe_load(f)

    with hydra.initialize(version_base=None, config_path="."):
        config = hydra.compose(
            config_name="experiment",
            overrides=[f"experiment={experiment_prefix}{experiment}"],
        )
        # to_object ensures the config is resolved
        experiment_conf = OmegaConf.to_yaml(config)
        experiment_conf = yaml.safe_load(experiment_conf)
        first = compare(snapshot, experiment_conf)
        second = compare(
            experiment_conf,
            snapshot,
            snapshot_label="experiment",
            experiment_label="snapshot",
        )

        return first and second


def compare(
    snapshot,
    experiment,
    path: str = "",
    snapshot_label="snapshot",
    experiment_label="experiment",
) -> bool:
    """Compare two configs hierarchically, ignoring key order at every level."""
    if type(snapshot) is not type(experiment):
        print(
            f"{path} types do not match: {snapshot_label}: {snapshot} != {experiment_label}: {experiment}"
        )
        return False
    if isinstance(snapshot, dict):
        for k in snapshot:
            if k not in experiment:
                print(f"Key {path}.{k} not in {experiment_label}")
                return False
            if not compare(
                snapshot[k],
                experiment[k],
                path=f"{path}.{k}",
                snapshot_label=snapshot_label,
                experiment_label=experiment_label,
            ):
                return False
        return True
    if isinstance(snapshot, list):
        if len(snapshot) != len(experiment):
            print(
                f"{path} lengths do not match: "
                f"{snapshot_label}: {snapshot} != {experiment_label}: {experiment}"
            )
            return False
        return all(
            compare(
                a,
                b,
                path=f"{path}.{i}",
                snapshot_label=snapshot_label,
                experiment_label=experiment_label,
            )
            for i, (a, b) in enumerate(zip(snapshot, experiment))
        )
    if snapshot != experiment:
        print(
            f"Values do not match {path}: "
            f"{snapshot_label}: {snapshot} != {experiment_label}: {experiment}"
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
