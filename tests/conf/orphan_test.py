# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import pytest
import yaml

from tests import HYDRA_ROOT

SEEDS: set[Path] = {
    Path("experiment.yaml"),
    # Used by `detailed_logging_config_test.py`
    Path("logging/detailed_info_evidence_eval_runs.yaml"),
    # Used by `models/policy_test.py`
    Path("monty/motor_system_config/policy/test_surface_curvature_informed.yaml"),
}
"""Explicit paths to include as roots of the search, relative to the HYDRA_ROOT.

Can be used to explicitly include configurations known to be referenced outside
of Hydra, e.g. configs used only in tests."""


def test_no_orphaned_component_configs() -> None:
    """Verify that there are no orphaned component configurations.

    An orphaned component is a component config, i.e. not a top-level experiment
    configuration, that is not referenced by an experiment configuration, either
    directly or indirectly.
    """
    roots = find_all_roots()
    referenced_configs: set[Path] = find_all_referenced_configs(roots)
    all_configs: set[Path] = set(HYDRA_ROOT.glob("**/*.yaml"))

    orphaned_configs = all_configs - referenced_configs

    if orphaned_configs:
        orphaned_paths_str = yaml.dump(
            # This is a bit involved because we want the paths to be
            # relative to HYDRA_ROOT **and** `yaml.dump` doesn't work with
            # Path objects correctly, so we need to convert to a string.
            # Sort them so they're easier to remove.
            [str(Path(c).relative_to(HYDRA_ROOT)) for c in sorted(orphaned_configs)],
            default_flow_style=False,  # uses bullet lists instead of JSON style lists
        )

        pytest.fail(
            f"There are orphaned configs in `{HYDRA_ROOT}`.\n"
            "Remove the following orphaned configs:\n"
            f"{orphaned_paths_str}"
        )


def find_all_roots() -> list[Path]:
    """Find all roots for the liveness search.

    Takes the SEEDS and adds all discoverable top-level experiment configurations
    to form the list of initial "live" roots for the search.

    Returns:
        a list of absolute paths to roots.
    """
    roots = [(HYDRA_ROOT / seed) for seed in SEEDS]
    experiment_files = HYDRA_ROOT.glob("experiment/**/*.yaml")
    roots.extend(experiment_files)
    return roots


def find_all_referenced_configs(roots: Sequence[Path]) -> set[Path]:
    """Find all component configs referenced by the roots.

    This includes all directly referenced as well as indirectly referenced
    component configurations.

    Args:
        roots: a list of absolute paths to roots to start the search from.

    Returns:
        set of absolute paths to referenced configs including the initial roots.
    """
    referenced_configs = set()
    files_to_explore = list(roots)

    while files_to_explore:
        current_file = files_to_explore.pop()
        if current_file in referenced_configs:
            continue

        referenced_configs.add(current_file)

        with current_file.open("r") as f:
            config_yaml = yaml.safe_load(f)

        defaults_list = config_yaml.get("defaults", [])

        for entry in defaults_list:
            # The code in this section uses terms from the Hydra Defaults List grammar.
            # See the Hydra docs for details: https://hydra.cc/docs/advanced/defaults_list/#introduction
            config_path: Path | None = None

            # Due to how YAML interprets list entries, we either get a dict or a string.
            if isinstance(entry, dict):
                # Dictionary are all entries with a `:` in them.
                for config_group, option in entry.items():
                    if not option or option == "??":
                        break

                    # Remove the package if it's present, as we don't need it.
                    clean_group = config_group.split("@", 1)[0]

                    clean_group = re.sub(
                        r"^(optional|override)", "", clean_group
                    ).strip()

                    root = current_file.parent
                    if clean_group.startswith("/"):
                        clean_group = clean_group[1:]
                        root = HYDRA_ROOT

                    config_path = root / clean_group / f"{option}.yaml"
            elif isinstance(entry, str):
                if entry == "_self_":
                    continue

                path_part = entry.split("@", 1)[0].strip()
                config_group, config_name = path_part.rsplit("/", 1)

                root = current_file.parent
                if config_group.startswith("/"):
                    config_group = config_group[1:]
                    root = HYDRA_ROOT

                config_path = root / config_group / f"{config_name}.yaml"

            if config_path and config_path not in referenced_configs:
                assert config_path.exists(), (
                    f"`{config_path}` referenced by "
                    f"`{current_file.relative_to(HYDRA_ROOT)}` in "
                    f"`{yaml.dump(entry)}` does not exist!"
                )
                files_to_explore.append(config_path)
    return referenced_configs
