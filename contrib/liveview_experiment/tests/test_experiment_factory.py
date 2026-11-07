"""The LiveView target keeps the experiment the config named.

`run.sh` used to replace `experiment._target_` with `MontyExperimentWithLiveView`
for every base config, which replaced any subclass the config named as well — a
different experiment, quietly. These tests are about the class the factory builds
instead, and about turning the name a person types into that config.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Add project root and Monty's own `src` to the path for imports. The repo's
# tests run under a `uv` environment that carries both; a plain pytest run of this
# file has neither, and the module under test subclasses Monty's experiment.
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from tbp.monty.frameworks.experiments.monty_experiment import MontyExperiment

from contrib.liveview_experiment.src.experiment_factory import liveview_experiment_class
from contrib.liveview_experiment.src.monty_experiment_with_liveview import (
    MontyExperimentWithLiveView,
)

LIVEVIEW_DIR = Path(__file__).parent.parent


class _NamedExperiment(MontyExperiment):
    """A stand-in for the subclasses Monty ships: it has its own episode hooks."""

    def pre_episode(self) -> None:
        """What a subclass does before an episode, which must not be lost."""

    def own_method(self) -> str:
        return "the experiment's own"


class TestLiveViewExperimentClass:
    def test_the_experiment_the_config_named_is_still_the_experiment(self):
        mixed = liveview_experiment_class(_NamedExperiment)

        assert issubclass(mixed, MontyExperimentWithLiveView)
        assert issubclass(mixed, _NamedExperiment)

        # The LiveView hook is the one that runs, and it calls `super()`, so what
        # comes next in the chain is the experiment's own — the order is the
        # whole point, not an implementation detail.
        chain = mixed.mro()
        assert mixed.pre_episode is MontyExperimentWithLiveView.pre_episode
        assert chain.index(_NamedExperiment) > chain.index(MontyExperimentWithLiveView)

        # And a method only the experiment has is still the experiment's.
        assert mixed.own_method is _NamedExperiment.own_method

    def test_an_experiment_that_already_broadcasts_is_not_wrapped_twice(self):
        once = liveview_experiment_class(MontyExperimentWithLiveView)
        twice = liveview_experiment_class(once)

        assert once is MontyExperimentWithLiveView
        assert twice is MontyExperimentWithLiveView


class TestExperimentLookup:
    """`scripts/experiment_lookup.sh`: the name a person types, to a config."""

    def resolve(self, name: str) -> subprocess.CompletedProcess:
        script = (
            f'LIVEVIEW_CONF="{LIVEVIEW_DIR}/conf/experiment"; '
            f'MONTY_CONF="{LIVEVIEW_DIR.parent.parent}/src/tbp/monty/conf/experiment"; '
            f'source "{LIVEVIEW_DIR}/scripts/experiment_lookup.sh"; '
            f'resolve_experiment "{name}"'
        )
        return subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, check=False
        )

    def test_a_nested_experiment_is_found_by_its_own_name(self):
        found = self.resolve("evidence_mujoco")

        assert found.returncode == 0
        assert found.stdout.strip() == "main test/evidence_lm/evidence_mujoco"

    def test_a_whole_path_is_found_as_it_stands(self):
        found = self.resolve("test/evidence_lm/evidence_mujoco")

        assert found.returncode == 0
        assert found.stdout.strip() == "main test/evidence_lm/evidence_mujoco"

    def test_a_contrib_experiment_is_found_in_the_contrib_tree(self):
        found = self.resolve("randrot_10distinctobj_surf_agent_with_liveview")

        assert found.returncode == 0
        assert found.stdout.startswith("contrib ")

    def test_a_name_nothing_has_is_refused(self):
        found = self.resolve("no_such_experiment")

        assert found.returncode == 2
        assert found.stdout.strip() == ""
