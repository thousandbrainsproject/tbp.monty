"""The chart's x axis counts the run, not the phase.

Monty keeps `total_train_steps` and `total_eval_steps` separately, one per phase.
Counting the mode's own total makes the curve cumulative inside a phase and
restarts it at zero when the phase changes, which drew the evaluation episodes at
the left edge of the chart again — the "last episodes jump back to the beginning
of the graph" a reader sees.
"""

from __future__ import annotations

from dataclasses import dataclass

from tbp.monty.frameworks.experiments.mode import ExperimentMode

from contrib.liveview_experiment.src.monty_experiment_with_liveview import (
    MontyExperimentWithLiveView,
)


@dataclass
class _Counters:
    """The four numbers `_calculate_cumulative_step` reads, and nothing else."""

    experiment_mode: ExperimentMode = ExperimentMode.TRAIN
    total_train_steps: int = 0
    total_eval_steps: int = 0
    train_episodes: int = 0
    eval_episodes: int = 0


def cumulative(counters: _Counters, step: int) -> int:
    cumulative_step, _episode = MontyExperimentWithLiveView._calculate_cumulative_step(
        counters, step
    )
    return cumulative_step


class TestCumulativeStep:
    def test_the_first_training_episode_starts_at_zero(self):
        assert cumulative(_Counters(), 5) == 5

    def test_a_later_training_episode_continues_where_the_last_one_ended(self):
        counters = _Counters(total_train_steps=30, train_episodes=2)

        assert cumulative(counters, 2) == 32

    def test_evaluation_continues_after_training_instead_of_restarting(self):
        # The training phase did 90 steps; the first evaluation episode's step 4
        # is step 94 of the run, not step 4 of the graph.
        counters = _Counters(
            experiment_mode=ExperimentMode.EVAL, total_train_steps=90
        )

        assert cumulative(counters, 4) == 94

    def test_and_a_later_evaluation_episode_continues_after_the_one_before(self):
        counters = _Counters(
            experiment_mode=ExperimentMode.EVAL,
            total_train_steps=90,
            total_eval_steps=20,
            eval_episodes=2,
        )

        assert cumulative(counters, 3) == 113
