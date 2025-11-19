# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import cProfile
import os
from pathlib import Path

import pandas as pd
import wandb

from tbp.monty.frameworks.experiments import MontyExperiment


def make_stats_df(stats):
    """Convert cProfile.Profile() stats to dataframe.

    Take a cProfile.Profile() object, gather stats, put in dataframe, sort by
    cumtime.

    Returns:
        The dataframe with the stats.
    """
    df = pd.DataFrame(
        stats.getstats(),
        columns=["func", "ncalls", "ccalls", "tottime", "cumtime", "callers"],
    )

    return df.sort_values("cumtime", ascending=False)


class ProfileExperimentMixin:
    """Save cProfile traces for each episode.

    Example:
        class ProfiledExp(ProfileExperimentMixin, SomeExp):
            pass

        my_config["experiment_class"] = ProfiledExp

        NOTE: make sure this class is leftmost in mixin order.
    """

    def __init_subclass__(cls, **kwargs):
        """Ensure that the mixin is used in the correct way.

        We want to ensure that the mixin is always the leftmost class listed in
        the base classes when used so that the methods defined here override the ones
        defined in MontyExperiment or its subclasses. We also want to ensure that
        any subclasses are actually extending MontyExperiment. This ensures that by
        raising an error if it is not.

        Raises:
            TypeError: when the mixin isn't the leftmost base class of the subclass
            being initialized or the base classes don't include a subclass of
            MontyExperiment.
        """
        super().__init_subclass__(**kwargs)
        if cls.__bases__[0] is not ProfileExperimentMixin:
            raise TypeError("ProfileExperimentMixin must be leftmost base class.")
        if not any(issubclass(b, MontyExperiment) for b in cls.__bases__):
            raise TypeError(
                "ProfileExperimentMixin must be mixed in with a subclass "
                "of MontyExperiment."
            )

    def make_profile_dir(self):
        self.profile_dir = os.path.join(self.output_dir, "profile")
        os.makedirs(self.profile_dir, exist_ok=True)

    def setup_experiment(self, config):
        filename = "profile-setup_experiment.csv"
        pr = cProfile.Profile()
        pr.enable()
        super().setup_experiment(config)
        pr.disable()

        self.make_profile_dir()
        df = make_stats_df(pr)
        filepath = os.path.join(self.profile_dir, filename)
        df.to_csv(filepath)

    def run_episode(self):
        mode, epoch, episode = self.get_epoch_state()
        filename = f"profile-{mode}_epoch_{epoch}_episode_{episode}.csv"
        pr = cProfile.Profile()
        pr.enable()
        super().run_episode()
        pr.disable()
        df = make_stats_df(pr)
        filepath = os.path.join(self.profile_dir, filename)
        df.to_csv(filepath)

    def train(self):
        filename = "profile-train.csv"
        pr = cProfile.Profile()
        pr.enable()
        super().train()
        pr.disable()
        df = make_stats_df(pr)
        filepath = os.path.join(self.profile_dir, filename)
        df.to_csv(filepath)

    def evaluate(self):
        filename = "profile-evaluate.csv"
        pr = cProfile.Profile()
        pr.enable()
        super().evaluate()
        pr.disable()
        df = make_stats_df(pr)
        filepath = os.path.join(self.profile_dir, filename)
        df.to_csv(filepath)

    def close(self):
        # If wandb is in use, send tables to wandb
        if len(self.wandb_handlers) > 0:
            profile_path = Path(self.profile_dir)
            for csv in profile_path.glob("*.csv"):
                df = pd.read_csv(csv)
                basename = csv.name
                table = wandb.Table(dataframe=df)
                wandb.log({basename: table})

        super().close()
