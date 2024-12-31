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

import pandas as pd
import wandb


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

    df = df.sort_values("cumtime", ascending=False)
    return df


class ProfileExperimentMixin:
    """Save cProfile traces for each episode.

    Example:
        class ProfiledExp(ProfileExperimentMixin, SomeExp):
            pass

        my_config["experiment_class"] = ProfiledExp

        NOTE: make sure this class is leftmost in mixin order.
    """

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
            profile_files = os.listdir(self.profile_dir)
            profile_paths = [
                os.path.join(self.profile_dir, file) for file in profile_files
            ]
            csv_files = [i for i in profile_paths if i.endswith(".csv")]

            for csv in csv_files:
                df = pd.read_csv(csv)
                basename = os.path.basename(csv)
                table = wandb.Table(dataframe=df)
                wandb.log({basename: table})

        super().close()
