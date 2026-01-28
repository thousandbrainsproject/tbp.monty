# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import hydra
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import wandb
from omegaconf import DictConfig, OmegaConf

from tbp.monty.frameworks.environments.embodied_data import (
    EnvironmentInterfacePerObject,
)
from tbp.monty.frameworks.environments.object_init_samplers import (
    Predefined,
)
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.experiments.pretraining_experiments import (
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.frameworks.experiments.profile import ProfileExperimentMixin
from tbp.monty.frameworks.loggers.monty_handlers import (
    BasicCSVStatsHandler,
    DetailedJSONHandler,
)
from tbp.monty.frameworks.utils.logging_utils import (
    maybe_rename_existing_dir,
    maybe_rename_existing_file,
)
from tbp.monty.hydra import register_resolvers

__all__ = ["main"]

logger = logging.getLogger(__name__)

RE_OPEN_LEFT = re.compile(r"^:(\d+)$")  # ":N"
RE_OPEN_RIGHT = re.compile(r"^(\d+):$")  # "N:"
RE_CLOSED = re.compile(r"^(\d+)\s*:\s*(\d+)$")  # "A:B"
RE_SINGLE = re.compile(r"^\d+$")  # "N"


def mv_files(filenames: Iterable[Path], outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    for src in filenames:
        dest = outdir / src.name

        if dest.exists():
            dest.unlink()

        src.replace(dest)


def cat_files(filenames: Iterable[Path], outfile: Path):
    if outfile.exists():
        print(f"Removing existing file before writing new one: {outfile}")
        outfile.unlink()

    with outfile.open("wb") as out_f:
        for file in filenames:
            with file.open("rb") as in_f:
                shutil.copyfileobj(in_f, out_f)


def cat_csv(filenames: Iterable[Path], outfile: Path):
    dfs = [pd.read_csv(file) for file in filenames]
    df = pd.concat(dfs)
    df.to_csv(outfile, index=False)


def sample_params_to_init_args(params):
    new_params = {}
    new_params["positions"] = [params["position"]]
    new_params["scales"] = [params["scale"]]
    new_params["rotations"] = [params["euler_rotation"]]

    return new_params


def post_parallel_log_cleanup(filenames: Iterable[Path], outfile: Path, cat_fn):
    existing_files = [f for f in filenames if f.exists()]
    if len(existing_files) == 0:
        return

    # Concatenate files together
    cat_fn(existing_files, outfile)

    # Remove json files
    for f in existing_files:
        f.unlink(missing_ok=True)


def post_parallel_profile_cleanup(parallel_dirs: Iterable[Path], base_dir: Path, mode):
    profile_dirs = [pdir / "profile" for pdir in parallel_dirs]

    episode_csvs = []
    setup_csvs = []
    overall_csvs = []

    for profile_dir in profile_dirs:
        epsd_csv_paths = list(profile_dir.glob("*episode*.csv"))
        setup_csv = profile_dir / "profile-setup_experiment.csv"
        overall_csv = profile_dir / f"profile-{mode}.csv"

        episode_csvs.extend(epsd_csv_paths)
        setup_csvs.append(setup_csv)
        overall_csvs.append(overall_csv)

    episode_outfile = base_dir / f"profile-{mode}_episodes.csv"
    setup_outfile = base_dir / "profile-setup_experiment.csv"
    overall_outfile = base_dir / f"profile-{mode}.csv"

    post_parallel_log_cleanup(episode_csvs, episode_outfile, cat_fn=cat_csv)
    post_parallel_log_cleanup(setup_csvs, setup_outfile, cat_fn=cat_csv)
    post_parallel_log_cleanup(overall_csvs, overall_outfile, cat_fn=cat_csv)


def print_config(config: DictConfig) -> None:
    """Print config with nice formatting."""
    print("\n\n")
    print("Printing config below")
    print("-" * 100)
    print(OmegaConf.to_yaml(config))
    print("-" * 100)


def parse_episode_spec(episode_spec: str | None, total: int) -> list[int]:
    """Parses a zero-based episode selection string into episode indices.

    Converts a human-friendly selection string into a sorted list of unique,
    zero-based episode indices in the half-open interval `[0, total)`.
    The parser supports single indices and Python-slice-like ranges using
    a colon (`:`), with the end index exclusive.

    Args:
        episode_spec: Selection string describing which episodes to run.
            See supported forms.
        total: Total number of episodes. Must be non-negative.

    Supported forms:
      - `"all"`, `":"`, or empty string: select all valid indices `[0, total)`
      - Comma-separated integers and ranges, for example `"0,3,5:8"`
      - Open-ended ranges (end-exclusive):
          - `":N"` selects `[0, N)` (i.e., indices `0` through `N-1`)
          - `"N:"` selects `[N, total)`

    Notes:
      - Ranges are validated, not clamped. If a range falls outside `[0, total)`,
        or is otherwise malformed, a `ValueError` is raised.
      - Duplicates are eliminated; the result is returned in ascending order.

    Returns:
        A sorted list of unique zero-based indices within `[0, total)` that match
        the selection described by `episode_spec`.

    Raises:
        ValueError: If the selection contains any invalid index or range.
    """
    if episode_spec is None:
        return list(range(total))
    s = episode_spec.strip().lower()
    if s in ("", "all", ":"):
        return list(range(total))

    selected: set[int] = set()

    for raw in s.split(","):
        part = raw.strip()
        if not part:
            continue

        m = RE_OPEN_LEFT.match(part)
        if m:
            idx_end = int(m.group(1))
            if 0 < idx_end <= total:
                selected.update(range(idx_end))
                continue

            raise ValueError(f"{m.group(0)} is not a valid range.")

        m = RE_OPEN_RIGHT.match(part)
        if m:
            idx_start = int(m.group(1))
            if 0 <= idx_start < total:
                selected.update(range(idx_start, total))
                continue

            raise ValueError(f"{m.group(0)} is not a valid range.")

        m = RE_CLOSED.match(part)
        if m:
            idx_start = int(m.group(1))
            idx_end = int(m.group(2))
            if 0 <= idx_start < idx_end and idx_start < idx_end <= total:
                selected.update(range(idx_start, idx_end))
                continue

            raise ValueError(f"{m.group(0)} is not a valid range.")

        if RE_SINGLE.match(part):
            idx = int(part)
            if 0 <= idx < total:
                selected.add(idx)
                continue

            raise ValueError(f"{part} is not a valid index.")

        raise ValueError(f"{part} is not a valid selection.")

    return sorted(selected)


def filter_episode_configs(configs: list[dict], episode_spec: str | None) -> list[dict]:
    idxs = parse_episode_spec(episode_spec, len(configs))
    return [cfg for i, cfg in enumerate(configs) if i in idxs]


def generate_parallel_eval_configs(
    experiment: DictConfig,
    name: str,
) -> list[Mapping]:
    """Generate configs for evaluation episodes in parallel.

    Create a config for each object and rotation in the experiment. Unlike with parallel
    training episodes, a config is created for each object + rotation separately.

    Notice that no matter what object_init_sampler is specified in the configuration,
    it is replaced by a Predefined object initializer with parameters that the sampler
    would generate for that specific episode and epoch.

    Args:
        experiment: Config for experiment to be broken into parallel configs.
        name: Name of experiment.

    Returns:
        List of configs for evaluation episodes.
    """
    sampler = hydra.utils.instantiate(
        experiment.config["eval_env_interface_args"]["object_init_sampler"]
    )
    object_names = experiment.config["eval_env_interface_args"]["object_names"]
    # sampler_params = sampler.all_combinations_of_params()

    new_experiments = []
    epoch_count = 0
    episode_count = 0
    n_epochs = experiment.config["n_eval_epochs"]
    seed = experiment.config["seed"]

    params = sample_params_to_init_args(
        sampler(seed, ExperimentMode.EVAL, epoch_count, episode_count)
    )

    # Try to mimic the exact workflow instead of guessing
    while epoch_count < n_epochs:
        for obj in object_names:
            new_experiment: Mapping = OmegaConf.to_object(experiment)  # type: ignore[assignment]

            # No training
            new_experiment["config"].update(
                do_eval=True,
                do_train=False,
                episode=episode_count,
                n_eval_epochs=1,
            )

            # Save results in parallel subdir of output_dir, update run_name
            output_dir = Path(new_experiment["config"]["logging"]["output_dir"])
            run_name = f"{name}-parallel_eval_episode_{episode_count}"
            new_experiment["config"]["logging"]["run_name"] = run_name
            new_experiment["config"]["logging"]["output_dir"] = (
                output_dir / name / run_name
            )
            if len(new_experiment["config"]["logging"]["wandb_handlers"]) > 0:
                new_experiment["config"]["logging"]["wandb_handlers"] = []
                new_experiment["config"]["logging"]["log_parallel_wandb"] = True
                new_experiment["config"]["logging"]["experiment_name"] = name
            else:
                new_experiment["config"]["logging"]["log_parallel_wandb"] = False

            new_experiment["config"]["eval_env_interface_args"].update(
                object_names=[obj],
                object_init_sampler=Predefined(**params),
            )

            new_experiments.append(new_experiment)
            episode_count += 1
            params = sample_params_to_init_args(
                sampler(seed, ExperimentMode.EVAL, epoch_count, episode_count)
            )

        epoch_count += 1
        params = sample_params_to_init_args(
            sampler(seed, ExperimentMode.EVAL, epoch_count, episode_count)
        )

    return new_experiments


def generate_parallel_train_configs(experiment: DictConfig, name: str) -> list[Mapping]:
    """Generate configs for training episodes in parallel.

    Create a config for each object in the experiment. Unlike with parallel eval
    episodes, each parallel config specifies a single object but all rotations.

    Args:
        experiment: Config for experiment to be broken into parallel configs.
        name: Name of experiment.

    Returns:
        List of configs for training episodes.

    Note:
        If we view the same object from multiple poses in separate experiments, we
        need to replicate what post_episode does in supervised pre-training. To avoid
        this, we just run training episodes in parallel across objects, but poses are
        still in sequence. By contrast, eval episodes are parallel across objects
        AND poses.

    """
    sampler = hydra.utils.instantiate(
        experiment.config["train_env_interface_args"]["object_init_sampler"]
    )
    object_names = experiment.config["train_env_interface_args"]["object_names"]
    new_experiments = []

    for obj in object_names:
        new_experiment: Mapping = OmegaConf.to_object(experiment)  # type: ignore[assignment]

        # No eval
        new_experiment["config"].update(do_eval=False, do_train=True, n_train_epochs=1)

        # Save results in parallel subdir of output_dir, update run_name
        output_dir = Path(new_experiment["config"]["logging"]["output_dir"])
        run_name = f"{name}-parallel_train_episode_{obj}"
        new_experiment["config"]["logging"]["run_name"] = run_name
        new_experiment["config"]["logging"]["output_dir"] = output_dir / name / run_name
        new_experiment["config"]["logging"]["wandb_handlers"] = []
        new_experiment["config"]["logging"]["log_parallel_wandb"] = False

        # Object id, pose parameters for single episode
        new_experiment["config"]["train_env_interface_args"].update(
            object_names=[obj for _ in range(len(sampler))]
        )
        new_experiment["config"]["train_env_interface_args"]["object_init_sampler"][
            "change_every_episode"
        ] = True

        new_experiments.append(new_experiment)

    return new_experiments


def single_train(experiment):
    output_dir = Path(experiment["config"]["logging"]["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)
    exp = hydra.utils.instantiate(experiment)
    with exp:
        print("---------training---------")
        exp.run()


def single_evaluate(experiment):
    output_dir = Path(experiment["config"]["logging"]["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)
    exp = hydra.utils.instantiate(experiment)
    with exp:
        print("---------evaluating---------")
        exp.run()
        if experiment["config"]["logging"]["log_parallel_wandb"]:
            # WARNING: This relies on logger in the experiment having
            # `self.use_parallel_wandb_logging` set to True
            # This way, the logger does not flush its buffer in the
            # `exp.run()` call above.
            return get_episode_stats(exp, "eval", exp.config.episode)


def get_episode_stats(exp, mode, episode: int = 0):
    eval_stats = exp.monty_logger.get_formatted_overall_stats(mode, episode)
    exp.monty_logger.flush()
    # Remove overall stats field since they are only averaged over 1 episode
    # and might cause confusion.
    for key in list(eval_stats.keys()):
        if key.startswith("overall"):
            del eval_stats[key]
    return eval_stats


def get_overall_stats(stats):
    overall_stats = {}
    # combines correct and correct_mlh
    overall_stats["overall/percent_correct"] = np.mean(stats["episode/correct"]) * 100
    overall_stats["overall/percent_confused"] = np.mean(stats["episode/confused"]) * 100
    # Only how many episodes were correct after time out
    overall_stats["overall/percent_correct_mlh"] = (
        np.mean(stats["episode/correct_mlh"]) * 100
    )
    overall_stats["overall/percent_confused_mlh"] = (
        np.mean(stats["episode/confused_mlh"]) * 100
    )
    overall_stats["overall/percent_no_match"] = np.mean(stats["episode/no_match"]) * 100
    overall_stats["overall/percent_pose_time_out"] = (
        np.mean(stats["episode/pose_time_out"]) * 100
    )
    overall_stats["overall/percent_time_out"] = np.mean(stats["episode/time_out"]) * 100
    overall_stats["overall/percent_used_mlh_after_timeout"] = (
        np.mean(stats["episode/used_mlh_after_time_out"]) * 100
    )
    overall_stats["overall/percent_correct_child_or_parent"] = (
        np.mean(stats["episode/consistent_child_or_parent"]) * 100
    )
    overall_stats["overall/percent_consistent_child_obj"] = (
        np.mean(stats["episode/consistent_child_obj"]) * 100
    )
    overall_stats["overall/avg_prediction_error"] = np.mean(
        stats["episode/avg_prediction_error"]
    )

    correct_ids = np.where(np.array(stats["episode/correct"]) == 1)
    correct_rotation_errs = np.array(stats["episode/rotation_error"])[correct_ids]
    overall_stats["overall/avg_rotation_error"] = np.mean(correct_rotation_errs)
    overall_stats["overall/avg_num_lm_steps"] = np.mean(stats["episode/lm_steps"])
    overall_stats["overall/avg_num_monty_steps"] = np.mean(stats["episode/monty_steps"])
    overall_stats["overall/avg_num_monty_matching_steps"] = np.mean(
        stats["episode/monty_matching_steps"]
    )
    overall_stats["overall/run_time"] = np.sum(stats["episode/run_time"])
    overall_stats["overall/avg_episode_run_time"] = np.mean(stats["episode/run_time"])
    overall_stats["overall/num_episodes"] = len(stats["episode/correct"])
    overall_stats["overall/avg_goal_attempts"] = np.mean(
        stats["episode/goal_states_attempted"]
    )
    overall_stats["overall/avg_goal_success"] = np.mean(
        stats["episode/goal_state_success_rate"]
    )

    return overall_stats


def collect_detailed_episodes_names(parallel_dirs: Iterable[Path]) -> list[Path]:
    filenames = []
    for pdir in parallel_dirs:
        filenames.extend((pdir / "detailed_run_stats").glob("*.json"))
    return filenames


def post_parallel_eval(experiments: list[Mapping], base_dir: Path) -> None:
    """Post-execution cleanup after running evaluation in parallel.

    Logs are consolidated across parallel runs and saved to disk.

    Args:
        experiments: List of experiments ran in parallel.
        base_dir: Directory where parallel logs are stored.
    """
    print("Executing post parallel evaluation cleanup")
    parallel_dirs = [
        Path(exp["config"]["logging"]["output_dir"]) for exp in experiments
    ]

    logging_config = experiments[0]["config"]["logging"]
    save_per_episode = logging_config.get("detailed_save_per_episode")

    # Loop over types of loggers, figure out how to clean up each one
    for handler in logging_config["monty_handlers"]:
        if issubclass(handler, DetailedJSONHandler):
            if save_per_episode:
                filenames = collect_detailed_episodes_names(parallel_dirs)
                outdir = base_dir / "detailed_run_stats"
                maybe_rename_existing_dir(outdir)
                post_parallel_log_cleanup(filenames, outdir, cat_fn=mv_files)
            else:
                filename = "detailed_run_stats.json"
                filenames = [pdir / filename for pdir in parallel_dirs]
                outfile = base_dir / filename
                maybe_rename_existing_file(outfile)
                post_parallel_log_cleanup(filenames, outfile, cat_fn=cat_files)
            continue

        if issubclass(handler, BasicCSVStatsHandler):
            filename = "eval_stats.csv"
            filenames = [pdir / filename for pdir in parallel_dirs]
            outfile = base_dir / filename
            maybe_rename_existing_file(outfile)
            post_parallel_log_cleanup(filenames, outfile, cat_fn=cat_csv)
            continue

    if experiments[0]["config"]["logging"]["python_log_to_file"]:
        filename = "log.txt"
        filenames = [pdir / filename for pdir in parallel_dirs]
        outfile = base_dir / filename
        post_parallel_log_cleanup(filenames, outfile, cat_fn=cat_files)

    exp = hydra.utils.instantiate(experiments[0])
    if isinstance(exp, ProfileExperimentMixin):
        post_parallel_profile_cleanup(parallel_dirs, base_dir, "evaluate")

    for pdir in parallel_dirs:
        shutil.rmtree(pdir)


def post_parallel_train(experiments: list[Mapping], base_dir: Path) -> None:
    """Post-execution cleanup after running training in parallel.

    Object models are consolidated across parallel runs and saved to disk.

    Args:
        experiments: List of experiments ran in parallel.
        base_dir: Directory where parallel logs are stored.
    """
    print("Executing post parallel training cleanup")
    parallel_dirs = [
        Path(exp["config"]["logging"]["output_dir"]) for exp in experiments
    ]
    pretraining = False
    exp = hydra.utils.instantiate(experiments[0])
    if isinstance(exp, MontySupervisedObjectPretrainingExperiment):
        parallel_dirs = [pdir / "pretrained" for pdir in parallel_dirs]
        pretraining = True

    # Loop over types of loggers, figure out how to clean up each one
    logging_config = experiments[0]["config"]["logging"]
    save_per_episode = logging_config.get("detailed_save_per_episode")

    for handler in logging_config["monty_handlers"]:
        if issubclass(handler, DetailedJSONHandler):
            if save_per_episode:
                filenames = collect_detailed_episodes_names(parallel_dirs)
                outdir = base_dir / "detailed_run_stats"
                maybe_rename_existing_dir(outdir)
                post_parallel_log_cleanup(filenames, outdir, cat_fn=mv_files)
            else:
                filename = "detailed_run_stats.json"
                filenames = [pdir / filename for pdir in parallel_dirs]
                outfile = base_dir / filename
                maybe_rename_existing_file(outfile)
                post_parallel_log_cleanup(filenames, outfile, cat_fn=cat_files)
            continue

        if issubclass(handler, BasicCSVStatsHandler):
            filename = "train_stats.csv"
            filenames = [pdir / filename for pdir in parallel_dirs]
            outfile = base_dir / filename
            maybe_rename_existing_file(outfile)
            post_parallel_log_cleanup(filenames, outfile, cat_fn=cat_csv)
            continue

    if experiments[0]["config"]["logging"]["python_log_to_file"]:
        filename = "log.txt"
        filenames = [pdir / filename for pdir in parallel_dirs]
        outfile = base_dir / filename
        post_parallel_log_cleanup(filenames, outfile, cat_fn=cat_files)

    if isinstance(exp, ProfileExperimentMixin):
        post_parallel_profile_cleanup(parallel_dirs, base_dir, "train")

    with exp:
        exp.model.load_state_dict_from_parallel(parallel_dirs, save=True)
        output_dir = Path(experiments[0]["config"]["logging"]["output_dir"]).parent
        if isinstance(exp, MontySupervisedObjectPretrainingExperiment):
            output_dir = output_dir / "pretrained"
        output_dir.mkdir(exist_ok=True, parents=True)
        saved_model_file = output_dir / "model.pt"
        torch.save(exp.model.state_dict(), saved_model_file)

    if pretraining:
        pdirs = [i.parent for i in parallel_dirs]
    else:
        pdirs = parallel_dirs

    for pdir in pdirs:
        print(f"Removing directory: {pdir}")
        shutil.rmtree(pdir)


def run_episodes_parallel(
    experiments: list[Mapping],
    num_parallel: int,
    experiment_name: str,
    train: bool = True,
) -> None:
    """Run episodes in parallel.

    Args:
        experiments: List of configs to run in parallel.
        num_parallel: Maximum number of parallel processes to run. If there
            are fewer configs to run than `num_parallel`, then the actual number of
            processes will be equal to the number of configs.
        experiment_name: Name of the experiment.
        train: Whether the episodes are training or evaluation episodes.
    """
    # Use fewer processes if there are fewer configs than `num_parallel`.
    num_parallel = min(len(experiments), num_parallel)
    exp_type = "training" if train else "evaluation"
    print(
        f"-------- Running {exp_type} experiment {experiment_name}"
        f" with {num_parallel} episodes in parallel --------"
    )
    start_time = time.time()
    log_parallel_wandb = experiments[0]["config"]["logging"]["log_parallel_wandb"]
    if log_parallel_wandb:
        run = wandb.init(
            name=experiment_name,
            group=experiments[0]["config"]["logging"]["wandb_group"],
            project="Monty",
            config=experiments[0],
            id=hydra.utils.instantiate(experiments[0]["config"]["logging"]["wandb_id"]),
        )
    print(f"Wandb setup took {time.time() - start_time} seconds")
    start_time = time.time()
    with mp.Pool(num_parallel, maxtasksperchild=1) as p:
        if train:
            # NOTE: since we don't use wandb logging for training right now
            # it is also not covered here. Might want to add that in the future.
            p.map(single_train, experiments)
        elif log_parallel_wandb:
            all_episode_stats: dict[str, list[Any]] = {}
            for result in p.imap(single_evaluate, experiments):
                run.log(result)
                if not all_episode_stats:  # first episode
                    for key in list(result.keys()):
                        all_episode_stats[key] = [result[key]]
                else:
                    for key in list(result.keys()):
                        all_episode_stats[key].append(result[key])
            overall_stats = get_overall_stats(all_episode_stats)
            # episode/run_time is the sum over individual episode run times.
            # when running parallel this may not be the actual run time so we
            # log this here additionally.
            overall_stats["overall/parallel_run_time"] = time.time() - start_time
            overall_stats["overall/num_processes"] = num_parallel
            run.log(overall_stats)
        else:
            p.map(single_evaluate, experiments)
    end_time = time.time()
    total_time = end_time - start_time

    base_dir = Path(experiments[0]["config"]["logging"]["output_dir"]).parent

    if train:
        post_parallel_train(experiments, base_dir)
        if log_parallel_wandb:
            csv_path = base_dir / "train_stats.csv"
            if csv_path.exists():
                train_stats = pd.read_csv(csv_path)
                train_table = wandb.Table(dataframe=train_stats)
                if run is not None:
                    run.log({"train_stats": train_table})
            else:
                print(f"No csv table found at {csv_path} to log to wandb")
    else:
        post_parallel_eval(experiments, base_dir)
        if log_parallel_wandb:
            csv_path = base_dir / "eval_stats.csv"
            if csv_path.exists():
                eval_stats = pd.read_csv(csv_path)
                eval_table = wandb.Table(dataframe=eval_stats)
                run.log({"eval_stats": eval_table})
            else:
                print(f"No csv table found at {csv_path} to log to wandb")

    print(
        f"Total time for {len(experiments)} running {num_parallel} episodes in "
        f"parallel: {total_time}"
    )
    if log_parallel_wandb:
        run.finish()

    print(f"Done running parallel experiments in {end_time - start_time} seconds")

    # Keep a record of how long everything takes
    with (base_dir / "parallel_log.txt").open("w") as f:
        f.write(f"experiment: {experiment_name}\n")
        f.write(f"num_parallel: {num_parallel}\n")
        f.write(f"total_time: {total_time}")


@hydra.main(config_path="../../../conf", config_name="experiment", version_base=None)
def main(cfg: DictConfig):
    if cfg.quiet_habitat_logs:
        os.environ["MAGNUM_LOG"] = "quiet"
        os.environ["HABITAT_SIM_LOG"] = "quiet"

    print_config(cfg)
    register_resolvers()

    if cfg.experiment.config.do_train:
        assert issubclass(
            cfg.experiment.config.train_env_interface_class,
            EnvironmentInterfacePerObject,
        ), "parallel experiments only work (for now) with per object env interfaces"

        train_configs = generate_parallel_train_configs(
            cfg.experiment, cfg.experiment.config.logging.run_name
        )
        train_configs = filter_episode_configs(train_configs, cfg.episodes)
        if cfg.print_cfg:
            print("Printing configs for spot checking")
            for config in train_configs:
                print_config(config)
        else:
            run_episodes_parallel(
                train_configs,
                cfg.num_parallel,
                cfg.experiment.config.logging.run_name,
                train=True,
            )

    if cfg.experiment.config.do_eval:
        assert issubclass(
            cfg.experiment.config.eval_env_interface_class,
            EnvironmentInterfacePerObject,
        ), "parallel experiments only work (for now) with per object env interfaces"

        eval_configs = generate_parallel_eval_configs(
            cfg.experiment, cfg.experiment.config.logging.run_name
        )
        eval_configs = filter_episode_configs(eval_configs, cfg.episodes)
        if cfg.print_cfg:
            print("Printing configs for spot checking")
            for config in eval_configs:
                print_config(config)
        else:
            run_episodes_parallel(
                eval_configs,
                cfg.num_parallel,
                cfg.experiment.config.logging.run_name,
                train=False,
            )
