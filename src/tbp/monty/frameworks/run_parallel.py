# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy
import os
import shutil
import time
from pathlib import Path
from typing import List, Mapping, Optional

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import wandb

from tbp.monty.frameworks.config_utils.cmd_parser import create_cmd_parser_parallel
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import (
    MontySupervisedObjectPretrainingExperiment,
    ProfileExperimentMixin,
)
from tbp.monty.frameworks.loggers.monty_handlers import (
    BasicCSVStatsHandler,
    DetailedJSONHandler,
    ReproduceEpisodeHandler,
)
from tbp.monty.frameworks.run import print_config
from tbp.monty.frameworks.utils.dataclass_utils import config_to_dict

"""
Just like run.py, but run episodes in parallel. Running in parallel is as simple as

`python run_parallel.py -e my_exp -n ${NUM_PARALLEL}`

Assumptions and notes:
--- There are some differences between training and testing in parallel. At train time,
    we parallelize across objects, but episodes with the same object are run in serial.
    In this case it is best to set num_parallel to num_objects in your dataset if
    possible. At test time, we separate all (object, pose) combos into separate jobs and
    run them in parallel. In this case, the total_n_jobs is n_objects * n_poses, and
    the more parallelism the better (assuming you won't have more than total_n_jobs
    available).
--- Only certain experiment classes are supported for training. Right now the focus is
    on SupervisedPreTraning. Some classes like ObjectRecognition are inherently
    not parallelizable because each episode depends on results from the previous.
--- Testing is experimental and not yet tested.
"""


def single_train(config):
    os.makedirs(config["logging_config"]["output_dir"], exist_ok=True)
    exp = config["experiment_class"]()
    with exp:
        exp.setup_experiment(config)
        print("---------training---------")
        exp.train()


def single_evaluate(config):
    os.makedirs(config["logging_config"]["output_dir"], exist_ok=True)
    exp = config["experiment_class"]()
    with exp:
        exp.setup_experiment(config)
        print("---------evaluating---------")
        exp.evaluate()
        if config["logging_config"]["log_parallel_wandb"]:
            eval_stats = get_episode_stats(exp, "eval")
            return eval_stats


def get_episode_stats(exp, mode):
    eval_stats = exp.monty_logger.get_formatted_overall_stats(mode, 0)
    exp.monty_logger.flush()
    # Remove overall stats field since they are only averaged over 1 episode
    # and might cause confusion.
    for key in list(eval_stats.keys()):
        if key.startswith("overall"):
            del eval_stats[key]
    return eval_stats


def get_overall_stats(stats):
    overall_stats = dict()
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


def sample_params_to_init_args(params):
    new_params = dict()
    new_params["positions"] = [params["position"]]
    new_params["scales"] = [params["scale"]]
    new_params["rotations"] = [params["euler_rotation"]]

    return new_params


def cat_files(filenames, outfile):
    if os.path.exists(outfile):
        print(f"Removing existing file before writing new one: {outfile}")
        os.remove(outfile)

    Path(outfile).touch()  # create file that captures output
    for file in filenames:
        os.system(f"cat {file} >> {outfile}")


def cat_csv(filenames, outfile):
    dfs = [pd.read_csv(file) for file in filenames]
    df = pd.concat(dfs)
    df.to_csv(outfile, index=False)


def post_parallel_log_cleanup(filenames, outfile, cat_fn):
    # Concatenate files together
    cat_fn(filenames, outfile)

    # Remove json files
    for f in filenames:
        os.remove(f)


def post_parallel_profile_cleanup(parallel_dirs, base_dir, mode):
    profile_dirs = [os.path.join(i, "profile") for i in parallel_dirs]

    episode_csvs = []
    setup_csvs = []
    overall_csvs = []

    for profile_dir in profile_dirs:
        epsd_csvs = [
            i for i in os.listdir(profile_dir) if "episode" in i and i.endswith(".csv")
        ]
        epsd_csv_paths = [os.path.join(profile_dir, i) for i in epsd_csvs]
        setup_csv = os.path.join(profile_dir, "profile-setup_experiment.csv")
        overall_csv = os.path.join(profile_dir, f"profile-{mode}.csv")

        episode_csvs.extend(epsd_csv_paths)
        setup_csvs.append(setup_csv)
        overall_csvs.append(overall_csv)

    episode_outfile = os.path.join(base_dir, f"profile-{mode}_episodes.csv")
    setup_outfile = os.path.join(base_dir, f"profile-setup_experiment.csv")
    overall_outfile = os.path.join(base_dir, f"profile-{mode}.csv")

    post_parallel_log_cleanup(episode_csvs, episode_outfile, cat_fn=cat_csv)
    post_parallel_log_cleanup(setup_csvs, setup_outfile, cat_fn=cat_csv)
    post_parallel_log_cleanup(overall_csvs, overall_outfile, cat_fn=cat_csv)


def move_reproducibility_data(base_dir, parallel_dirs):
    outdir = os.path.join(base_dir, "reproduce_episode_data")
    if os.path.exists(outdir):
        shutil.rmtree(outdir)

    os.makedirs(outdir)
    repro_dirs = [
        os.path.join(pdir, "reproduce_episode_data") for pdir in parallel_dirs
    ]

    # Headache to accont for the fact that everyone is episode 0
    for cnt, rdir in enumerate(repro_dirs):
        files = os.listdir(rdir)
        assert "eval_episode_0_actions.jsonl" in files
        assert "eval_episode_0_target.txt" in files
        action_file = f"eval_episode_{cnt}_actions.jsonl"
        target_file = f"eval_episode_{cnt}_target.txt"
        os.rename(
            os.path.join(rdir, "eval_episode_0_actions.jsonl"),
            os.path.join(outdir, action_file),
        )
        os.rename(
            os.path.join(rdir, "eval_episode_0_target.txt"),
            os.path.join(outdir, target_file),
        )


def post_parallel_eval(configs: List[Mapping], base_dir: str) -> None:
    """Post-execution cleanup after running evaluation in parallel.

    Logs are consolidated across parallel runs and saved to disk.

    Args:
        configs (List[Mapping]): List of configs ran in parallel.
        base_dir (str): Directory where parallel logs are stored.
    """
    print("Executing post parallel evaluation cleanup")
    parallel_dirs = [cfg["logging_config"]["output_dir"] for cfg in configs]

    # Loop over types of loggers, figure out how to clean up each one
    for handler in configs[0]["logging_config"]["monty_handlers"]:
        if issubclass(handler, DetailedJSONHandler):
            filename = "detailed_run_stats.json"
            filenames = [os.path.join(pdir, filename) for pdir in parallel_dirs]
            outfile = os.path.join(base_dir, filename)
            post_parallel_log_cleanup(filenames, outfile, cat_fn=cat_files)
            continue

        if issubclass(handler, BasicCSVStatsHandler):
            filename = "eval_stats.csv"
            filenames = [os.path.join(pdir, filename) for pdir in parallel_dirs]
            outfile = os.path.join(base_dir, filename)
            post_parallel_log_cleanup(filenames, outfile, cat_fn=cat_csv)
            continue

        if issubclass(handler, ReproduceEpisodeHandler):
            move_reproducibility_data(base_dir, parallel_dirs)
            continue

    if configs[0]["logging_config"]["python_log_to_file"]:
        filename = "log.txt"
        filenames = [os.path.join(pdir, filename) for pdir in parallel_dirs]
        outfile = os.path.join(base_dir, filename)
        post_parallel_log_cleanup(filenames, outfile, cat_fn=cat_files)

    if issubclass(configs[0]["experiment_class"], ProfileExperimentMixin):
        post_parallel_profile_cleanup(parallel_dirs, base_dir, "evaluate")

    for pdir in parallel_dirs:
        shutil.rmtree(pdir)


def post_parallel_train(configs: List[Mapping], base_dir: str) -> None:
    """Post-execution cleanup after running training in parallel.

    Object models are consolidated across parallel runs and saved to disk.

    Args:
        configs (List[Mapping]): List of configs ran in parallel.
        base_dir (str): Directory where parallel logs are stored.
    """
    print("Executing post parallel training cleanup")
    parallel_dirs = [cfg["logging_config"]["output_dir"] for cfg in configs]
    pretraining = False
    if issubclass(
        configs[0]["experiment_class"], MontySupervisedObjectPretrainingExperiment
    ):
        parallel_dirs = [os.path.join(pdir, "pretrained") for pdir in parallel_dirs]
        pretraining = True

    if configs[0]["logging_config"]["python_log_to_file"]:
        filename = "log.txt"
        filenames = [os.path.join(pdir, filename) for pdir in parallel_dirs]
        outfile = os.path.join(base_dir, filename)
        post_parallel_log_cleanup(filenames, outfile, cat_fn=cat_files)

    if issubclass(configs[0]["experiment_class"], ProfileExperimentMixin):
        post_parallel_profile_cleanup(parallel_dirs, base_dir, "train")

    config = configs[0]
    exp = config["experiment_class"]()
    with exp:
        exp.setup_experiment(config)
        exp.model.load_state_dict_from_parallel(parallel_dirs, True)
        output_dir = os.path.dirname(configs[0]["logging_config"]["output_dir"])
        if issubclass(
            configs[0]["experiment_class"], MontySupervisedObjectPretrainingExperiment
        ):
            output_dir = os.path.join(output_dir, "pretrained")
        os.makedirs(output_dir, exist_ok=True)
        saved_model_file = os.path.join(output_dir, "model.pt")
        torch.save(exp.model.state_dict(), saved_model_file)

        if pretraining:
            pdirs = [os.path.dirname(i) for i in parallel_dirs]
        else:
            pdirs = parallel_dirs

        for pdir in pdirs:
            print(f"Removing directory: {pdir}")
            shutil.rmtree(pdir)


def run_episodes_parallel(
    configs: List[Mapping],
    num_parallel: int,
    experiment_name: str,
    train: bool = True,
    is_unittest: bool = False,
) -> None:
    """Run episodes in parallel.

    Args:
        configs (List[Mapping]): List of configs to run in parallel.
        num_parallel (int): Maximum number of parallel processes to run. If there
            are fewer configs to run than `num_parallel`, then the actual number of
            processes will be equal to the number of configs.
        experiment_name (str): name of experiment
        train (bool): whether to run training or evaluation
        is_unittest (bool): whether to run in unittest mode
    """
    # Use fewer processes if there are fewer configs than `num_parallel`.
    num_parallel = min(len(configs), num_parallel)
    exp_type = "training" if train else "evaluation"
    print(
        f"-------- Running {exp_type} experiment {experiment_name}"
        f" with {num_parallel} episodes in parallel --------"
    )
    start_time = time.time()
    if configs[0]["logging_config"]["log_parallel_wandb"]:
        run = wandb.init(
            name=experiment_name,
            group=configs[0]["logging_config"]["wandb_group"],
            project="Monty",
            config=configs[0],
            id=configs[0]["logging_config"]["wandb_id"],
        )
    print(f"Wandb setup took {time.time() - start_time} seconds")
    start_time = time.time()
    # Avoid complications with unittests running in parallel and just do in serial
    # but test the config gen and cleanup functions
    if is_unittest:
        run_fn = single_train if train else single_evaluate
        for config in configs:
            run_fn(config)
    else:
        with mp.Pool(num_parallel) as p:
            if train:
                # NOTE: since we don't use wandb logging for training right now
                # it is also not covered here. Might want to add that in the future.
                p.map(single_train, configs)
            else:
                if configs[0]["logging_config"]["log_parallel_wandb"]:
                    all_episode_stats = dict()
                    for result in p.imap(single_evaluate, configs):
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
                    overall_stats["overall/parallel_run_time"] = (
                        time.time() - start_time
                    )
                    overall_stats["overall/num_processes"] = num_parallel
                    run.log(overall_stats)
                else:
                    p.map(single_evaluate, configs)
    end_time = time.time()
    total_time = end_time - start_time

    output_dir = configs[0]["logging_config"]["output_dir"]
    base_dir = os.path.dirname(output_dir)

    if train:
        post_parallel_train(configs, base_dir)
        if configs[0]["logging_config"]["log_parallel_wandb"]:
            csv_path = os.path.join(base_dir, "train_stats.csv")
            if os.path.exists(csv_path):
                train_stats = pd.read_csv(csv_path)
                train_table = wandb.Table(dataframe=train_stats)
                run.log({"train_stats": train_table})
            else:
                print(f"No csv table found at {csv_path} to log to wandb")
    else:
        post_parallel_eval(configs, base_dir)
        if configs[0]["logging_config"]["log_parallel_wandb"]:
            csv_path = os.path.join(base_dir, "eval_stats.csv")
            if os.path.exists(csv_path):
                eval_stats = pd.read_csv(csv_path)
                eval_table = wandb.Table(dataframe=eval_stats)
                run.log({"eval_stats": eval_table})
            else:
                print(f"No csv table found at {csv_path} to log to wandb")

    print(
        f"Total time for {len(configs)} running {num_parallel} episodes in parallel: "
        f"{total_time}"
    )
    if configs[0]["logging_config"]["log_parallel_wandb"]:
        run.finish()

    print(f"Done running parallel experiments in {end_time - start_time} seconds")

    # Keep a record of how long everything takes
    with open(os.path.join(base_dir, "parallel_log.txt"), "w") as f:
        f.write(f"experiment: {experiment_name}\n")
        f.write(f"num_parallel: {num_parallel}\n")
        f.write(f"total_time: {total_time}")


def generate_parallel_train_configs(
    exp: Mapping, experiment_name: str
) -> List[Mapping]:
    """Generate configs for training episodes in parallel.

    Create a config for each object in the experiment. Unlike with parallel eval
    episodes, each parallel config specifies a single object but all rotations.

    Args:
        exp (dict): Config for experiment to be broken into parallel configs.
        experiment_name (str): Name of experiment.

    Returns:
        List of configs for training episodes.

    Note:
        If we view the same object from multiple poses in separate experiments, we
        need to replicate what post_episode does in supervised pre training. To avoid
        this, we just run training episodes parallel across OBJECTS, but poses are
        still in sequence. By contrast, eval episodes are parallel across objects
        AND poses.

    """
    sampler = exp["train_dataloader_args"]["object_init_sampler"]
    sampler.rng = np.random.RandomState(exp["experiment_args"]["seed"])
    object_names = exp["train_dataloader_args"]["object_names"]
    new_configs = []

    for obj in object_names:
        obj_config = copy.deepcopy(exp)

        # No eval
        obj_config["experiment_args"].update(
            do_eval=False, do_train=True, n_train_epochs=1
        )

        # Save results in parallel subdir of output_dir, update run_name
        output_dir = obj_config["logging_config"]["output_dir"]
        run_name = os.path.join(f"{experiment_name}-parallel_train_episode_{obj}")
        obj_config["logging_config"]["run_name"] = run_name
        obj_config["logging_config"]["output_dir"] = os.path.join(
            output_dir, experiment_name, run_name
        )
        obj_config["logging_config"]["wandb_handlers"] = []

        # Object id, pose parameters for single episode
        obj_config["train_dataloader_args"].update(
            object_names=[obj for _ in range(len(sampler))]
        )
        obj_config["train_dataloader_args"][
            "object_init_sampler"
        ].change_every_episode = True

        new_configs.append(obj_config)

    return new_configs


def generate_parallel_eval_configs(exp: Mapping, experiment_name: str) -> List[Mapping]:
    """Generate configs for evaluation episodes in parallel.

    Create a config for each object and rotation in the experiment. Unlike with parallel
    training episodes, a config is created for each object + rotation separately.

    Args:
        exp (dict): Config for experiment to be broken into parallel configs.
        experiment_name (str): Name of experiment.

    Returns:
        List of configs for evaluation episodes.
    """
    sampler = exp["eval_dataloader_args"]["object_init_sampler"]
    sampler.rng = np.random.RandomState(exp["experiment_args"]["seed"])
    object_names = exp["eval_dataloader_args"]["object_names"]
    # sampler_params = sampler.all_combinations_of_params()

    new_configs = []
    epoch_count = 0
    episode_count = 0
    n_epochs = exp["experiment_args"]["n_eval_epochs"]

    params = sample_params_to_init_args(sampler())
    start_seed = exp["experiment_args"]["seed"]

    # Try to mimic the exact workflow instead of guessing
    while epoch_count <= n_epochs:
        for obj in object_names:
            new_config = copy.deepcopy(exp)
            new_config["experiment_args"]["seed"] = start_seed + episode_count

            # No training
            new_config["experiment_args"].update(
                do_eval=True, do_train=False, n_eval_epochs=1
            )

            # Save results in parallel subdir of output_dir, update run_name
            output_dir = new_config["logging_config"]["output_dir"]
            run_name = os.path.join(
                f"{experiment_name}-parallel_eval_episode_{episode_count}"
            )
            new_config["logging_config"]["run_name"] = run_name
            new_config["logging_config"]["output_dir"] = os.path.join(
                output_dir, experiment_name, run_name
            )
            if len(new_config["logging_config"]["wandb_handlers"]) > 0:
                new_config["logging_config"]["wandb_handlers"] = []
                new_config["logging_config"]["log_parallel_wandb"] = True
                new_config["logging_config"]["experiment_name"] = experiment_name
            else:
                new_config["logging_config"]["log_parallel_wandb"] = False

            new_config["eval_dataloader_args"].update(
                object_names=[obj],
                object_init_sampler=PredefinedObjectInitializer(**params),
            )

            new_configs.append(new_config)
            episode_count += 1
            sampler.post_episode()
            params = sample_params_to_init_args(sampler())

        sampler.post_epoch()
        params = sample_params_to_init_args(sampler())

        epoch_count += 1
        if epoch_count >= n_epochs:
            break

    return new_configs


def main(
    all_configs: Optional[Mapping[str, Mapping]] = None,
    exp: Optional[Mapping] = None,
    experiment: Optional[str] = None,
    num_parallel: Optional[int] = None,
    quiet_habitat_logs: bool = True,
    print_cfg: bool = False,
    is_unittest: bool = False,
):
    """Run an experiment in parallel.

    This function runs an experiment in parallel by breaking a config into a set
    of configs that can be run in parallel, executing them, and performing cleanup
    to consolidate logs across parallelized configs. How configs are split up into
    parallelizable configs depends on the type of experiment. For supervised
    pre-training, a config is created for each object, and that config will run all
    object rotations in serial. For evaluation experiments, a config is created for
    each object + a rotation individually. Note that we don't parallelize over rotations
    during training since graphs should be merged in the order in which the object was
    seen; updating the graph at rotation *i* should influence how observations at
    rotation *i + 1* are incorporated into the graph. This is not the case for
    evaluation experiments where graphs aren't modified, and evaluation is therefore
    embarrassingly parallelizable.

    This function is typically called by `run_parallel.py` through command line
    (i.e., `python run_parallel.py -e my_exp`) but is sometimes called directly,
    such as when performing a unit test. As a result, the arguments required depend
    on how the function is called. When run via command line, the following arguments
    are required:
        - `all_configs`
        - `experiment`
        - `num_parallel`

    When called directly, the following arguments are required:
        - `exp`
        - `experiment`
        - `num_parallel`

    Args:
        all_configs (Mapping[str, Mapping], optional): A mapping from config name to
            config dict.
        exp (Mapping, optional): Config for experiment to run. Not required if running
            from command line as the config is selected from `all_configs`.
        experiment (str, optional): Name of experiment to run. Not required if running
            from command line.
        num_parallel (int, optional): Maximum number of parallel processes to run. If
            the config is broken into fewer parallel configs than `num_parallel`, then
            the actual number of processes will be equal to the number of parallel
            configs. Not required if running from command line.
        quiet_habitat_logs (bool): Whether to quiet Habitat logs. Defaults to True.
        print_cfg (bool): Whether to print configs for spot checking. Defaults to
            False.
        is_unittest (bool): Whether to run in unittest mode. If `True`, parallel runs
            are done in serial. Defaults to False.
    """
    # Handle args passed directly (only used by unittest) or command line (normal)
    if experiment:
        assert num_parallel, "missing arg num_parallel"
        assert exp, "missing arg exp"

    else:
        cmd_parser = create_cmd_parser_parallel(experiments=list(all_configs.keys()))
        cmd_args = cmd_parser.parse_args()
        experiment = cmd_args.experiment
        num_parallel = cmd_args.num_parallel
        quiet_habitat_logs = cmd_args.quiet_habitat_logs
        print_cfg = cmd_args.print_cfg
        is_unittest = False

    if quiet_habitat_logs:
        os.environ["MAGNUM_LOG"] = "quiet"
        os.environ["HABITAT_SIM_LOG"] = "quiet"

    exp = exp if is_unittest else all_configs[experiment]
    exp = config_to_dict(exp)

    if len(exp["logging_config"]["run_name"]) > 0:
        experiment = exp["logging_config"]["run_name"]

    # Simplifying assumption: let's only deal with the main type of exp which involves
    # per object dataloaders, otherwise hard to figure out what all goes into an exp
    if exp["experiment_args"]["do_train"]:
        assert issubclass(
            exp["train_dataloader_class"], ED.EnvironmentDataLoaderPerObject
        ), "parallel experiments only work (for now) with per object dataloaders"

        train_configs = generate_parallel_train_configs(exp, experiment)
        if print_cfg:
            print("Printing configs for spot checking")
            for cfg in train_configs:
                print_config(cfg)
        else:
            run_episodes_parallel(
                train_configs,
                num_parallel,
                experiment,
                train=True,
                is_unittest=is_unittest,
            )

    if exp["experiment_args"]["do_eval"]:
        assert issubclass(
            exp["eval_dataloader_class"], ED.EnvironmentDataLoaderPerObject
        ), "parallel experiments only work (for now) with per object dataloaders"

        eval_configs = generate_parallel_eval_configs(exp, experiment)
        if print_cfg:
            print("Printing configs for spot checking")
            for cfg in eval_configs:
                print_config(cfg)
        else:
            run_episodes_parallel(
                eval_configs,
                num_parallel,
                experiment,
                train=False,
                is_unittest=is_unittest,
            )
