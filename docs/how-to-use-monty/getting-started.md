---
title: Getting Started
description: How to get the code running.
---
> [!NOTE]
> You can find our code at https://github.com/thousandbrainsproject/tbp.monty
>
> This is our open-source repository. We call it **Monty** after Vernon Mountcastle, who proposed cortical columns as a repeating functional unit across the neocortex.

# 1. Get the Code

It is best practice (and required if you ever want to contribute code) first to **make a fork of our repository** and then make any changes on your local fork. To do this you can simply [visit our repository](https://github.com/thousandbrainsproject/tbp.monty) and click on the fork button as shown in the picture below. For more detailed instructions see the [GitHub documentation on Forks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo).

![](../figures/how-to-use-monty/fork.png)


Next, you need to **clone the repository onto your device**. To do that, open the terminal, navigate to the folder where you want the code to be downloaded and run `git clone repository_path.git`. You can find the `repository_path.git` on GitHub by clicking on the `<>Code` button as shown below. For more details see the [GitHub documentation on cloning](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo#cloning-your-forked-repository).


![](../figures/how-to-use-monty/clone.png)

> [!NOTE]
> If you want the same setup as we use at Numenta by default, clone the repository at `${HOME}/tbp/`. If you don't have an `nta` folder in your home directory yet you can run `cd ~; mkdir tbp; cd tbp` to create it. It's not required to clone the code in this folder but it is the path we assume in our tutorials.

## 1.2 Make Sure Your Local Copy is Up-To-Date

If you just forked and cloned this repository, you may skip this step, but any other time you get back to this code, you will want to synchronize it to work with the latest changes.

To make sure your fork is up to date with our repository you need to click on `Sync fork` -> `Update branch` in the GitHub interface. Afterwards, you will need to **get the newest version of the code** into your local copy by running `git pull` inside this folder.

![](../figures/how-to-use-monty/update_branch.png)


You can also update your code using the terminal by calling `git fetch upstream; git merge upstream/main` in the terminal. If you have not linked the upstream repository yet, you may first need to call `git remote add upstream upstream_repository_path.git`

# 2. Set up Your Environment

> [!NOTE]
> The Easiest Way to Set Up this Environment is With Conda
>
> For instructions on how to install conda (Miniconda or Anaconda) on your machine see <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>.

The most straightforward way to set up an environment to run the Monty code is to use conda. Simply **use the conda commands below**. Make sure to `cd` into the tbp.monty root directory before running these commands.

Note that the commands are slightly different depending on whether you are setting up the environment on an Intel or ARM64 architecture and whether you are using Miniconda or the full Anaconda distribution.

## Miniconda

If you have miniconda installed, you can create the environment with the following commands:

```shell Intel
conda env create
conda activate tbp.monty
```
```shell ARM64 (Apple Silicon)
conda env create -f environment_arm64.yml --subdir=osx-64
conda activate tbp.monty
conda config --env --set subdir osx-64
```

## Anaconda

If you have the full anaconda distribution installed, you can create the environment with the following commands:

```shell Intel
conda env create
conda activate tbp.monty
```
```shell ARM64 (Apple Silicon)
conda env create -f environment_arm64.yml
conda activate tbp.monty
```

# 3. Run the Code 🎉

The best way to see whether everything works is to **run the unit tests** using this command:

```shell
pytest
```

Running the tests might take a little while (depending on what hardware you are running on), but in the end, you should see something like this:

![](../figures/how-to-use-monty/tests.png)


Note that by the time you read that, there may be more or less unit tests in the code base, so the exact numbers you see will not match with this screenshot. The important thing is that all tests either pass or are skipped and **none of the tests fail**.

# 4. Run an Experiment

In your usual interaction with this code base, you will most likely run experiments, not just unit tests. You can find experiment configs in the `benchmarks/configs/` folder.

## 4.1 Download the YCB Dataset

A lot of our current experiments are based on the [YCB dataset](https://www.ycbbenchmarks.com/) which is a dataset of 77 3D objects that we render in habitat. To download the dataset, run `python -m habitat_sim.utils.environments_download --uids ycb --data-path ~/tbp/data/habitat`.

## 4.2 Download Pretrained Models

> [!WARNING]
> TODO OSS: Update this section with how to download the models


## [Optional] Set Environment Variables

### MONTY_MODELS

If you did not save the pre-trained models in the `~/tbp/results/monty/pretrained_models/` folder, you will need to set the **MONTY_MODELS** environment variable.

```shell
export MONTY_MODELS=/path/to/your/pretrained/models/dir
```

This path should point to the `pretrained_models` folder that contains the`pretrained_ycb_v8`folders.

### MONTY_LOGS

If you would like to log your experiment results in a different folder than the default path (`~/tbp/results/monty/`) you need to set the **MONTY_LOGS** environment variable.

```shell
export MONTY_LOGS=/path/to/log/folder
```

### WANDB_DIR

We recommend not saving the wandb logs in the repository itself (default save location). If you have already set the MONTY_LOGS variable, you can set the directory like this:

```shell
export WANDB_DIR=${MONTY_LOGS}/wandb
```

## 4.3 Run the Experiment

Now you can finally run an experiment! To do this, simply use this command:

```shell
python benchmarks/run.py -e my_experiment
```

Replace `my_experiment` with the name of one of the experiment configs in `benchmarks/configs/`. For example, a good one to start with could be `randrot_noise_10distinctobj_surf_agent`.

If you want to run an experiment with parallel processing to make use of multiple CPUs, simply use the `run_parallel.py` script instead of the `run.py` script like this:

```shell
python benchmarks/run_parallel.py -e my_experiment
```