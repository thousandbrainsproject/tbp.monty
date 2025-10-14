---
title: Running An Experiment From A Different Repository
---

If you have your own repository and want to run your own experiment or a benchmark, you do not need to replicate the `tbp.monty` benchmarks setup.

> [!NOTE]
>
> We have a [tbp.monty_project_template](https://github.com/thousandbrainsproject/tbp.monty_project_template) template repository, so that you can quickly use [tbp.monty](https://github.com/thousandbrainsproject/tbp.monty) for your project, prototype, or paper.

You have the option of running everything from a single script file. The general setup is:

```python
from tbp.monty.frameworks.run_env import setup_env

# call setup_env() to initialize environment used by
# tbp.monty configuration and runtime.
setup_env()

# imports from tbp.monty for use in your configuration

# import run or run_parallel to run the experiment
from tbp.monty.frameworks.run import run  # noqa: E402
from tbp.monty.frameworks.run_parallel import run_parallel  # noqa: E402

experiment_config = # your configuration

run(experiment_config)
# or
run_parallel(experiment_config)
```

A more filled out example:

```python
import os

from tbp.monty.frameworks.run_env import setup_env

setup_env()

from tbp.monty.frameworks.config_utils.config_args import (  # noqa: E402
    LoggingConfig,
    SingleCameraMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (  # noqa: E402
    ExperimentArgs,
    get_env_dataloader_per_object_by_idx,
)
from tbp.monty.frameworks.environments import embodied_data as ED  # noqa: E402
from tbp.monty.frameworks.experiments import MontyExperiment  # noqa: E402
from tbp.monty.frameworks.run import run  # noqa: E402
from tbp.monty.simulators.habitat.configs import (  # noqa: E402
    SinglePTZHabitatDatasetArgs,
)

first_experiment = dict(
    experiment_class=MontyExperiment,
    logging_config=LoggingConfig(
        log_parallel_wandb=False,
        run_name="test",
        output_dir=os.path.expanduser(
            os.path.join(os.getenv("MONTY_LOGS"), "projects/monty_runs/test")
        ),
    ),
    experiment_args=ExperimentArgs(
        do_eval=False,
        max_train_steps=1,
        n_train_epochs=1,
    ),
    monty_config=SingleCameraMontyConfig(),
    dataset_args=SinglePTZHabitatDatasetArgs(),
    train_dataloader_class=ED.EnvironmentDataLoaderPerObject,
    train_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=1),
)

run(first_experiment)
```
