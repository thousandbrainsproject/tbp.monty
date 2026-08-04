---
title: Running An Experiment From A Different Repository
---

If you have your own repository and want to run your own experiment or a benchmark, you do not need to replicate the `tbp.monty` benchmarks setup.

> [!NOTE]
>
> We have a [tbp.monty_project_template](https://github.com/thousandbrainsproject/tbp.monty_project_template) template repository, so that you can quickly use [tbp.monty](https://github.com/thousandbrainsproject/tbp.monty) for your project, prototype, or paper.

First, you will need to install `tbp.monty` as a dependency using `conda`. You will need to add `thousandbrainsproject` to your list of channels, as well as the specific `tbp.monty` dependency:

```yaml
# environment.yaml

# ...
channels:
  # ...
  - thousandbrainsproject

dependencies:
  # ...
  - thousandbrainsproject::tbp.monty
  - pip
  - pip:
    - -e .[dev]
```

You'll also want to add `tbp.monty` and a [Hydra](https://hydra.cc/) dependency to your `pyproject.toml`:

```toml
[project]
# ...
dependencies = [
    # ...
    "tbp.monty", # imported via conda (thousandbrainsproject::tbp.monty)
    "hydra-core>=1.3.2", # Hydra is used for configuring Monty
]
```

With the `enviroment.yaml` configured, create your environment:

```
conda env create -f environment.yaml
```

> [!NOTE]
>
> `tbp.monty` is only distributed for `x86_64` architecture, so if you are on Apple Silicon, your `conda` environment needs to be created using `--subdir=osx64` option:
> ```
> conda env create -f environment.yaml --subdir=osx-64
> ```

Once your `conda` environment is configured, the general setup is:

```python
from tbp.monty.frameworks.run_env import setup_env

# call setup_env() to initialize environment used by
# tbp.monty configuration and runtime.
setup_env()

# import run to run the experiment
from tbp.monty.frameworks.run import main  # noqa: E402

if __name__ == "__main__":
    main()
```

or

```python
from tbp.monty.frameworks.run_env import setup_env

# call setup_env() to initialize environment used by
# tbp.monty configuration and runtime.
setup_env()

# import run_parallel to run the experiment, parallelizing across episodes
from tbp.monty.frameworks.run_parallel import main  # noqa: E402

if __name__ == "__main__":
    main()
```

The above scripts let you run the existing `tbp.monty` experiments from your own project. For example:
```
python run.py experiment=tutorial/first_experiment
```

To run your own experiment, you need to configure it using [Hydra](https://hydra.cc/) (`hydra-core>=1.3.2`). For the purpose of this tutorial, we'll assume that your Hydra configuration is stored in `src/project/conf`:

```
src
`- project
   `- conf
      `- experiment
         `- example.yaml
environment.yaml
pyproject.toml
run.py
run_parallel.py
```

With the above setup in place and a correct configuration present in `example.yaml`, you can now execute it via:
```
python run.py experiment=example
```
or
```
python run_parallel.py experiment=example
```

For a full working example, see the template repository: https://github.com/thousandbrainsproject/tbp.monty_project_template.
