# Configuration

This folder contains Monty configurations.

## Config Groups

The following config groups are available at the top level:

| Group | Purpose | Package Target |
|-------|---------|----------------|
| `mode` | Experiment mode with base settings (inference or pretrain) | `_global_` |
| `logging` | Logging configuration | `config.logging` |
| `monty` | Monty agent configuration (sensors, learning modules, motor) | `config.monty_config` |
| `agent_config` | Environment interface (agent setup, sensors, transforms) | `config.env_interface_config` |
| `environment` | Object selection for both training and evaluation | `config` (sets both `train_env_interface_args` and `eval_env_interface_args`) |

### Modes

The `mode` config group provides base experiment settings and execution mode. All modes extend `_defaults.yaml` which provides shared settings (epochs, steps, seed, etc.).

| Mode | `do_train` | `do_eval` | Description |
|------|------------|-----------|-------------|
| `inference` (default) | false | true | Evaluate pretrained models (nullifies train_env) |
| `pretrain` | true | false | Train models from scratch with supervised learning (nullifies eval_env) |

Usage:
```bash
python run.py experiment=... mode=pretrain
```

Modes do not select monty configs - use the `monty` config group or experiment defaults for that.

## Config Syntax

### New syntax (preferred)

Use `override` syntax to compose experiments from top-level config groups:

```yaml
defaults:
  - override /mode: pretrain
  - override /logging: pretrain
  - override /monty: patch_and_view_learning
  - override /agent_config: patch_view_habitat
  - override /environment: per_object
```

### Legacy syntax

Older experiments reference configs from `experiment/config/` with explicit package targets:

```yaml
defaults:
  - config/monty/patch_and_view@config.monty_config
  - config/environment/patch_view_finder_mount_habitat@config.env_interface_config
```

This syntax is deprecated for new experiments but remains supported for backward compatibility.

## Experiments

The `experiment` folder contains Monty experiment configurations. Most of these experiments are benchmarks and you can learn more about them at [Running Benchmarks](https://thousandbrainsproject.readme.io/docs/running-benchmarks). The experiments in the `experiment/tutorial` folder are used in [Tutorials](https://thousandbrainsproject.readme.io/docs/tutorials).

### Pretraining models

The pretraining configurations are used for running supervised pretraining experiments to generate the models used for follow-on benchmark evaluation experiments. These only need to be rerun if a functional change to the way a learning module learns is introduced. We keep track of version numbers for these, e.g., `ycb_pretrained_v11`.

Note that instead of running pretraining, you can also download our pretrained models as outlined in our [getting started guide](https://thousandbrainsproject.readme.io/docs/getting-started#42-download-pretrained-models).

> [!CAUTION]
>
> Ensure that `config.logging.output_dir` for each pretraining experiment is set to where you want the model to be written to.

#### YCB Experiments

To generate models for the YCB experiments, run the following pretraining:

- `python run_parallel.py experiment=supervised_pre_training_base`
- `python run_parallel.py experiment=only_surf_agent_training_10obj`
- `python run_parallel.py experiment=only_surf_agent_training_10simobj`
- `python run_parallel.py experiment=only_surf_agent_training_allobj`
- `python run_parallel.py experiment=supervised_pre_training_5lms`
- `python run_parallel.py experiment=supervised_pre_training_5lms_all_objects`

All of the above can be run at the same time, in parallel.

#### Objects with logos Experiments

To generate models for the objects with logos experiments, run the following pretraining. Note that some of the pretraining depends on the previous ones.

##### Phase 1

- `python run_parallel.py experiment=supervised_pre_training_flat_objects_wo_logos`

##### Phase 2

- `python run_parallel.py experiment=supervised_pre_training_logos_after_flat_objects`

##### Phase 3

- `python run_parallel.py experiment=supervised_pre_training_curved_objects_after_flat_and_logo`
- `python run_parallel.py experiment=supervised_pre_training_objects_with_logos_lvl1_monolithic_models`
- `python run_parallel.py experiment=supervised_pre_training_objects_with_logos_lvl1_comp_models`
- `python run_parallel.py experiment=supervised_pre_training_objects_with_logos_lvl1_comp_models_resampling`

##### Phase 4

- `python run_parallel.py experiment=supervised_pre_training_objects_with_logos_lvl2_comp_models`
- `python run_parallel.py experiment=supervised_pre_training_objects_with_logos_lvl3_comp_models`
- `python run_parallel.py experiment=supervised_pre_training_objects_with_logos_lvl4_comp_models`


For more details, see [Running Benchmarks](https://thousandbrainsproject.readme.io/docs/running-benchmarks) and [Benchmark Experiments](https://thousandbrainsproject.readme.io/docs/benchmark-experiments) in the documentation.

## Tests

The `test` folder contains Monty test configurations.

## Validation

The `validate.py` script is a quick way to verify that a configuration is properly formatted. It loads the configuration without running the experiment. You can use it by running `python src/tbp/monty/conf/validate.py experiment=experiment_name`.
