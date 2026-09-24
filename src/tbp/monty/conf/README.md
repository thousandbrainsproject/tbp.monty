# Configuration

This folder contains Monty configurations.

## Experiments

The `experiment` folder contains Monty experiment configurations. Most of these experiments are benchmarks and you can learn more about them at [Running Benchmarks](https://docs.thousandbrains.org/docs/running-benchmarks). The experiments in the `experiment/tutorial` folder are used in [Tutorials](https://docs.thousandbrains.org/docs/tutorials).

### Pretraining models

The pretraining configurations are used for running supervised pretraining experiments to generate the models used for follow-on benchmark evaluation experiments. These only need to be rerun if a functional change to the way a learning module learns is introduced. We keep track of version numbers for these, e.g., `ycb_pretrained_v13`.

Note that instead of running pretraining, you can also download our pretrained models as outlined in our [getting started guide](https://docs.thousandbrains.org/docs/getting-started#42-download-pretrained-models).

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

#### COWS Experiments

Compositional Objects With Stickers (COWS) has small (19 objects) and large (119 objects) variants. Both use `compositional_objects_1.4` and share the five plain 3D children.

Run shared 3D pretraining first, then each size's 2D children before its parent models.

```sh
python run_parallel.py experiment=supervised_pre_training_cows_3d_children
python run_parallel.py experiment=supervised_pre_training_cows_small_2d_children 
python run_parallel.py experiment=supervised_pre_training_cows_small_compositional
python run_parallel.py experiment=supervised_pre_training_cows_small_monolithic
python run_parallel.py experiment=supervised_pre_training_cows_large_2d_children
python run_parallel.py experiment=supervised_pre_training_cows_large_compositional
python run_parallel.py experiment=supervised_pre_training_cows_large_monolithic
```

For more details, see [Running Benchmarks](https://docs.thousandbrains.org/docs/running-benchmarks) and [Benchmark Experiments](https://docs.thousandbrains.org/docs/benchmark-experiments) in the documentation.

## Tests

The `test` folder contains Monty test configurations.

## Validation

The `validate.py` script is a quick way to verify that a configuration is properly formatted. It loads the configuration without running the experiment. You can use it by running `python src/tbp/monty/conf/validate.py experiment=experiment_name`.
