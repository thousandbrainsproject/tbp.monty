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

#### Objects with Stickers Experiments

The compositional benchmark uses objects with stickers as a single baseline condition. It includes flat objects, curved objects, and rotated stickers.

To generate models for the objects with stickers experiments, run the following pretraining in order. Some pretraining depends on the previous models.

- `python run_parallel.py experiment=supervised_pre_training_objects_with_stickers_3d_children`
- `python run_parallel.py experiment=supervised_pre_training_objects_with_stickers_2d_children`
- `python run_parallel.py experiment=supervised_pre_training_objects_with_stickers_comp_models`
- `python run_parallel.py experiment=supervised_pre_training_objects_with_stickers_monolithic_models`


For more details, see [Running Benchmarks](https://docs.thousandbrains.org/docs/running-benchmarks) and [Benchmark Experiments](https://docs.thousandbrains.org/docs/benchmark-experiments) in the documentation.

## Tests

The `test` folder contains Monty test configurations.

## Validation

The `validate.py` script is a quick way to verify that a configuration is properly formatted. It loads the configuration without running the experiment. You can use it by running `python src/tbp/monty/conf/validate.py experiment=experiment_name`.
