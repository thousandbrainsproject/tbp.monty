> _CAUTION_
>
> Ensure that `config.logging.output_dir` for each pretraining experiment is set to where you want the model to be written to.

## YCB Experiments

To generate models for the YCB experiments, run the following pretraining:

- `./run_parallel.py experiment=supervised_pre_training_base`
- `./run_parallel.py experiment=only_surf_agent_training_10obj`
- `./run_parallel.py experiment=only_surf_agent_training_10simobj`
- `./run_parallel.py experiment=only_surf_agent_training_allobj`
- `./run_parallel.py experiment=supervised_pre_training_5lms`
- `./run_parallel.py experiment=supervised_pre_training_5lms_all_objects`

All of the above can be run at the same time, in parallel.

## Objects with logos Experiments

To generate models for the objects with logos experiments, run the following pretraining. Note that some of the pretraining depends on the previous ones.

### Phase 1

- `./run_parallel.py experiment=supervised_pre_training_flat_objects_wo_logos`

### Phase 2

- `./run_parallel.py experiment=supervised_pre_training_logos_after_flat_objects`

### Phase 3

- `./run_parallel.py experiment=supervised_pre_training_curved_objects_after_flat_and_logo`
- `./run_parallel.py experiment=supervised_pre_training_objects_with_logos_lvl1_monolithic_models`
- `./run_parallel.py experiment=supervised_pre_training_objects_with_logos_lvl1_comp_models`
- `./run_parallel.py experiment=supervised_pre_training_objects_with_logos_lvl1_comp_models_resampling`

### Phase 4

- `./run_parallel.py experiment=supervised_pre_training_objects_with_logos_lvl2_comp_models`
- `./run_parallel.py experiment=supervised_pre_training_objects_with_logos_lvl3_comp_models`
- `./run_parallel.py experiment=supervised_pre_training_objects_with_logos_lvl4_comp_models`


For more details, see [Running Benchmarks](https://thousandbrainsproject.readme.io/docs/running-benchmarks) and [Benchmark Experiments](https://thousandbrainsproject.readme.io/docs/benchmark-experiments) in the documentation.
