TODO: update app code to handle the following edits

- we changed `test_rotations_all` from a list of `np.array`s into a list of lists
- we changed `load_environment_interfaces` to remove usage of \["experiment_args"] due to how Hydra is instantiating the configs.
- configuring WandB for run vs run_parallel logging is now done by "MONTY_PARALLEL_WANDB_LOGGING" environment variable
    - ensure that run_parallel sets this to MONTY_PARALLEL_WANDB_LOGGING=1
