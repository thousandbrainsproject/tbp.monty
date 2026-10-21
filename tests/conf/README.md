# Snapshot testing

The purpose of snapshot tests is to ensure that any changes to configurations are deliberate.

## Why?

Because the configurations use inheritance and are assembled from many smaller configurations files, a small change in one of the configuration files can have cascading effects across all configurations. This may not be immediately apparent from the small diff in the one configuration file that changed.

By failing a snapshot test every time a configuration is changed, we ensure that the overall cascading effects are deliberate.

## Updating snapshots

Once you observe failing tests and decide that the final configuration changes are as intended, follow this checklist:

1. Update Habitat snapshots in the Conda environment.
    1. With the `tbp.monty` conda environment active: `conda activate tbp.monty`.
    2. Run `python src/tbp/monty/conf/update_snapshots.py`. This automatically updates all of the Habitat experiment snapshots to reflect the current configurations.
2. Update MuJoCo snapshots in the `uv` environment.
    1. Ensure the `uv` environment is setup: `uv sync --extra dev --extra simulator_mujoco`.
    2. Run `uv run python src/tbp/monty/conf/update_snapshots.py --mujoco`. This automatically updates all of the MuJoCo experiment snapshots to reflect the current configurations.
3. Commit the `tests/conf/snapshots` changes to source control to confirm that all changes are intended.

After this, your tests will once again pass, since you are comparing generated configs to the ones you just generated.
