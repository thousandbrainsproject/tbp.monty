# Hydra Configuration Tests

These tests verify that our Hydra configurations are in a usable state and haven't been changed unexpectedly. They currently consist of two tests:

- Snapshot Test
- Orphaned Configs Test

## Snapshot Test

The purpose of the snapshot tests in `conf_test.py` is to ensure that any changes to configurations are deliberate.

### Why?

Because the configurations use inheritance and are assembled from many smaller configurations files, a small change in one of the configuration files can have cascading effects across all configurations. This may not be immediately apparent from the small diff in the one configuration file that changed.

By failing a snapshot test every time a configuration is changed, we ensure that the overall cascading effects are deliberate.

### Updating snapshots

Once you observe failing tests and decide that the final configuration changes are as intended, follow this checklist:

1. Update experiment snapshots.
    1. Ensure the `uv` environment is set up: `uv sync --extra dev --extra simulator_mujoco`.
    2. Run `uv run python src/tbp/monty/conf/update_snapshots.py`. This automatically updates all the experiment snapshots to reflect the current configurations.
2. Commit the `tests/conf/snapshots` changes to source control to confirm that all changes are intended.

After this, your tests will once again pass, since you are comparing generated configs to the ones you just generated.

## Orphaned Configs Test

The purpose of the orphaned config tests in `orphan_test.py` is to make sure when we are editing configurations that we don't leave any orphaned component configurations and end up keeping around dead configs.
