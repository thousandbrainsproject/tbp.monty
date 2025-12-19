# Snapshot testing

The purpose of snapshot tests is to ensure that any changes to configurations are deliberate.

## Why?

Because the configurations use inheritance and are assembled from many smaller configurations files, a small change in one of the configuration files can have cascading effects across all configurations. This may not be immediately apparent from the small diff in the one configuration file that changed.

By failing a snapshot test every time a configuration is changed, we ensure that the overall cascading effects are deliberate.

## Updating snapshots

Once you observe failing tests and decide that the final configuration changes are as intended, follow this two-step checklist:

1. Run `python conf/update_snapshots.py`. This automatically updates all of the snapshots to reflect the current configurations. After this, your tests will once again pass, since you are comparing generated configs to the ones you just generated.

2. Commit the `tests/conf/snapshots` changes to source control to confirm that all changes are intended.


