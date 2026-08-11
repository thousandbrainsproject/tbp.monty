# analysis

Visualization scripts for experiment output, adapted from
`~/tbp/projects/segmentation` (which targeted the old 3-SM / 2-LM setup) to the
current 2-SM / 1-LM configuration used by e.g. `conf/experiment/potato.yaml`:

- `SM_1` is the SalienceSM ("view_finder"); `SM_0` (CameraSM "patch") records
  no raw observations, so it does not appear in the stats.
- Segmentation masks and region proposals live in `SM_1.segmentation_maps` /
  `SM_1.regions` (the old per-frame `info` blocks are gone).
- The voxel grid moved from the SM's `region` telemetry to a top-level
  `attention_system` block, with `age` and `count` per voxel.
- Evidence comes from `LM_0`, the only learning module.

## Usage

```sh
# Animate every episode of the potato run ($MONTY_LOGS/projects/monty_runs/potato):
uv run python analysis/visualize_3d.py

# Or a specific experiment directory / episode / voxel colouring:
uv run python analysis/visualize_3d.py path/to/exp_dir --episode 0 --voxel-feature count
```

GIFs are written to `<exp_dir>/visualizations/segmentation_<episode>.gif`. Each
frame shows the sensor view with the segmentation mask tinted green, the
proposed region in 3D coloured by salience, the attention system's voxel grid,
and (when recorded) the LM's evidence traces and ranking.

`detailed_stats.py` is a small self-contained loader for detailed run stats
(both the single-file and per-episode stores), so these scripts need nothing
outside this repo's environment.
