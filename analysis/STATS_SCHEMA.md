# Detailed run stats — JSON schema

Schema of the files written by `DetailedJSONHandler`
([monty_handlers.py](../src/tbp/monty/frameworks/loggers/monty_handlers.py)):
either `detailed_run_stats/episode_NNNNNN.json` (one file per episode, with
`detailed_save_per_episode`) or `detailed_run_stats.json` (one JSON line per
episode). Derived from the 3LM `debug_3lm` run of 2026-08-19 and the writer
code; shapes marked with config-dependent symbols vary by experiment.

Symbols: `S` = monty (matching) steps in the episode, `S_lm` = steps a given
LM actually processed, `H×W` = sensor resolution, `N` = variable count.

Since 2026-08-19 the writer uses orjson: files are strict JSON and NaN is
written as `null` (older files hold bare `NaN` tokens; `detailed_stats._loads`
repairs those on read).

## Container

```
{ "<global_episode_id>": { <episode data> } }    # id is a stringified int
```

Episode data is keyed by module. `DetailedJSONHandler.get_detailed_stats`
merges the BASIC-level stats row into each LM's DETAILED block, which is why
LM blocks carry episode bookkeeping alongside telemetry.

| key                | present when                                            | source |
|--------------------|---------------------------------------------------------|--------|
| `LM_<n>`           | always (one per learning module)                        | LM buffer + BASIC stats row |
| `SM_<n>`           | the SM has `save_raw_obs: true`                         | SM telemetry state_dict |
| `attention_system` | the attention system recorded ≥1 voxel grid             | `AttentionSystem.state_dict()` |
| `motor_system`     | always                                                  | motor system state_dict |
| `target`           | always                                                  | env interface |

## `target`

```
primary_target_object:          str
primary_target_position:        [3] float
primary_target_rotation_euler:  [3] float
primary_target_rotation_quat:   [4] float
primary_target_scale:           float
```

## `LM_<n>`

Episode bookkeeping (from the BASIC stats row):

```
total_train_steps, train_episodes, train_epochs,
total_eval_steps, eval_episodes, eval_epochs, episode_seed:  int
mode:                    "train" | "eval"
target: {
  object: str, semantic_id: int,
  rotation: [4], euler_rotation: [3], quat_rotation: [4],
  position: [3], scale: [3],
  consistent_child_objects: [str, ...],       # heterarchy configs
}
```

Sensor-step streams (length `S_lm`; keyed by the LM's input patch id):

```
locations:      { "<patch_id>": [S_lm, 3] }    # world-frame sensed locations
<patch_id>:     { <feature>: [S_lm, dim] }     # one array per SM feature, e.g.
                                               # pose_vectors [S_lm, 9], hsv [S_lm, 3],
                                               # on_object [S_lm, 1], ... (config-dependent)
displacements:  { displacement: [S_lm, 3] }
time:           [S_lm]        relative_time: [S_lm]
mlh_prediction_error: [S_lm - 1]
stepwise_targets_list: [S] str
lm_processed_steps:    [S] bool               # maps monty steps -> LM steps
```

Hypothesis-space streams (per LM step; dicts keyed by object/graph id, with
`N_obj` = points in that object's model that carry hypotheses):

```
possible_matches:   [S_lm] of [str, ...]           # objects still in play
evidences:          [S_lm] of { "<obj>": [N_obj] }
possible_locations: [S_lm] of { "<obj>": [N_obj, 3] }
possible_rotations: [1]    of { "<obj>": [N_obj, 3, 3] }   # only step 0 logged
symmetry_evidence:  [S_lm]
current_mlh:        [S_lm] of { graph_id: str, location: [3],
                                rotation: [3], scale: number, evidence: float }
```

Goal-state generator output (empty lists when the GSG emitted nothing):

```
goal_states: [N] of Goal                           # see Goal below; info holds
                                                   # proposed_surface_loc,
                                                   # hypothesis_to_test (an MLH dict),
                                                   # achieved, matching_step_when_output_goal_set
matching_step_when_output_goal_set: [N] int
goal_state_achieved:                [N] bool
```

Terminal-state fields (null until the LM converges):

```
detected_path, detected_rotation, detected_rotation_quat, detected_scale,
symmetric_rotations, symmetric_locations, individual_ts_reached_at_step,
individual_ts_object, individual_ts_pose, symmetric_rotations_ts:  null | value
detected_location_on_model, detected_location_rel_body:            [3] float
```

## `SM_<n>` (SalienceSM telemetry)

From `SalienceSMTelemetry.state_dict()`
([telemetry.py](../src/tbp/monty/frameworks/models/salience/telemetry.py)).
All keys always present; lists empty when the stream was not produced.
One entry per recorded sensor step (`is_exploring` steps are skipped).

```
raw_observations: [S] of {
  rgba:              [H, W, 4] uint8
  depth:             [H, W] float
  cam_to_world:      [4, 4] float
  sensor_frame_data: [H*W, 4] float      # x, y, z, on-object flag (sensor frame)
  semantic_3d:       [H*W, 4] float      # x, y, z, semantic id (world frame)
}
sm_properties:     [S] of { sm_rotation: [4], sm_location: [3] }
salience_maps:     [S] of [H, W] float   # normalized [0, 1]
segmentation_maps: [S] of [H, W] (0/1)   # null per-step if no strategy ran
regions:           [S] of [N] of { location: [3], weight: int }
```

Plain-camera SMs (`CameraSM` with `save_raw_obs`) write only
`raw_observations` and `sm_properties`.

## `attention_system`

From `AttentionSystem.state_dict()`
([attention_system.py](../src/tbp/monty/attention/attention_system.py) +
its telemetry). Omitted entirely if no voxel grid was ever recorded.

```
voxel_size:     float                    # metres per voxel edge
voxel_grids: [S] of {
  voxels: [V, 3] int                     # voxel indices (lower corners);
                                         # centre = (idx + 0.5) * voxel_size
  weight: [V] float                      # decays toward zero by decay_rate
}
pre_filter_goals:  [S] of [N] of Goal    # goals before voxel-grid filtering
post_filter_goals: [S] of [N] of Goal    # goals passed to the motor system
```

Runs before 2026-08-19 also carry `voxel_lifetime: int` and integer weights
(the countdown-based decay model that preceded ``decay_rate``).

## `motor_system`

```
action_sequence: [S] of [N] of [name_or_dict, params_dict]
  # e.g. ["turn_left", {action, agent_id, rotation_degrees, ...}]
action_details: { pc_heading: [...], avoidance_heading: [...], z_defined_pc: [...] }
policy_selector: { selected_goals: [S] of Goal }
```

## Goal (shared shape)

Serialized `tbp.monty.cmp.Goal` message; appears in `goal_states`,
`pre/post_filter_goals`, and `selected_goals`.

```
{
  location:                   [3] float | null
  morphological_features:     { pose_vectors, pose_fully_defined, on_object } | null
  non_morphological_features: dict | null
  confidence:                 float          # [0, 1]
  use_state:                  bool
  sender_id:                  str            # e.g. "view_finder", "LM_1"
  sender_type:                "SM" | "LM" | "GSG"
  goal_tolerances:            dict | null
  info:                       dict           # sender-specific, often {}
}
```

## Step-count alignment

Streams tick at different rates: `SM_<n>` and `attention_system` record per
monty step where the SM produced output; LM per-sensor-step streams record
only steps that LM processed (`lm_processed_steps` maps between them, via
`cumsum(lm_processed_steps) - 1`); `stepwise_targets_list` and
`lm_processed_steps` are always full monty-step length. Consumers should
clamp or map indices rather than assume equal lengths.

## Loading

Use `analysis/detailed_stats.py`: `DetailedStats.load(exp_dir, episode)`
returns the episode dict with typed accessors (`rgba`, `depth`,
`salience_maps`, `segmentation_maps`, `sm_regions`, `goals`, `fov_centres`,
`voxel_grids`, `pixel_locations`, ...) that convert these blocks to numpy.
