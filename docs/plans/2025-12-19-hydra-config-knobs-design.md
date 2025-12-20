# Hydra Config Knobs Design

## Problem

Monty's Hydra configuration has deep nesting: experiments compose monty configs, which compose sensor modules, learning modules, motor systems, etc. When users need to change a deeply nested value (e.g., `max_nneighbors` in the hypotheses updater), they face two bad options:

1. **Config file duplication**: Copy entire config blocks just to change one value, leading to config explosion
2. **Long CLI overrides**: `config.monty_config.learning_module_configs.learning_module_0.learning_module_args.hypotheses_updater_args.max_nneighbors=10`

Most common use cases:
- Changing 1-2 hyperparameters for experiment variants
- Toggling features (noise, logging, etc.)
- Swapping sensor/LM types

## Solution

A hybrid approach:
1. **Knobs**: Top-level variables for frequently-changed scalar values
2. **Config groups**: Semantic groupings for structural variations (different classes, different argument sets)

## Design

### Knobs Architecture

Knobs are root-level variables that get resolved at composition time before the experiment class receives its `config` argument.

#### Directory Structure

```
conf/knobs/
├── defaults.yaml          # composes all sub-files
├── learning_module.yaml   # LM-related knobs
├── sensor.yaml            # sensor-related knobs
├── motor_system.yaml      # motor-related knobs
└── runtime.yaml           # runtime/experiment knobs
```

#### File Format

```yaml
# conf/knobs/defaults.yaml
# @package _global_
defaults:
  - learning_module
  - sensor
  - motor_system
  - runtime

knobs: {}  # merged from defaults above
```

```yaml
# conf/knobs/learning_module.yaml
# @package _global_
knobs:
  max_nneighbors: 5
  evidence_threshold: 80
  max_match_distance: 0.01
```

```yaml
# conf/knobs/sensor.yaml
# @package _global_
knobs:
  noise_enabled: false
  save_raw_obs: false
  blur_sigma: 0.5
```

```yaml
# conf/knobs/runtime.yaml
# @package _global_
knobs:
  min_eval_steps: 20
  n_eval_epochs: 14
  fast_mode: true
```

#### Consuming Knobs in Components

Components reference knobs via interpolation:

```yaml
# conf/monty/learning_module/components/hypotheses_updater/fast.yaml
max_nneighbors: ${knobs.max_nneighbors}
```

```yaml
# conf/monty/sensor_module/components/patch_distant.yaml
defaults:
  - _habitat_sm

sensor_module_args:
  save_raw_obs: ${knobs.save_raw_obs}
  noise_params: ${oc.if:${knobs.noise_enabled},${vars.default_all_noise_params},null}
```

#### Experiment Config Integration

```yaml
# conf/experiment/benchmarks_v2/base_config.yaml
# @package _global_

defaults:
  - /knobs: defaults              # loads knobs at root level
  - /vars: defaults               # loads vars under config.vars
  - /monty: benchmark_evidence_sota
  - /agent_config: patch_view_habitat
  - /environment: ycb_distinct

config:
  n_eval_epochs: ${knobs.n_eval_epochs}
  model_name_or_path: ${vars.pretrained_dir}/surf_agent_1lm_10distinctobj/pretrained/
  # ...
```

### Knobs vs Vars

| Namespace | Purpose | Package | Example |
|-----------|---------|---------|---------|
| `knobs.*` | Tunable experiment parameters | `_global_` (root) | `max_nneighbors`, `noise_enabled` |
| `vars.*` | Shared data, presets, paths | `config.vars` | `pretrained_dir`, `rotations_all` |

Knobs are frequently overridden per-experiment. Vars are stable shared data.

### Unused Knobs

Knobs can contain more variables than any single class needs. Unused knobs are inert—if nothing references `${knobs.some_value}`, it has no effect.

Rule: If a config references a knob, that knob must exist in `knobs/`.

### Config Groups for Structural Variations

Structural variations (different classes, different argument sets) use config groups:

```yaml
# conf/monty/benchmark_evidence_sota.yaml
defaults:
  - sensor: patch_distant           # swappable config group
  - learning_module: evidence_fast
  - motor_system: informed_goal
```

Override in experiments:

```yaml
# conf/experiment/2d_sensor_experiment.yaml
defaults:
  - /monty: benchmark_evidence_sota
  - override /monty/sensor: two_d_pose_patch
```

### Usage Patterns

#### CLI Override

```bash
python run.py experiment=base_config knobs.max_nneighbors=10 knobs.noise_enabled=true
```

#### Variant Config (Minimal)

```yaml
# conf/experiment/benchmarks_v2/accurate_variant.yaml
# @package _global_

defaults:
  - base_config

knobs:
  max_nneighbors: 10
  noise_enabled: true
```

#### Hydra Multirun Sweeps

```bash
python run.py -m experiment=base_config knobs.max_nneighbors=5,10,15
```

## Final Config Structure

```yaml
# Composed config at runtime:

knobs:                    # root level, resolved before instantiation
  max_nneighbors: 5
  noise_enabled: false
  # ...

config:                   # passed to experiment class
  vars:
    pretrained_dir: /path/to/models
    rotations_all: [[0,0,0], ...]
  model_name_or_path: /path/to/models/...
  n_eval_epochs: 14       # resolved from ${knobs.n_eval_epochs}
  monty_config:
    learning_module_configs:
      learning_module_0:
        learning_module_args:
          hypotheses_updater_args:
            max_nneighbors: 5   # resolved from ${knobs.max_nneighbors}
```

## Migration Strategy

### Phase 1: Add Knobs Infrastructure

1. Create `conf/knobs/` directory with initial files
2. Add `- /knobs: defaults` to experiment configs
3. No behavior change—just plumbing

### Phase 2: Wire One Component (Proof of Concept)

Pick `max_nneighbors`:

1. Add to `knobs/learning_module.yaml`
2. Update component: `max_nneighbors: ${knobs.max_nneighbors}`
3. Test: `python run.py experiment=base_config knobs.max_nneighbors=10`

### Phase 3: Migrate Remaining Knobs

For each tunable value:
1. Add to appropriate `knobs/*.yaml`
2. Replace hardcoded value with interpolation
3. Delete duplicate configs that existed only to change that value

### Phase 4: Consolidate Config Groups

1. Identify structural variations (sensor types, LM types)
2. Ensure clean config groups with semantic names
3. Update experiments to use `override /monty/sensor: xxx`

### What to Delete After Migration

- Duplicate experiment configs differing by 1-2 values
- Inlined monty configs (e.g., `benchmark_evidence_noisy.yaml` → base + knobs)

## Benefits

| Before | After |
|--------|-------|
| 10+ experiment variants with duplicated config | Few base configs + knobs overrides |
| 60-line inlined configs | Composed from smaller pieces |
| Values repeated across files | Single source of truth |
| Deep CLI override paths | `knobs.max_nneighbors=10` |

## Open Questions

1. Should `min_eval_steps` move from `vars` to `knobs`? (Recommendation: yes)
2. Naming convention for knobs: flat (`max_nneighbors`) vs prefixed (`lm_max_nneighbors`)?
3. Which existing experiment configs can be consolidated after migration?