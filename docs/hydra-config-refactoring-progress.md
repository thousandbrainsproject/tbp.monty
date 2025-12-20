# Hydra Config Refactoring Progress

**Plan location:** `/Users/hlee/.claude/plans/eager-waddling-wozniak.md`

**Branch:** `extract_edge`

**Last updated:** 2025-12-20

---

## Summary

Refactoring Hydra configs to enable clean `override /group: value` syntax instead of verbose `config/group/file@package.target` syntax.

---

## Progress

### Phase 1: Complete Top-Level Config Groups
**Status:** Deferred (incremental)

The existing top-level config groups (`monty/`, `agent_config/`, `train_env/`, `eval_env/`) have enough configs for current experiments. Additional configs (motor_system, learning_module variants) will be added as needed.

---

### Phase 2: Rename Config Groups
**Status:** COMPLETED

| Task | Status |
|------|--------|
| Rename `conf/environment/` to `conf/agent_config/` | Done |
| Update `conf/experiment.yaml` defaults | Done |

**Changes made:**
- `git mv conf/environment conf/agent_config`
- Updated `experiment.yaml`: `environment: patch_view_habitat` → `agent_config: patch_view_habitat`

---

### Phase 3: Create Migration Pattern Documentation
**Status:** COMPLETED

| Task | Status |
|------|--------|
| Add config groups table to README | Done |
| Document new vs legacy syntax | Done |
| Document mode config group | Done |

**Files modified:**
- `conf/README.md` - Added config groups table, syntax examples, mode documentation

---

### Phase 4: Update Existing 2d_sm Experiments
**Status:** PARTIAL (incremental)

| Task | Status |
|------|--------|
| Verify disk_* experiments use new syntax | Done (already using it) |
| Migrate lvl1_* experiments | Deferred (need monty subgroup expansion) |

**Notes:**
- `disk_learning_control.yaml` and related files already use `override /presets:`, `override /logging:`, etc.
- 7 files in `conf/experiment/2d_sm/` still use legacy syntax - they reference fine-grained monty configs that don't have top-level equivalents yet.

---

### Phase 5: Add Mode Config Group
**Status:** COMPLETED

| Task | Status |
|------|--------|
| Create `conf/mode/inference.yaml` | Done |
| Create `conf/mode/pretrain.yaml` | Done |
| Add `mode: inference` to `experiment.yaml` | Done |
| Update README documentation | Done |

**New files:**
- `conf/mode/inference.yaml` - sets `do_train: false`, `do_eval: true`
- `conf/mode/pretrain.yaml` - sets `do_train: true`, `do_eval: false`

**Usage:**
```bash
# Inference (default)
python run.py experiment=2d_sm/inference/disk_inference_control_on_control

# Pretraining
python run.py experiment=2d_sm/learning/disk_learning_control mode=pretrain
```

---

### Phase 6: Refactor Benchmark Base Config
**Status:** COMPLETED

Created a refactored version of `base_config_10distinctobj_dist_agent.yaml` using new syntax.

| Task | Status |
|------|--------|
| Create `conf/presets/eval.yaml` | Done |
| Create `conf/eval_env/ycb_distinct.yaml` | Done |
| Create `conf/monty/monty_class/evidence_matching.yaml` | Done |
| Create `conf/monty/benchmark_evidence_sota.yaml` | Done |
| Create refactored benchmark in `benchmarks_v2/` | Done |
| Generate and compare snapshots | Done |

**New files:**
- `conf/presets/eval.yaml` - Evaluation preset (do_train=false, do_eval=true)
- `conf/eval_env/ycb_distinct.yaml` - YCB 10 distinct objects list
- `conf/monty/monty_class/evidence_matching.yaml` - Evidence matching Monty class
- `conf/monty/benchmark_evidence_sota.yaml` - Composite benchmark monty config
- `conf/experiment/benchmarks_v2/base_config_10distinctobj_dist_agent.yaml` - Refactored benchmark

**Syntax comparison:**

Original (11 lines of defaults):
```yaml
defaults:
  - config/eval
  - config/logging/parallel_evidence_lm
  - config/monty/patch_and_view_sota@config.monty_config
  - config/monty/learning_modules/clear_learning_module_configs@config.monty_config
  - config/monty/learning_modules/lower_max_nneighbors_1lm@config.monty_config.learning_module_configs
  - config/monty/args/clear_monty_args@config.monty_config
  - config/monty/args/defaults@config.monty_config.monty_args
  - config/environment/patch_view_finder_mount_habitat@config.env_interface_config
  - config/environment_interface/per_object@config.eval_env_interface_args
  - config/environment_interface/ycb/distinct_objects@config.eval_env_interface_args.object_names
```

Refactored (5 lines of defaults):
```yaml
defaults:
  - override /presets: eval
  - override /logging: defaults
  - override /monty: benchmark_evidence_sota
  - override /agent_config: patch_view_habitat
  - override /eval_env: ycb_distinct
```

**Snapshot verification:**
- Original: `tests/conf/snapshots/base_config_10distinctobj_dist_agent.yaml`
- Refactored: `tests/conf/snapshots/benchmarks_v2_base_config_10distinctobj_dist_agent.yaml`
- Key values verified identical: `monty_class`, `max_nneighbors: 5`, object lists, motor/sensor/LM configs

---

### Phase 7: Migrate presets/ into mode/
**Status:** COMPLETED

Eliminated the `presets/` config group, absorbing all functionality into `mode/`.

| Task | Status |
|------|--------|
| Create `conf/mode/_defaults.yaml` with base values | Done |
| Modify `mode/inference.yaml` to extend _defaults | Done |
| Modify `mode/pretrain.yaml` to extend _defaults | Done |
| Remove `presets: defaults` from experiment.yaml | Done |
| Update experiments using presets (3 files) | Done |
| Delete `conf/presets/` directory | Done |
| Update README documentation | Done |

**Design decision:** Modes are pure execution modes (do NOT select monty configs). This keeps concerns separated:
- `mode/` controls train/eval flags, base settings, and env nullification
- `monty/` selection stays in experiment.yaml defaults or experiment overrides

**Bug fix:** The old system had incorrect composition precedence where `mode: inference` (default) was overriding `presets: supervised_pretraining` for train/eval flags. Pretraining experiments incorrectly had `do_train: false, do_eval: true`. The new unified `mode: pretrain` correctly sets `do_train: true, do_eval: false`.

**New mode/ structure:**
```
conf/mode/
├── _defaults.yaml    # Base experiment values (epochs, steps, seed, etc.)
├── inference.yaml    # do_train=false, do_eval=true, nullify train_env
└── pretrain.yaml     # do_train=true, do_eval=false, supervised_lm_ids=all, nullify eval_env
```

**Experiments migrated:**
- `2d_sm/learning/disk_learning_control.yaml`: `override /presets: supervised_pretraining` → `override /mode: pretrain`
- `2d_sm/learning/angles_control.yaml`: `override /presets: supervised_pretraining` → `override /mode: pretrain`
- `benchmarks_v2/base_config_10distinctobj_dist_agent.yaml`: `override /presets: eval` → removed (inference is default)

---

## Validation

Both syntax styles validated:

```bash
# New syntax (2d_sm experiments)
conda run -n tbp.monty PYTHONPATH=src:$PYTHONPATH python run.py \
  experiment=2d_sm/inference/disk_inference_control_on_control --cfg job

# Legacy syntax (benchmarks)
conda run -n tbp.monty PYTHONPATH=src:$PYTHONPATH python run.py \
  experiment=benchmarks/base_config_10distinctobj_dist_agent --cfg job

# Mode override
conda run -n tbp.monty PYTHONPATH=src:$PYTHONPATH python run.py \
  experiment=2d_sm/learning/disk_learning_control mode=pretrain --cfg job
```

---

## Current Config Structure

```
conf/
├── experiment.yaml              # Entry point (7 config groups)
├── vars/                        # Shared variables for interpolation
│   └── defaults.yaml            # ${vars.pretrained_dir}, ${vars.rotations_all}, etc.
├── mode/                        # Execution mode with base settings
│   ├── _defaults.yaml           # Base experiment values (epochs, steps, seed)
│   ├── inference.yaml           # do_train=false, do_eval=true
│   └── pretrain.yaml            # do_train=true, do_eval=false, supervised
├── agent_config/                # RENAMED from environment/
│   ├── patch_view_habitat.yaml
│   ├── patch_view_habitat_rgb_blur.yaml
│   └── surface_habitat.yaml
├── monty/                       # Monty agent configs
│   ├── patch_and_view_inference.yaml
│   ├── patch_and_view_learning.yaml
│   ├── benchmark_evidence_sota.yaml
│   ├── surface_and_view.yaml
│   ├── five_lm.yaml
│   ├── monty_class/
│   │   ├── graph_matching.yaml
│   │   └── evidence_matching.yaml
│   ├── motor_system/           # Existing subgroup
│   ├── learning_module/        # Existing subgroup
│   ├── sensor_module/          # Existing subgroup
│   └── connectivity/           # Existing subgroup
├── train_env/                   # Training object lists
├── eval_env/                    # Evaluation object lists
│   ├── per_object.yaml
│   └── ycb_distinct.yaml
├── logging/                     # Logging configs
│   ├── defaults.yaml            # Base logging (DETAILED, JSON+CSV+Reproduce)
│   ├── pretrain.yaml            # Silent for training
│   ├── csv.yaml                 # Minimal - CSV stats only
│   ├── eval.yaml                # Feature eval with wandb
│   └── parallel.yaml            # Parallel benchmarks (BASIC, wandb)
└── experiment/
    ├── config/                  # Legacy configs (still supported)
    ├── benchmarks/              # Benchmark experiments (legacy syntax)
    ├── benchmarks_v2/           # Refactored benchmarks (new syntax)
    │   ├── base_config_10distinctobj_dist_agent.yaml
    │   └── randrot_noise_10distinctobj_surf_agent.yaml
    └── 2d_sm/                   # 2D sensor experiments (new syntax)
```

---

### Phase 8: Migrate Surface Agent Benchmark
**Status:** COMPLETED

Migrated the noisy surface agent random rotation benchmark from legacy to new syntax.

| Task | Status |
|------|--------|
| Create `conf/experiment/benchmarks_v2/randrot_noise_10distinctobj_surf_agent.yaml` | Done |
| Validate config structure | Done |

**New file:**
- `conf/experiment/benchmarks_v2/randrot_noise_10distinctobj_surf_agent.yaml`

**Syntax comparison:**

Original (15 lines of defaults with deep inheritance):
```yaml
defaults:
  - randrot_noise_10distinctobj_dist_agent
  - config/monty/clear_monty_config@config
  - config/monty/surface_and_view_sota@config.monty_config
  - config/monty/learning_modules/clear_learning_module_configs@config.monty_config
  - config/monty/learning_modules/default_evidence_surf_1lm@config.monty_config.learning_module_configs
  - config/monty/motor_system/clear_motor_system_config@config.monty_config
  - config/monty/motor_system/cur_informed_surface_goal_state_driven@config.monty_config.motor_system_config
  - config/monty/args/clear_monty_args@config.monty_config
  - config/monty/args/defaults@config.monty_config.monty_args
  - config/monty/sensor_modules/clear_sensor_module_configs@config.monty_config
  - config/monty/sensor_modules/sensor_module/default_all_noisy_surf_agent@config.monty_config.sensor_module_configs.sensor_module_0
  - config/environment/clear_env_interface_config@config
  - config/environment/surface_view_finder_mount_habitat@config.env_interface_config
```

Refactored (5 lines of defaults, flat structure):
```yaml
defaults:
  - /environment/object_sampler: random_rotation
  - override /logging: defaults
  - override /monty: benchmark_evidence_sota_surface
  - override /agent_config: surface_habitat
  - override /environment: ycb_distinct
```

**Key configuration verified:**
- Surface sensor module (`is_surface_sm: true`)
- Curvature-informed surface policy
- Noise params via `${vars.default_all_noise_params}`
- Random rotation object sampler
- 10 distinct YCB objects
- `n_eval_epochs: 10`, `max_total_steps: 5000`

---

### Phase 9: Complete Logging Config Group
**Status:** COMPLETED

Added missing logging configs to `conf/logging/` for feature parity with legacy `conf/experiment/config/logging/`.

| Task | Status |
|------|--------|
| Create `conf/logging/csv.yaml` | Done |
| Create `conf/logging/eval.yaml` | Done |
| Create `conf/logging/parallel.yaml` | Done |

**New files:**
- `conf/logging/csv.yaml` - Minimal logging (CSV stats only)
- `conf/logging/eval.yaml` - Evaluation with wandb handlers
- `conf/logging/parallel.yaml` - Parallel benchmark runs (BASIC level, WARNING python, wandb)

**Mapping from legacy:**

| Legacy Config | New Equivalent | Notes |
|---------------|----------------|-------|
| `defaults.yaml` | `defaults.yaml` | Already existed |
| `pretrain.yaml` | `pretrain.yaml` | Already existed |
| `csv.yaml` | `csv.yaml` | Created |
| `eval.yaml` | `eval.yaml` | Created |
| `eval_evidence_lm.yaml` | `parallel.yaml` | Consolidated |
| `parallel_evidence_lm.yaml` | `parallel.yaml` | Consolidated |
| `clear_logging.yaml` | N/A | Not needed with `override` syntax |

**New logging/ structure:**
```
conf/logging/
├── defaults.yaml    # Base logging (DETAILED, JSON+CSV+Reproduce)
├── pretrain.yaml    # Silent for training
├── csv.yaml         # Minimal - CSV stats only
├── eval.yaml        # Feature eval with wandb
└── parallel.yaml    # Parallel benchmarks (BASIC, WARNING, wandb)
```

---

## Next Steps

1. **Migrate remaining benchmarks to `benchmarks_v2/`** (incremental):
   - Use `base_config_10distinctobj_dist_agent.yaml` as template
   - Other benchmarks can extend this base with overrides
   - Currently ~50 files in `benchmarks/` using legacy syntax

2. **Migrate remaining 2d_sm experiments** (when needed):
   - `base_lvl1.yaml` and derivatives still use legacy syntax
   - Can be migrated as they're touched

3. **Consider consolidating benchmarks/ and benchmarks_v2/**:
   - Once all benchmarks are migrated, can replace old directory
   - Update test snapshots accordingly

4. **Consider adding more mode variants** (optional):
   - `mode/train.yaml` - both train and eval (do_train=true, do_eval=true)

---

## Files Changed This Session

### Phase 12: Add YCB Similar Environment and Migrate Benchmarks

| File | Change |
|------|--------|
| `conf/environment/ycb_similar.yaml` | Created - YCB similar objects list |
| `conf/experiment/benchmarks_v2/base_10simobj_surf_agent.yaml` | Created - surface agent with similar objects |
| `conf/experiment/benchmarks_v2/randrot_noise_10simobj_dist_agent.yaml` | Created - dist agent, noise, random rotation, similar objects |
| `conf/experiment/benchmarks_v2/randrot_noise_10simobj_surf_agent.yaml` | Created - surface agent, noise, random rotation, similar objects |
| `conf/experiment/benchmarks_v2/randomrot_rawnoise_10distinctobj_surf_agent.yaml` | Created - surface agent with raw depth noise (inlined transform) |
| `conf/experiment/benchmarks/base_10simobj_surf_agent.yaml` | Deleted |
| `conf/experiment/benchmarks/randrot_noise_10simobj_dist_agent.yaml` | Deleted |
| `conf/experiment/benchmarks/randrot_noise_10simobj_surf_agent.yaml` | Deleted |
| `conf/experiment/benchmarks/randomrot_rawnoise_10distinctobj_surf_agent.yaml` | Deleted |

### Phase 11: Add 5LM Config Group and Migrate Benchmark

Created new config group for 5LM (multi-learning-module) experiments.

| File | Change |
|------|--------|
| `conf/monty/connectivity/five_lm.yaml` | Created - 5 patch + view_finder connectivity with vote matrix |
| `conf/agent_config/five_lm_habitat.yaml` | Created - 5LM habitat agent config |
| `conf/monty/benchmark_evidence_sota_5lm.yaml` | Created - 5LM evidence matching monty config |
| `conf/experiment/benchmarks_v2/randrot_noise_10distinctobj_5lms_dist_agent.yaml` | Created - 5LM benchmark |
| `conf/experiment/benchmarks/randrot_noise_10distinctobj_5lms_dist_agent.yaml` | Deleted |

**New config group structure:**
- `connectivity/five_lm.yaml` - reusable for any 5LM experiment
- `benchmark_evidence_sota_5lm.yaml` - composes 5 evidence LMs with vote matrix
- `agent_config/five_lm_habitat.yaml` - 5 patch sensors + view finder

### Phase 10: Migrate Benchmarks to v2

| File | Change |
|------|--------|
| `conf/experiment/benchmarks_v2/randrot_noise_10distinctobj_dist_on_distm.yaml` | Created - dist agent with noise, random rotation, supervised_pre_training_base model |
| `conf/experiment/benchmarks_v2/randrot_10distinctobj_surf_agent.yaml` | Created - surface agent with random rotation (no noise) |
| `conf/experiment/benchmarks/randrot_noise_10distinctobj_dist_on_distm.yaml` | Deleted |
| `conf/experiment/benchmarks/randrot_10distinctobj_surf_agent.yaml` | Deleted |

**Key simplifications:**
- Uses `override /logging: parallel` with only wandb_group/run_name overrides
- Uses unified `env_interface_class` (instead of `eval_env_interface_class`)
- ~22-26 lines vs original's deep inheritance chain

### Phase 9: Complete Logging Config Group

| File | Change |
|------|--------|
| `conf/logging/csv.yaml` | Created - minimal CSV-only logging |
| `conf/logging/eval.yaml` | Created - evaluation with wandb |
| `conf/logging/parallel.yaml` | Created - parallel benchmarks |

### Phase 8: Migrate Surface Agent Benchmark

| File | Change |
|------|--------|
| `conf/experiment/benchmarks_v2/randrot_noise_10distinctobj_surf_agent.yaml` | Created - surface agent with noise and random rotation |

### Phase 7: Migrate presets/ into mode/

| File | Change |
|------|--------|
| `conf/mode/_defaults.yaml` | Created - base experiment values |
| `conf/mode/inference.yaml` | Modified - extends _defaults, removed monty override |
| `conf/mode/pretrain.yaml` | Modified - extends _defaults, removed monty/presets overrides |
| `conf/experiment.yaml` | Modified - removed `presets: defaults` from defaults |
| `conf/experiment/2d_sm/learning/disk_learning_control.yaml` | Modified - presets → mode |
| `conf/experiment/2d_sm/learning/angles_control.yaml` | Modified - presets → mode |
| `conf/experiment/benchmarks_v2/base_config_10distinctobj_dist_agent.yaml` | Modified - removed presets override |
| `conf/presets/` | Deleted directory |
| `conf/README.md` | Updated - removed presets docs, updated mode docs |

### Earlier Phases

| File | Change |
|------|--------|
| `conf/environment/` → `conf/agent_config/` | Renamed directory |
| `conf/experiment.yaml` | Updated defaults: `agent_config`, `mode`, `monty` |
| `conf/README.md` | Added documentation for config groups, syntax, modes |
| `conf/mode/inference.yaml` | Created |
| `conf/mode/pretrain.yaml` | Created |
| `conf/presets/eval.yaml` | Created - evaluation preset |
| `conf/eval_env/ycb_distinct.yaml` | Created - YCB distinct objects |
| `conf/monty/monty_class/evidence_matching.yaml` | Created - evidence matching class |
| `conf/monty/benchmark_evidence_sota.yaml` | Created - composite benchmark config |
| `conf/experiment/benchmarks_v2/base_config_10distinctobj_dist_agent.yaml` | Created - refactored benchmark |
| `tests/conf/snapshots/benchmarks_v2_base_config_10distinctobj_dist_agent.yaml` | Created - snapshot for comparison |
