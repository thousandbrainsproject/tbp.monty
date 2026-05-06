---
title: Running Inference with a Pretrained Model
---

# Introduction

This tutorial is a follow-up of our tutorial on [pretraining a model](pretraining-a-model.md). Here we will load the pretraining data into the model and perform object recognition under noisy conditions and several arbitrary object rotations.

> [!NOTE]
> The [first part](pretraining-a-model.md) of this tutorial must be completed for the code in this tutorial to run.
>
# Setting up the Experiment Config for Inference
To follow along, open the `src/tbp/monty/conf/experiment/tutorial/surf_agent_2obj_eval.yaml` file. Let's highlight the various aspects of an evaluation experiment configuration.

```yaml
# @package _global_

defaults:
  # ...

experiment:
  _target_: tbp.monty.frameworks.experiments.object_recognition_experiments.MontyObjectRecognitionExperiment
  config:
    # ...
    # Specify the path where we saved the pretrained model
    model_name_or_path: ${path.expanduser:"~/tbp/results/monty/projects/surf_agent_1lm_2obj_train/pretrained"}
    # ...
    logging:
      # Specify directory where an output directory will be created.
      output_dir: ${path.expanduser:"~/tbp/results/monty/projects"}
      # Specify a name for the evaluation run
      run_name: surf_agent_2obj_eval
```

Now we specify that we want to test the model on "mug" and "banana", and that we want the objects to be rotated a few different ways.

```yaml
# @package _global_

defaults:
  # ...
  - /env_interface: tutorial_eval_2obj_predefined_r3
  # ...

experiment:
  # ...
  # We will evaluate each of the 3 test rotations
  n_train_epochs: 3
  # ...
```

We can see the details at `src/tbp/monty/conf/env_interface/tutorial_eval_2obj_predefined_r3.yaml`:
```yaml
# @package experiment.config

do_eval: true
eval_env_interface_args:
  parent_to_child_mapping: null
  # Specify objects to test.
  object_names:
  - mug
  - banana
  # Specify the rotations in which objects will be presented.
  object_init_sampler:
    _target_: tbp.monty.frameworks.environments.object_init_samplers.Predefined
    rotations:
    - ${np.array:[0.0, 15.0, 30.0]}
    - ${np.array:[7.0, 77.0, 2.0]}
    - ${np.array:[81.0, 33.0, 90.0]}
eval_env_interface_class: ${monty.class:tbp.monty.experiment.environment.OneObjectPerEpisodeInterface}
```

Since this config is going to be a bit more complex, we will build it up in pieces. Here is the configuration for sensor modules from `src/tbp/monty/conf/monty/sensor_module/camera_tutorial_surf_agent_2obj.yaml`:

```yaml
# @package experiment.config.monty_config.sensor_module_configs

sensor_module_0:
  sensor_module_class: ${monty.class:tbp.monty.frameworks.models.sensor_modules.CameraSM}
  sensor_module_args:
    sensor_module_id: patch
    # Features that will be extracted and sent to LM
    # note: don't have to be all the features extracted during pretraining.
    features:
    - pose_vectors
    - pose_fully_defined
    - on_object
    - object_coverage
    - min_depth
    - mean_depth
    - hsv
    - principal_curvatures
    - principal_curvatures_log
    save_raw_obs: false
    # CameraSM will only send an observation to the LM if features or location
    # changed more than these amounts.
    delta_thresholds:
      on_object: 0
      n_steps: 20
      hsv:
      - 0.1
      - 0.1
      - 0.1
      pose_vectors: ${np.list_eval:[np.pi / 4, np.pi * 2, np.pi * 2]}
      principal_curvatures_log:
      - 2
      - 2
      distance: 0.01
    is_surface_sm: true # for surface agent
    # Let's add some noise to the sensor module outputs to make the task more challenging.
    noise_params:
      features:
        pose_vectors: 2 # rotate by random degrees along xyz
        hsv: ${np.array:[0.1, 0.2, 0.2]} # add noise to each channel (the values here specify std. deviation of gaussian for each channel individually)
        principal_curvatures_log: 0.1
        pose_fully_defined: 0.01 # flip bool in 1% of cases
      location: 0.002 # add gaussian noise with 0.002 std (0.2cm)
sensor_module_1:
  sensor_module_class: ${monty.class:tbp.monty.frameworks.models.sensor_modules.Probe}
  sensor_module_args:
    sensor_module_id: view_finder
    save_raw_obs: false
```

There are two main differences between this config and the pretraining sensor module config. First, we are adding some noise to the sensor patch, so we define noise parameters and add them to `sensor_module_0`'s dictionary. Second, we're using `delta_threshold` parameters to only send an observation to the learning module if the features have changed significantly. Note that `CameraSM` can be used as either a surface or distant agent, for which `is_surface_sm` should be appropriately set.

For the learning module, we specify configuration available in `src/tbp/monty/conf/monty/learning_module/evidence_tutorial_surf_agent_2obj.yaml`:

```yaml
# @package experiment.config.monty_config.learning_module_configs

learning_module_0:
  learning_module_class: ${monty.class:tbp.monty.frameworks.models.evidence_matching.learning_module.EvidenceGraphLM}
  learning_module_args:
    # Search the model in a radius of 1cm from the hypothesized location on the model.
    max_match_distance: 0.01 # = 1cm
    # Tolerances within which features must match stored values in order to add evidence
    # to a hypothesis.
    tolerances:
      patch:
        hsv: ${np.array:[0.1, 0.2, 0.2]}
        principal_curvatures_log:
        - 1
        - 1
    # Features where weight is not specified default to 1.
    feature_weights:
      patch:
        # Weighting saturation and value less since these might change under different
        # lighting conditions.
        hsv: ${np.array:[1, 0.5, 0.5]}
    # Most likely hypothesis needs to have 20% more evidence than the others to
    # be considered certain enough to trigger a terminal condition (match).
    x_percent_threshold: 20
    # Update all hypotheses with evidence > 80% of the max hypothesis evidence
    evidence_threshold_config: 80%
    # Config for goal generator of LM which is used for model-based action
    # suggestions, such as hypothesis-testing actions.
    gsg:
      _target_: tbp.monty.frameworks.models.goal_generation.EvidenceGoalGenerator
      # Tolerance(s) when determining goal success
      goal_tolerances:
        location: 0.015 # distance in meters
      # Number of necessary steps for a hypothesis-testing action to be considered
      min_post_goal_success_steps: 5
    hypotheses_updater_args:
      # Look at features associated with (at most) the 10 closest learned points.
      max_nneighbors: 10
```

Since this learning module will be performing graph matching, we want to specify further parameters that will define various thresholds and weights to be given to different features. We're also using the `EvidenceGraphLM` (rather than the `GraphLM`) which keeps a continuous evidence count for all its hypotheses and is currently the best-performing LM in this codebase.

We then integrate these sensor and learning module configs into the overall experiment config.

```yaml
# @package _global_

defaults:
  - /monty: evidencegraph_exp1000_emin_t3_tot2500
  - /monty/motor_system_config: surface_curvature_informed_goal1
  - /monty/learning_module: evidence_tutorial_surf_agent_2obj
  - /monty/sensor_module: camera_tutorial_surf_agent_2obj
  - /monty/connectivity: 1lm_1sm
  - /environment: habitat_ycb_surf_agent
  - /env_interface: tutorial_eval_2obj_predefined_r3
  - /env_interface/transform: missing_depthto3d_sensor2_semantic0_clip
  - /logging: basic_info_monty_runs

experiment:
  _target_: tbp.monty.frameworks.experiments.object_recognition_experiments.MontyObjectRecognitionExperiment
  config:
    show_sensor_output: true
    max_train_steps: 1000
    max_eval_steps: 500
    max_total_steps: 5000
    # We will evaluate each of the 3 test rotations
    n_train_epochs: 3
    # Specify the path where we saved the pretrained model
    model_name_or_path: ${path.expanduser:"~/tbp/results/monty/projects/surf_agent_1lm_2obj_train/pretrained"}
    n_eval_epochs: 3
    min_lms_match: 1
    seed: 42
    supervised_lm_ids: []
    python_log_level: DEBUG
    logging:
      # Specify directory where an output directory will be created.
      output_dir: ${path.expanduser:"~/tbp/results/monty/projects"}
      # Specify a name for the evaluation run
      run_name: surf_agent_2obj_eval
      wandb_group: gm_eval_runs

```

Note that we have changed the Monty experiment class and the logging config. We also opted for a policy whereby learning-modules generate actions to test hypotheses by producing "goals" for the low-level motor system.

# Running the Experiment

To run the experiment, call the `run.py` script with an experiment name as the `experiment` argument.

```shell
python run.py experiment=tutorial/surf_agent_2obj_eval
```

Once the run is complete, you can inspect the inference logs located in `~/tbp/results/monty/projects/surf_agent_2obj_eval`. Since `EvalLoggingConfig` includes a CSV-logger, you should be able to open `eval_stats.csv` and find 6 rows (one for each episode) detailing whether the object was correctly identified, the number of steps required in an episode, etc.

You now know how to pretrain a model and use it to perform inference. In our next tutorial, we will demonstrate how to use Monty for [unsupervised continual learning](unsupervised-continual-learning.md).
