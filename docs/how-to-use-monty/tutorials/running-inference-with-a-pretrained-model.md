---
title: Running Inference with a Pretrained Model
---

# Introduction

This tutorial is a follow-up of our tutorial on [pretraining a model](pretraining-a-model.md). Here we will load the pretraining data into the model and perform object recognition under noisy conditions and several arbitrary object rotations.

> [!NOTE]
> The [first part](pretraining-a-model.md) of this tutorial must be completed for the code in this tutorial to run.
>
# Setting up the Experiment Config for Inference
To follow along, open the `conf/experiment/tutorial/surf_agent_2obj_eval.yaml` file. Let's highlight the various aspects of an evaluation experiment configuration.

```yaml
# Basic setup
config:
  # Specify the path where we saved the pretrained model
  model_name_or_path: ${path.expanduser:"~/tbp/results/monty/projects/surf_agent_1lm_2obj_train/pretrained"}
  logging:
    # Specify directory where an output directory will be created.
    output_dir: ${path.expanduser:"~/tbp/results/monty/projects"}
    # Specify a name for the evaluation run
    run_name: surf_agent_2obj_eval
```

Now we specify that we want to test the model on "mug" and "banana", and that we want the objects to be rotated a few different ways.

```yaml
config:
  n_train_epochs: 3 # we will evaluate each of the 3 test rotations
  eval_env_interface_args:
    # Specify objects to test.
    object_names:
      - mug
      - banana
    object_init_sampler:
      _target_: tbp.monty.frameworks.environments.object_initializers.Predefined
      # Specify the rotations in which objects will be presented.
      rotations:
        - [0.0, 15.0, 30.0]
        - [7.0, 77.0, 2.0]
        - [81.0, 33.0, 90.0]
```

Since this config is going to be a bit more complex, we will build it up in pieces. Here is the configuration for sensor modules.

```yaml
config:
  monty_config:
    sensor_module_configs:
      sensor_module_0:
        sensor_module_class: ${monty.class:tbp.monty.frameworks.models.sensor_modules.HabitatSM}
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
          # HabitatSM will only send an observation to the LM if features or location
          # changed more than these amounts.
          delta_thresholds:
            on_object: 0
            n_steps: 20
            hsv: [0.1, 0.1, 0.1]
            pose_vectors: ${np.list_eval:[np.pi / 4, np.pi * 2, np.pi * 2]}
            principal_curvatures_log: [2, 2]
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

There are two main differences between this config and the pretraining sensor module config. First, we are adding some noise to the sensor patch, so we define noise parameters and add them to `sensor_module_0`'s dictionary. Second, we're using `delta_threshold` parameters to only send an observation to the learning module if the features have changed significantly. Note that `HabitatSM` can be used as either a surface or distant agent, for which `is_surface_sm` should be appropriately set.

For the learning module, we specify

```yaml
config:
  monty_config:
    learning_module_configs:
      learning_module_0:
        learning_module_class: ${monty.class:tbp.monty.frameworks.models.evidence_matching.learning_module.EvidenceGraphLM}
        learning_module_args:
          # Search the model in a radius of 1cm from the hypothesized location on the model.
          max_match_distance: 0.01 # =1cm
          # Tolerances within which features must match stored values in order to add evidence
          # to a hypothesis.
          tolerances:
            patch:
              hsv: ${np.array:[0.1, 0.2, 0.2]}
              principal_curvatures_log: [1, 1]
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
          # Config for goal state generator of LM which is used for model-based action
          # suggestions, such as hypothesis-testing actions.
          gsg_class: ${monty.class:tbp.monty.frameworks.models.goal_state_generation.EvidenceGoalStateGenerator}
          gsg_args:
            # Tolerance(s) when determining goal-state success
            goal_tolerances:
              location: 0.015 # distance in meters
            # Number of necessary steps for a hypothesis-testing action to be considered
            min_post_goal_success_steps: 5
          hypotheses_updater_args:
            # Look at features associated with (at most) the 10 closest learned points.
            max_nneighbors: 10
```

Since this learning module will be performing graph matching, we want to specify further parameters that will define various thresholds and weights to be given to different features. We're also using the `EvidenceGraphLM` rather than the `GraphLM` which keeps a continuous evidence count for all its hypotheses and is currently the best-performing LM in this codebase.

We then integrate these sensor and learning module configs into the overall experiment config.

```yaml
# The configuration for the evaluation experiment.
defaults:
  # We use eval config defaults.
  - /experiment/config/eval@config
  - /experiment/config/logging/eval@config.logging
  - /experiment/config/monty/patch_and_view_sota@config.monty_config
  - /experiment/config/monty/args/clear_monty_args@config.monty_config
  - /experiment/config/monty/args/defaults@config.monty_config.monty_args
  - /experiment/config/monty/learning_modules/clear_learning_module_configs@config.monty_config
  - /experiment/config/monty/sensor_modules/clear_sensor_module_configs@config.monty_config
  - /experiment/config/monty/motor_system/clear_motor_system_config@config.monty_config
  - /experiment/config/monty/motor_system/cur_informed_surface_goal_state_driven@config.monty_config.motor_system_config
  - /experiment/config/environment/surface_view_finder_mount_habitat@config.env_interface_config
  - /experiment/config/environment_interface/per_object@config.eval_env_interface_args

_target_: tbp.monty.frameworks.experiments.object_recognition_experiments.MontyObjectRecognitionExperiment
config:
  # load the pre-trained models from this path; this needs to be the same name as used for pretraining
  model_name_or_path: ${path.expanduser:"~/tbp/results/monty/projects/surf_agent_1lm_2obj_train/pretrained"}
  n_eval_epochs: 3
  max_total_steps: 5000
  show_sensor_output: true # live visualization of Monty's observations and MLH
  logging:
    # where to save eval logs
    output_dir: ${path.expanduser:"~/tbp/results/monty/projects"}
    run_name: surf_agent_2obj_eval
    wandb_handlers: [] # remove this line if you, additionally, want to log to WandB
  monty_config:
    monty_args:
      min_eval_steps: 20
    sensor_module_configs:
      sensor_module_0:
        sensor_module_class: ${monty.class:tbp.monty.frameworks.models.sensor_modules.HabitatSM}
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
          # HabitatSM will only send an observation to the LM if features or location
          # changed more than these amounts.
          delta_thresholds:
            on_object: 0
            n_steps: 20
            hsv: [0.1, 0.1, 0.1]
            pose_vectors: ${np.list_eval:[np.pi / 4, np.pi * 2, np.pi * 2]}
            principal_curvatures_log: [2, 2]
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
    learning_module_configs:
      learning_module_0:
        learning_module_class: ${monty.class:tbp.monty.frameworks.models.evidence_matching.learning_module.EvidenceGraphLM}
        learning_module_args:
          # Search the model in a radius of 1cm from the hypothesized location on the model.
          max_match_distance: 0.01 # =1cm
          # Tolerances within which features must match stored values in order to add evidence
          # to a hypothesis.
          tolerances:
            patch:
              hsv: ${np.array:[0.1, 0.2, 0.2]}
              principal_curvatures_log: [1, 1]
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
          # Config for goal state generator of LM which is used for model-based action
          # suggestions, such as hypothesis-testing actions.
          gsg_class: ${monty.class:tbp.monty.frameworks.models.goal_state_generation.EvidenceGoalStateGenerator}
          gsg_args:
            # Tolerance(s) when determining goal-state success
            goal_tolerances:
              location: 0.015 # distance in meters
            # Number of necessary steps for a hypothesis-testing action to be considered
            min_post_goal_success_steps: 5
          hypotheses_updater_args:
            # Look at features associated with (at most) the 10 closest learned points.
            max_nneighbors: 10
  eval_env_interface_class: ${monty.class:tbp.monty.frameworks.environments.embodied_data.InformedEnvironmentInterface}
  eval_env_interface_args:
    object_names:
      - mug
      - banana
    object_init_sampler:
      _target_: tbp.monty.frameworks.environments.object_initializers.Predefined
      rotations:
        - ${np.array:[0.0, 15.0, 30.0]}
        - ${np.array:[7.0, 77.0, 2.0]}
        - ${np.array:[81.0, 33.0, 90.0]}
  train_env_interface_class: ${monty.class:tbp.monty.frameworks.environments.embodied_data.InformedEnvironmentInterface}
  train_env_interface_args:
    object_names:
      - mug
      - banana
    object_init_sampler:
      _target_: tbp.monty.frameworks.environments.object_initializers.Predefined
      rotations:
        - ${np.array:[0.0, 15.0, 30.0]}
        - ${np.array:[7.0, 77.0, 2.0]}
        - ${np.array:[81.0, 33.0, 90.0]}

```

Note that we have changed the Monty experiment class and the logging config. We also opted for a policy whereby learning-modules generate actions to test hypotheses by producing "goal-states" for the low-level motor system.

# Running the Experiment

To run the experiment, call the `run.py` script with an experiment name as the `experiment` argument.

```shell
python run.py experiment=tutorial/surf_agent_2obj_eval
```

Once the run is complete, you can inspect the inference logs located in `~/tbp/results/monty/projects/surf_agent_2obj_eval`. Since `EvalLoggingConfig` includes a CSV-logger, you should be able to open `eval_stats.csv` and find 6 rows (one for each episode) detailing whether the object was correctly identified, the number of steps required in an episode, etc.

You now know how to pretrain a model and use it to perform inference. In our next tutorial, we will demonstrate how to use Monty for [unsupervised continual learning](unsupervised-continual-learning.md).
