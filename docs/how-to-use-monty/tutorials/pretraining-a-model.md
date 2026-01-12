---
title: Pretraining a Model
---

# Introduction

This tutorial demonstrates how to configure and run Monty experiments for pretraining. In the [next tutorial](running-inference-with-a-pretrained-model.md), we show how to load our pretrained model and use it to perform inference. Though Monty is designed for continual learning and does not require separate training and evaluation modes, this set of experiments is useful for understanding many of our [benchmarks experiments](../running-benchmarks.md).

The pretraining differs from [Monty's default learning setup](../../how-monty-works/experiment.md#different-phases-of-learning) in that it is supervised. Under normal conditions, Monty learns by first trying to recognize an object, and then updating its models depending on the outcome of this recognition (unsupervised). Here, we provide the object name and pose to the model directly, so there is no inference phase required. This provides an experimental condition where we can ensure that the model learns the correct models for each of the objects, and we can focus on the inference phase afterward. Naturally, unsupervised learning provides a more challenging but also more naturalistic learning condition, and we will cover this condition later.

Our model will have one surface agent connected to one sensor module connected to one learning module. For simplicity and speed, we will only train/test on two objects in the [YCB](https://www.ycbbenchmarks.com/) dataset.

> [!NOTE]
> **Don't have the YCB Dataset Downloaded?**
>
> You can find instructions for downloading the YCB dataset [here](../getting-started.md#41-download-the-ycb-dataset). Alternatively, you can run these experiments using the built-in Habitat primitives, such as `capsule3DSolid` and `cubeSolid`. Simply change the items in the  `object_names` list.
>


# Config Overview

Monty experiments are defined using a nested dictionary. These dictionaries define the experiment class and associated simulation parameters, logging configs, the Monty model (which includes sensor modules, learning modules, and a motor system), and the environment interface. This is the basic structure of a complete experiment config, along with their expected types:

- `_target_`: `MontyExperiment` Manages the highest-level calls to the environment and Monty model.
- `config`: Arguments supplied to the experiment class.
- `logging`: Specifies which loggers should be used.
- `monty_config`:
  - `monty_class`: `Monty` The type of Monty model to use, e.g. for evidence-based graph matching: `MontyForEvidenceGraphMatching`.
  - `monty_args`: Arguments supplied to the Monty class.
  - `sensor_module_configs`: `Mapping[str:Mapping]`
  - `learning_module_configs`: `Mapping[str:Mapping]`
  - `motor_system_config`: configuration of the motor system and motor policies.
  - `sm_to_agent_dict`: mapping of which sensors connect to which sensor modules.
  - `sm_to_lm_matrix`: mapping of which sensor modules connect to which learning modules.
  - `lm_to_lm_matrix`: hierarchical connectivity between learning modules.
  - `lm_to_lm_vote_matrix`: lateral connectivity between learning modules.
- `env_interface_config`: `dataclass` (specifies environment interface-related args incl. the environment that the interface wraps around (`env_init_func`), arguments for initializing this environment, such as the agent and sensor configuration (`env_init_args`), and optional transformations that occur before information reaches a sensor module. For an example, see `SurfaceViewFinderMountHabitatEnvInterfaceConfig`)
- `train_env_interface_class`: `EnvironmentInterface`
- `train_env_interface_args`: Specifies how the interface should interact with the environment. For instance, which objects should be shown in what episodes and in which orientations and locations.
- `eval_env_interface_class`: `EnvironmentInterface`
- `eval_env_interface_args`: Same purpose as `train_env_interface_args` but allows for presenting Monty with different conditions between training and evaluation.

# Setting up the Experiment Config for Pretraining

To follow along, open the `conf/experiment/tutorial/surf_agent_2obj_train.yaml`. Let's highlight the various aspects of a training experiment configuration.

```yaml
# Basic setup
config:
  logging:
    # Specify directory where an output directory will be created.
    output_dir: ${path.expanduser:"~/tbp/results/monty/projects"}
    # Specify a name for the training run
    run_name: surf_agent_1lm_2obj_train
```

> [!NOTE]
> **Where Logs and Models are Saved**
>
> Loggers have `output_dir` and `run_name` parameters, and since we will use `run.py`, the output will be saved to `OUTPUT_DIR/RUN_NAME`. The `MontySupervisedObjectPretrainingExperiment` suffixes `pretrained`, so the final model will be stored at `OUTPUT_DIR/RUN_NAME/pretrained`, which in our case will be `~/tbp/results/monty/projects/surf_agent_1lm_2obj_train/pretrained`.
>

Next, we specify which objects the model will train on in the dataset, including the rotations in which the objects will be presented. The following code specifies two objects ("mug" and "banana") and 14 unique rotations, which means that both the mug and the banana will be shown 14 times, each time in a different rotation. During each of the overall 28 episodes, the sensors will move over the respective object and collect multiple observations to update the model of the object.

```yaml
config:
  # Get predefined object rotations count that give good views of the object from 14 angles.
  n_train_epochs: ${benchmarks.rotations_all_count}
  train_env_interface_args:
    # Here we specify which objects to learn. 'mug' and 'banana' come from the YCB dataset.
    # If you don't have the YCB dataset, replace with names from habitat (e.g.,
    # 'capsule3DSolid', 'cubeSolid', etc.).
    object_names:
      - mug
      - banana
    object_init_sampler:
      _target_: tbp.monty.frameworks.environments.object_initializers.Predefined
      # Get predefined object rotations that give good views of the object from 14 angles.
      rotations: ${benchmarks.rotations_all}
```

The constant `${benchmarks.rotations_all}` is used in our pretraining and many of our benchmark experiments since the rotations it returns provide a good set of views from all around the object. Its name comes from picturing an imaginary cube surrounding an object. If we look at the object from each of the cube's faces, we get 6 unique views that typically cover most of the object's surface. We can also look at the object from each of the cube's 8 corners which provides an extra set of views that help fill in any gaps. The 14 rotations provided by `${benchmarks.rotations_all}` will rotate the object as if an observer were looking at the object from each of the cube's faces and corners like so:

![learned_models](../../figures/how-to-use-monty/cube_face_and_corner_views_spam.png)

Now we define the entire configuration that specifies one complete Monty experiment:

```yaml
# The configuration for the pretraining experiment.
defaults:
  # We use supervised pretraining config defaults.
  - /experiment/config/supervised_pretraining@config
  # Specify logging config defaults.
  - /experiment/config/logging/pretrain@config.logging
  # Specify the Monty config defaults.
  - /experiment/config/monty/patch_and_view@config.monty_config
  # Clear config.monty_config.monty_args so that next import overrides instead of merging
  - /experiment/config/monty/args/clear_monty_args@config.monty_config
  # Specify config.monty_config.monty_args defaults
  - /experiment/config/monty/args/defaults@config.monty_config.monty_args
  # Clear config.monty_config.learning_module_configs so that overrides override instead of merging
  - /experiment/config/monty/learning_modules/clear_learning_module_configs@config.monty_config
  # Clear config.monty_config.sensor_module_configs so that overrides override instead of merging
  - /experiment/config/monty/sensor_modules/clear_sensor_module_configs@config.monty_config
  # Motor system config specific to surface agent.
  - /experiment/config/monty/motor_system/curvature_informed_surface@config.monty_config.motor_system_config
  # Set up the environment and agent
  - /experiment/config/environment/surface_view_finder_mount_habitat@config.env_interface_config
  # Specify config.train_env_interface_args defaults
  - /experiment/config/environment_interface/per_object@config.train_env_interface_args

# The MontySupervisedObjectPretrainingExperiment class will provide the model
# with object and pose labels for supervised pretraining.
_target_: tbp.monty.frameworks.experiments.pretraining_experiments.MontySupervisedObjectPretrainingExperiment
config:
  n_train_epochs: ${benchmarks.rotations_all_count}
  # Specify logging config overrides.
  logging:
    output_dir: ${path.expanduser:"~/tbp/results/monty/projects"}
    run_name: surf_agent_1lm_2obj_train
    wandb_handlers: []
  # Specify the Monty config overrides.
  monty_config:
    monty_args:
      num_exploratory_steps: 500
    # sensory module configs: one surface patch for training (sensor_module_0),
    # and one view-finder for initializing each episode and logging
    # (sensor_module_1).
    sensor_module_configs:
      sensor_module_0:
        sensor_module_class: ${monty.class:tbp.monty.frameworks.models.sensor_modules.HabitatSM}
        sensor_module_args:
          is_surface_sm: true
          sensor_module_id: patch
          # a list of features that the SM will extract and send to the LM
          features:
            - pose_vectors
            - pose_fully_defined
            - on_object
            - object_coverage
            - rgba
            - hsv
            - min_depth
            - mean_depth
            - principal_curvatures
            - principal_curvatures_log
            - gaussian_curvature
            - mean_curvature
            - gaussian_curvature_sc
            - mean_curvature_sc
          save_raw_obs: false
      sensor_module_1:
        sensor_module_class: ${monty.class:tbp.monty.frameworks.models.sensor_modules.Probe}
        sensor_module_args:
          sensor_module_id: view_finder
          save_raw_obs: false
    # learning module config: 1 graph learning module.
    learning_module_configs:
      learning_module_0:
        learning_module_class: ${monty.class:tbp.monty.frameworks.models.graph_matching.GraphLM}
        learning_module_args: {} # Use default LM args
  train_env_interface_class: ${monty.class:tbp.monty.frameworks.environments.embodied_data.InformedEnvironmentInterface}
  # Specify environment interface overrides
  train_env_interface_args:
    object_names:
      - mug
      - banana
    object_init_sampler:
      _target_: tbp.monty.frameworks.environments.object_initializers.Predefined
      rotations: ${benchmarks.rotations_all}

```

Here, we explicitly specified most parameters in config classes for transparency. The remaining parameters (e.g., `sm_to_lm_matrix`, etc.) aren't supplied since `/experiment/config/monty/patch_and_view`s defaults will work fine here. If you use a different number of SMs or LMs or want a custom connectivity between them, you will have to specify those as well.

Briefly, we specified our experiment class and the number of epochs to run. We also configured a [logger](../logging-and-analysis.md) and a training environment interface to initialize our objects at different orientations for each episode. `monty_config` is a nested config that describes the complete sensorimotor modeling system. Here is a short breakdown of its components:

- `/experiment/config/monty/patch_and_view`: the top-level Monty config object defaults that specify that we will have a sensor patch and an additional viewfinder as inputs to the system. They also specify the routing matrices between sensors, SMs and LMs (using defaults in this simple setup).
  - `monty_args`: a dictionary specifying we want 500 exploratory steps per episode.
  - `sensor_module_configs`: a dictionary specifying sensor module class and arguments. These dictionaries specify that
    - `sensor_module_0` will be a `HabitatSM` with `is_surface_sm=True` (a small sensory patch for a surface agent). The sensor module will extract the given list of features for each patch. We won't save raw observations here since it is memory-intensive and only required for detailed logging/plotting.
    - `sensor_module_1` will be a `Probe` which we can use for logging. We could also store raw observations from the viewfinder for later visualization/analysis if needed. This sensor module is not connected to a learning module and, therefore, is not used for learning. It is called `view_finder` since it helps initialize each episode on the object.
  - `learning_module_configs`: a dictionary specifying the learning module class and arguments. This dictionary specifies that
    - `learning_module_0` will be a `GraphLM` that constructs a graph of the object being explored.
  - `motor_system_config`: a motor system config object that specifies a motor policy class to use. This policy here will move orthogonal to the surface of the object with a preference of following principal curvatures that are sensed. When doing pretraining with the distant agent, the `/experiment/config/monty/motor_system/naive_scan_spiral` policy is recommended since it ensures even coverage of the object from the available view point.

To get an idea of what each sensor module sees and the information passed on to the learning module, check out the documentation on [observations, transforms, and sensor modules](../../how-monty-works/observations-transforms-sensor-modules.md). To learn more about how learning modules construct object graphs from sensor output, refer to the [graph building](../../how-monty-works/learning-module/object-models.md#graph-building) documentation.

# Running the Pretraining Experiment

To run this experiment, call the `run.py` script with the experiment name as the `experiment` argument.
```shell
python run.py experiment=tutorial/surf_agent_2obj_train
```

This will take a few minutes to complete and then you can inspect and visualize the learned models. To do so, create a script and paste in the following code. The location and name of the script is unimportant, but we called it `pretraining_tutorial_analysis.py` and placed it outside of the repository at `~/monty_scripts`.

```python
import os
from pathlib import Path

import matplotlib.pyplot as plt

from tbp.monty.frameworks.utils.logging_utils import load_stats
from tbp.monty.frameworks.utils.plot_utils_dev import plot_graph

# Specify where pretraining data is stored.
exp_path = Path("~/tbp/results/monty/projects/surf_agent_1lm_2obj_train").expanduser()
pretrained_dict = exp_path / "pretrained"

train_stats, eval_stats, detailed_stats, lm_models = load_stats(
    exp_path,
    load_train=False,  # doesn't load train csv
    load_eval=False,  # doesn't try to load eval csv
    load_detailed=False,  # doesn't load detailed json output
    load_models=True,  # loads models
    pretrained_dict=pretrained_dict,
)

# Visualize the mug graph from the pretrained graphs loaded above from
# pretrained_dict. Replace 'mug' with 'banana' to plot the banana graph.
plot_graph(lm_models["pretrained"][0]["mug"]["patch"], rotation=120)
plt.show()
```
Replace `"mug"` with `"banana"` in the second to last line to visualize the banana's graph. After running the script, you should see a graph of the mug/banana.

![learned_models](../../figures/how-to-use-monty/pretraining_tutorial_mug_banana.png)

See [logging and analysis](../logging-and-analysis.md) for more detailed information about experiment logs and how to work with them. You can now move on to [part two](running-inference-with-a-pretrained-model.md) of this tutorial where we load our pretrained model and use it for inference.
