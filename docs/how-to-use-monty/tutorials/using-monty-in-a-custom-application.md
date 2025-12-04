---
title: Using Monty in a Custom Application
---

> ⚠️ This is an Advanced Tutorial
> If you've arrived at this page and you're relatively new to Monty, then we would recommend you start by reading some of our other documentation first. Once you're comfortable with the core concepts of Monty, then we think you'll enjoy learning about how to apply it to custom applications in the following tutorial!

# Introduction
Monty aims to implement a **general-purpose algorithm for understanding and interacting with the world**. It was designed to be very modular so that the same Monty configuration can be tested in many different environments and various Monty configurations can be compared in the same environment. Up to now, the tutorials have demonstrated Monty in a simulated environment (HabitatSim) where a sensor explores 3D objects and recognizes their ID and pose. Here, we will show you how to use Monty in other environments.

## What Kind of Applications can Monty Be Used For?
**Monty is a sensorimotor modeling system. It is NOT made for learning from static datasets** (although some can be framed to introduce movement, such as the Omniglot example below). Any application where you want to use Monty should have some concept of movement and how movement will change the state of the agent and what is being observed.

> ⚠️ Monty Currently Expects Movement to be in 3D Euclidean Space
In the current implementation, movement should happen in 3D (or less) space and be tracked using Euclidean location coordinates. Although we are convinced that the basic principles of Monty will also apply to abstract spaces ([potentially embedded in 3D space](https://thousandbrains.discourse.group/t/abstract-concept-in-monty/533/4)) and we know that the [brain uses different mechanisms to encode space](https://youtu.be/zRRazfFstvY), the current implementation relies on 3D Euclidean space.

# Customizing Monty for Your Application
The diagram below shows the base abstract classes in Monty. For general information on how to customize those classes, see our guide on [Customizing Monty](../customizing-monty.md). The Experiment class coordinates the experiment (learning and evaluation). It initializes and controls Monty and the environment and coordinates the interaction between them. The `EmbodiedEnvironment` class is wrapped in an `EnvironmentInterface` subclass, which exposes methods to interact with the environment. An experiment can have two environment interfaces associated with it: one for training and one for evaluation.

![Class structure in tbp.monty. Each class can be customized independently, allowing for easy modification and testing of individual components.](../../figures/how-to-use-monty/monty_class_diagram.png#width=500px)

Information flow in Monty implements a sensorimotor loop. Observations from the environment are first processed by the sensor module. The resulting CMP-compliant output is then used by the learning modules to model and recognize what it is sensing. The learning modules can suggest an action (`GoalState`) to the motor system at each step. The motor system decides which action to execute and translates it into motor commands. The environment interface then uses this action to extract the next observation from the environment. The next observation is sent to the sensor module(s) and the loop repeats.

![Information flow in Monty. Note, this is a simplified view as there can also be model-free policies that bypass the learning module.](../../figures/how-to-use-monty/monty_information_flow_simplified.png)

Additionally, the `EnvironmentInterface` and Environment can implement specific functions to be executed at different points in the experiment, such as resetting the agent position and showing a new object or scene at the beginning of a new episode.

To use Monty in a custom environment, you usually need to customize the `EnvironmentInterface` and `EmbodiedEnvironment` classes. For example, if you look back at the previous tutorials, you will see that for those Habitat experiments, we've been using the `EnvironmentInterfacePerObject` and the `HabitatEnvironment`. The diagram below shows some key elements that need to be defined for these two classes. It's best to start thinking about the environment setup first, as this will force you to think through how to structure your application correctly for Monty to tackle.
![Key elements to define for a custom environment interface](../../figures/how-to-use-monty/defining_env_and_env_interface.png)

### EmbodiedEnvironment
The first thing to figure out is how movement should be defined in your environment. What actions are possible, and how do these actions change the agent's state and observations?

If you are working with an existing environment, such as one used for reinforcement learning (for example, the Habitat environment we are using), you might just need to wrap this into the `.step()` function of your custom `EmbodiedEnvironment` class such that when `env.step(actions)` is called, an observation is returned. If you work with an application that isn't already set up like that, defining how actions lead to the next observation may be more involved. You can look at the `OmniglotEnvironmentInterface` or `SaccadeOnImageEnvironmentInterface` as examples (more details below).

The observations should be returned as `Observations` with one entry per agent in the environment. Each agent should have `SensorObservations` for each of its sensors. For example, if there is one agent with two sensors that each sense two types of modalities, it would look like this:

```python
obs = Observations(
  {
    AgentID("agent_id_0"): AgentObservations(
      {
        SensorID("patch_0"): SensorObservations(
          {
            Modality("depth"): depth_sensed_by_patch_0,
            Modality("rgba"): rgba_sensed_by_patch_0
          }
        ),
        SensorID("patch_1"): SensorObservations(
          {
            Modality("depth"): depth_sensed_by_patch_1,
            Modality("semantic"): semantic_sensed_by_patch_1
          }
        )
      }
    )
  }
)
```

Related to defining how actions change observations, you will also need to define how actions change the state of the agent. This is what the `get_state()` method returns. The returned state needs to be a `ProprioceptiveState(Dict[str, AgentState])` with an entry per agent in the environment. The entry should contain the agent's position and orientation relative to some global reference point (`AgentState`). For each sensor associated with that agent, a `SensorState` should contain the sensor's position and orientation relative to the agent.

For example, if you have one agent with two sensors, the state dictionary could look like this:
```python
state = ProprioceptiveState(
  {
    AgentID("agent_id_0"): AgentState(
      {
        "position": current_agent_position,
        "rotation": current_agent_rotation,
        "sensors":
        {
          SensorID("patch_0.depth"): SensorState(
            {
              "position": current_depth_sensor_position
              "rotation": current_depth_sensor_rotation
            }
          ),
          SensorID("patch_0.rgba"): SensorState(
            {
              "position": current_rgba_sensor_position
              "rotation": current_rgba_sensor_rotation
            }
          ),
        }
      }
    )
  }
)
```

Lastly, you need to define what happens when the environment is initialized (`__init__()`), when it is reset (`reset()`, usually at the end of an episode), and when it is closed (`close()`, at the end of an experiment). Resetting could include loading a new scene, resetting the agent position, or changing the arrangement of objects in the environment. It might also reset some of the environment's internal variables, such as step counters. Note that, as customary for RL environments, the `reset()` method is also expected to return `Observations`.

### Environment Interface
The `EnvironmentInterface` manages retrieving observations from the `EmbodiedEnvironment` given actions. The EmbodiedEnvironment, in turn, applies basic transforms to the raw observations from the environment.

The `EnvironmentInterface` should define all the key events at which the environment needs to be accessed or modified. This includes initializing the environment (`__init__()`), retrieving the next observation (`__next__()`), and things that happen at the beginning or end of episodes and epochs (`pre_episode()`, `post_episode()`, `pre_epoch()`, `post_epoch()`). Note that not all of those are relevant to every application.

Think about how your experiment should be structured. What defines an episode? What happens with the environment at the beginning or end of each episode? What happens at the beginning or end of epochs? Does anything need to happen at every step besides retrieving the observation and environment state?

## Example 1: Omniglot
As one of Monty's strengths is the ability to learn from small amounts of data, one interesting application is the [Omniglot dataset](https://github.com/brendenlake/omniglot). It contains drawings of 1623 characters from 50 alphabets. Each of the characters is drawn 20 times by different people, as shown below.
![Example character from the Omniglot dataset in its 20 variations.](../../figures/how-to-use-monty/omniglot_character_exp.png)

Since this is a static dataset, and Monty is a sensorimotor learning system, we first have to define how movement looks on this dataset. A sensor module in Monty always receives a small patch as input and the learning module then integrates the extracted features and locations over time to learn and infer complete objects. So, in this case, we can take a small patch on the character (as shown on the right in the figure below) and move this patch further along the strokes at each step. Following the strokes is easy in this case as the Omniglot dataset also contains the temporal sequence of x,y, and z coordinates in which the characters were drawn (the example image above is colored by the order in which the strokes were drawn but also within each stroke we have access to the temporal sequence in which it was drawn). If this information were unavailable, the patch could be moved arbitrarily or use heuristics such as following the sensed principal curvature directions.
![An observation at each step is a small patch on the character.](../../figures/how-to-use-monty/omniglot_obs_exp.png#width=400px)

At each step, the sensor module will extract a location and pose in a common reference frame and send it to the learning module. To define the pose at each location, we extract a [surface normal and two principal curvature directions](https://thousandbrainsproject.readme.io/docs/observations-transforms-sensor-modules#surface-normals-and-principle-curvatures) from a gaussian smoothed image of the patch. As you can see in the images below, the surface normal will always point straight out of the image (as this is a 2D image, not a 3D object surface) and the first principal curvature direction aligns with the stroke direction while the second one is orthogonal to it. The learning module then stores those relative locations and orientations in the model of the respective character and can use them to recognize a character during inference.
![The learned models store poses at locations relative to each other. Pose is defined by surface normal and curvature directions.](../../figures/how-to-use-monty/omniglot_model_exp.png#width=600px)

Learning and inference on Omniglot characters can be implemented by writing two custom classes, the [OmniglotEnvironment](https://github.com/thousandbrainsproject/tbp.monty/blob/f9acc1f013c716f8d2c126ec8fd789ceb4595ac0/src/tbp/monty/frameworks/environments/two_d_data.py#L40) and the [OmniglotEnvironmentInterface](https://github.com/thousandbrainsproject/tbp.monty/blob/f9acc1f013c716f8d2c126ec8fd789ceb4595ac0/src/tbp/monty/frameworks/environments/embodied_data.py#L790):
1. `OmniglotEnvironment`:
   - Defines initialization of all basic variables in the `__init__(patch_size, data_path)` method.
   - In this example, we define the action space as `None` because we give Monty no choice in how to move. The step method just returns the next observation by following the predefined stroke order in the dataset. Note this will still be formulated as a sensorimotor task, as the retrieval of the next observation corresponds to a (pre-defined) movement and we get a relative displacement of the sensor.
   - Defines the `step(actions)` method, which uses the current `step_num` in the episode to determine where we are in the stroke sequence and extracts a patch around that location. It then returns a Gaussian smoothed version of this patch as the observation.
   - Defines `get_state()`, which returns the current x, y, z location on the character as a state dict (z is always zero since we are in 2D space here).
   - Defines `reset()` to reset the `step_num` counter and return the first observation on a new character.
   - Helper methods such as
     - `switch_to_object` and `load_new_character_data` to load a new character
     - `get_image_patch(img, loc, patch_size)` to extract the patch around a given pixel location
     - `motor_to_locations` to convert the movement information from the Omniglot dataset into locations (pixel indices) on the character image
2. `OmniglotEnvironmentInterface`:
   - Defines initialization of basic variables such as episode and epoch counters in the `__init__` method.
   - Defines the `post_episode` method, which calls `cycle_object` to call the environment's `switch_to_object` method. Using the episode and epoch counters, it keeps track of which character needs to be shown next.

![Custom classes for character recognition on the Omniglot dataset](../../figures/how-to-use-monty/omniglot_custom_classes.png#width=500px)

An experiment config for training on the Omniglot dataset can then look like this:
```yaml
defaults:
  - /experiment/config/supervised_pretraining@config
  - /experiment/config/logging/pretrain@config.logging
  - /experiment/config/monty/patch_and_view@config.monty_config
  - /experiment/config/monty/motor_system/clear_motor_system_config@config.monty_config
  - /experiment/config/monty/motor_system/informed_no_trans_step_s1@config.monty_config.motor_system_config
  - /experiment/config/monty/sensor_modules/clear_sensor_module_configs@config.monty_config
  - /experiment/config/environment/omniglot@config.env_interface_config
  - /experiment/config/environment_interface/omniglot@config.train_env_interface_args

_target_: tbp.monty.frameworks.experiments.pretraining_experiments.MontySupervisedObjectPretrainingExperiment
config:
  n_train_epochs: 1
  logging:
    output_dir: ${path.expanduser:${oc.env:MONTY_MODELS}/my_trained_models}
    run_name: omniglot_training
  monty_config:
    sensor_module_configs:
      sensor_module_0:
        sensor_module_class: ${monty.class:tbp.monty.frameworks.models.sensor_modules.HabitatSM}
        sensor_module_args:
          sensor_module_id: patch
          features:
            - pose_vectors
            - pose_fully_defined
            - on_object
            - principal_curvatures_log
          save_raw_obs: false
          # Need to set this lower since curvature is generally lower
          pc1_is_pc2_threshold: 1
      sensor_module_1:
        sensor_module_class: ${monty.class:tbp.monty.frameworks.models.sensor_modules.Probe}
        sensor_module_args:
          sensor_module_id: view_finder
          save_raw_obs: false
  train_env_interface_class: ${monty.class:tbp.monty.frameworks.environments.embodied_data.OmniglotEnvironmentInterface}
  train_env_interface_args:
    # Train on the first version of each character (there are 20 drawings for each
    # character in each alphabet, here we see one of them). The default
    # OmniglotEnvironmentInterfaceArgs specify alphabets = [0, 0, 0, 1, 1, 1] and
    # characters = [1, 2, 3, 1, 2, 3]) so in the first episode we will see version 1
    # of character 1 in alphabet 0, in the next episode version 1 of character 2 in
    # alphabet 0, and so on.
    versions: [1, 1, 1, 1, 1, 1]
```

And a config for inference on those trained models could look like this:
```yaml
defaults:
  - /experiment/config/defaults@config
  - /experiment/config/logging/defaults@config.logging
  - /experiment/config/monty/patch_and_view@config.monty_config
  - /experiment/config/monty/learning_modules/clear_learning_module_configs@config.monty_config
  - /experiment/config/monty/sensor_modules/clear_sensor_module_configs@config.monty_config
  - /experiment/config/environment/omniglot@config.env_interface_config
  - /experiment/config/environment_interface/omniglot@config.eval_env_interface_args

_target_: tbp.monty.frameworks.experiments.object_recognition_experiments.MontyObjectRecognitionExperiment
config:
  model_name_or_path: ${path.expanduser:${oc.env:MONTY_MODELS}/my_trained_models/omniglot_training/pretrained/}
  do_train: false
  n_eval_epochs: 1
  logging:
    run_name: omniglot_inference
  monty_config:
    monty_class: ${monty.class:tbp.monty.frameworks.models.evidence_matching.model.MontyForEvidenceGraphMatching}
    learning_module_configs:
      learning_module_0:
        learning_module_class: ${monty.class:tbp.monty.frameworks.models.evidence_matching.learning_module.EvidenceGraphLM}
        learning_module_args:
          # xyz values are in larger range so need to increase mmd
          max_match_distance: 5
          tolerances:
            patch:
              principal_curvatures_log: ${np.ones:2}
              pose_vectors: ${np.array:[45, 45, 45]}
          # Surface normal always points up, so they are not useful
          feature_weights:
            patch:
              pose_vectors: [0, 1, 0]
          hypotheses_updater_args:
            # We assume the letter is presented upright
            initial_possible_poses: [[0, 0, 0]]
    sensor_module_configs:
      sensor_module_0:
        sensor_module_class: ${monty.class:tbp.monty.frameworks.models.sensor_modules.HabitatSM}
        sensor_module_args:
          sensor_module_id: patch
          features:
            - pose_vectors
            - pose_fully_defined
            - on_object
            - principal_curvatures_log
          save_raw_obs: false
          # Need to set this lower since curvature is generally lower
          pc1_is_pc2_threshold: 1
      sensor_module_1:
        sensor_module_class: ${monty.class:tbp.monty.frameworks.models.sensor_modules.Probe}
        sensor_module_args:
          sensor_module_id: view_finder
          save_raw_obs: false
  eval_env_interface_class: ${monty.class:tbp.monty.frameworks.environments.embodied_data.OmniglotEnvironmentInterface}
  eval_env_interface_args:
    # Using version 1 means testing on the same version of the character as trained.
    # Version 2 is a new drawing of the previously seen characters. In this small test
    # setting these are 3 characters from 2 alphabets.
    versions: [1, 1, 1, 1, 1, 1]
```

> 📘 Follow Along
> To run the above experiment, you first need to download the [Omniglot dataset](https://github.com/brendenlake/omniglot). You can do this by running `cd ~/tbp/data` and `git clone https://github.com/brendenlake/omniglot.git`. You will need to unzip the `omniglot/python/images_background.zip` and `omniglot/python/strokes_background.zip` files.

The above configurations are already included in Monty at `conf/experiment/tutorial/omniglot_training.yaml` and `conf/experiment/tutorial/omniglot_inference.yaml`.

You can run training by calling `python run.py experiment=tutorial/omniglot_training` and then run inference on these models by calling `python run.py experiment=tutorial/omniglot_inference`. You can check the `eval_stats.csv` file in `~/tbp/results/monty/projects/monty_runs/omniglot_inference/` to see how Monty did. If you copied the code above, it should have recognized all six characters correctly.

> ❗️ Generalization Performance on Omniglot is Bad Without Hierarchy
> Note that we currently don't get good generalization performance on the Omniglot dataset. If you use `eval_env_interface_args.versions: [2, 2, 2, 2, 2, 2]` in the inference config, which shows previously unseen versions of the characters, you will see that performance degrades a lot. This is because the Omniglot characters are fundamentally compositional objects (strokes relative to each other), and compositional objects can only be modeled by stacking two learning modules hierarchically. The above configs do not do this. Our research team is hard at work getting Monty to model compositional objects.

## Example 2: Monty Meets World
Monty Meets World is the code name for our first demo of Monty on real-world data. For a video of this momentous moment (or is that Montymentous?), see our [project showcase page](https://thousandbrainsproject.readme.io/docs/project-showcase#monty-for-object-detection-with-the-ipad-camera).

In this application we test Monty's object recognition skills on 2.5D images, which means a photograph that includes depth information (RGBD). In this case, the pictures are taken with the iPad's TrueDepth camera (the user-facing camera used for face recognition).

In this use case, we assume that Monty has already learned 3D models of the objects, and we just test its inference capabilities. For training, we scanned a set of real-world objects using photogrammetry, providing us with 3D models of the objects. You can find instructions to download this `numenta_lab` dataset [here](https://thousandbrainsproject.readme.io/docs/benchmark-experiments#monty-meets-world). We then render those 3D models in Habitat and learn them by moving a sensor patch over them, just as we do with the YCB dataset. We train Monty in the 3D simulator because in the 2D image setup, Monty has no way of moving around the object and, therefore, would have a hard time learning complete 3D models.

![Dataset: The `numenta_lab` dataset is a collection of 12 real-world objects (left). They are scanned and turned into 3D models using photogrammetry.](../../figures/how-to-use-monty/MMW_dataset.png)
![Training: We move a sensor patch over the 3D model using the Habitat simulator.](../../figures/how-to-use-monty/patchon3dmug.gif#width=500px)

To run this pre-training yourself, you can use the [only_surf_agent_training_numenta_lab_obj](https://github.com/thousandbrainsproject/tbp.monty/blob/db6b0404e3c3fa9c95688db5db33fa58053fcb8d/conf/experiment/only_surf_agent_training_numenta_lab_obj.yaml) config. Alternatively, you can download the pre-trained models using the [benchmark experiment instructions](https://thousandbrainsproject.readme.io/docs/benchmark-experiments#monty-meets-world).

For inference, we use the RGBD images taken with the iPad camera. Movement is defined as a small patch on the image moving up, down, left, and right. At the beginning of an episode, the depth image is converted into a 3D point cloud with one point per pixel. The sensor's location at every step is then determined by looking up the current center pixel location in that 3D point cloud. Each episode presents Monty with one image, and Monty takes as many steps as needed to make a confident classification of the object and its pose.

![Inference: We move a patch over an RGBD image to recognize the object and it's pose.](../../figures/how-to-use-monty/patchon2dimage.gif#width=500px)

This can be implemented using two custom classes the [SaccadeOnImageEnvironment](https://github.com/thousandbrainsproject/tbp.monty/blob/4bc857580ae6ac015586af1a61b3e292a7827b6f/src/tbp/monty/frameworks/environments/two_d_data.py#L258) and [SaccadeOnImageEnvironmentInterface](https://github.com/thousandbrainsproject/tbp.monty/blob/db6b0404e3c3fa9c95688db5db33fa58053fcb8d/src/tbp/monty/frameworks/environments/embodied_data.py#L898):
1. `SaccadeOnImageEnvironment`:
   - Defines initialization of all basic variables in the `__init__(patch_size, data_path)` method.
   - Defines the `step(actions)` method, which uses the sensor's current location, the given actions, and their amounts to determine the new location on the image and extract a patch. It updates `self.current_loc` and returns the sensor patch observations as a dictionary.
   - Defines `get_state()`, which returns the current state as a dictionary. The dictionary mostly contains `self.current_loc` and placeholders for the orientation, as the sensor and agent orientation never change.
   - Helper methods such as
     - `switch_to_object(scene_id, scene_version_id)` to load a new image
     - `get_3d_scene_point_cloud` to extract a 3D point cloud from the depth image
     - `get_next_loc(action_name, amount)` to determine valid next locations in pixel space
     - `get_3d_coordinates_from_pixel_indices(pixel_ids)` to get the 3D location from a pixel index
     - `get_image_patch(loc)` to extract a patch at a location in the image.
  	These methods are all used internally within the `__init__`, `step`, and `get_state` methods (except for the `switch_to_object` method, which is called by the `SaccadeOnImageEnvironmentInterface`).
2. `SaccadeOnImageEnvironmentInterface`:
   - Defines initialization of basic variables such as episode and epoch counters in the `__init__` method.
   - Defines the `post_episode` method, which calls `cycle_object` to call the environment's `switch_to_object` method. Using the episode and epoch counters, it keeps track of which image needs to be shown next.


![Custom classes for object recognition in RGBD images](../../figures/how-to-use-monty/MMW_custom_classes.png#width=500px)

An experiment config can then look like this:
```yaml
defaults:
  - /experiment/config/eval@config
  - /experiment/config/logging/parallel_evidence_lm@config.logging
  - /experiment/config/monty/patch_and_view@config.monty_config
  - /experiment/config/monty/learning_modules/clear_learning_module_configs@config.monty_config
  - /experiment/config/monty/learning_modules/default_evidence_1lm@config.monty_config.learning_module_configs
  - /experiment/config/monty/args/clear_monty_args@config.monty_config
  - /experiment/config/monty/args/defaults@config.monty_config.monty_args
  - /experiment/config/monty/motor_system/clear_motor_system_config@config.monty_config
  # move 20 pixels at a time
  - /experiment/config/monty/motor_system/informed_no_trans_step_s20@config.monty_config.motor_system_config
  - /experiment/config/environment/world_image@config.env_interface_config
  - /experiment/config/environment/init_args/clear_env_init_args@config.env_interface_config
  - /experiment/config/environment/init_args/monty_world_standard_scenes@config.env_interface_config.env_init_args
  - /experiment/config/environment_interface/clear_eval_env_interface_args@config
  - /experiment/config/environment_interface/world_image@config.eval_env_interface_args

_target_: tbp.monty.frameworks.experiments.object_recognition_experiments.MontyObjectRecognitionExperiment
config:
  model_name_or_path: ${path.expanduser:${benchmarks.pretrained_dir}/surf_agent_1lm_numenta_lab_obj/pretrained/}
  n_eval_epochs: 1
  show_sensor_output: true
  logging:
    run_name: monty_meets_world_2dimage_inference
    wandb_group: benchmark_experiments
  monty_config:
    monty_args:
      min_eval_steps: ${benchmarks.min_eval_steps}
  eval_env_interface_class: ${monty.class:tbp.monty.frameworks.environments.embodied_data.SaccadeOnImageEnvironmentInterface}
  eval_env_interface_args:
    # list(np.repeat(range(12), 4))
    scenes: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11]
    # list(np.tile(range(4), 12))
    versions: [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]

```
For more configs to test on different subsets of the Monty Meets World dataset (such as bright or dark images, hand intrusion, and multiple objects), you can find the RGBD image benchmark configs at:
- [world_image_from_stream_on_scanned_model](../../../conf/experiment/world_image_from_stream_on_scanned_model.yaml)
- [world_image_on_scanned_model](../../../conf/experiment/world_image_on_scanned_model.yaml)
- [dark_world_image_on_scanned_model](../../../conf/experiment/dark_world_image_on_scanned_model.yaml)
- [bright_world_image_on_scanned_model](../../../conf/experiment/bright_world_image_on_scanned_model.yaml)
- [hand_intrusion_world_image_on_scanned_model](../../../conf/experiment/hand_intrusion_world_image_on_scanned_model.yaml)
- [multi_object_world_image_on_scanned_model](../../../conf/experiment/multi_object_world_image_on_scanned_model.yaml)

> 📘 Follow Along
> To run this experiment, you first need to download our 2D image dataset called `worldimages`. You can find instructions for this [here](https://thousandbrainsproject.readme.io/docs/benchmark-experiments#monty-meets-world).
>
> You will also need to [download the pre-trained models](https://thousandbrainsproject.readme.io/docs/getting-started#42-download-pretrained-models). Alternatively, you can run pre-training yourself by running `python run.py experiment=only_surf_agent_training_numenta_lab_obj`. Running pre-training requires the Habitat simulator and [downloading the `numenta_lab` 3D mesh dataset](https://thousandbrainsproject.readme.io/docs/benchmark-experiments#monty-meets-world).

To run the experiment, call `python run.py experiment=tutorial/monty_meets_world_2dimage_inference`. If you don't want to log to wandb, add ` wandb_handlers: []` to `config.logging`. If you just want to run a quick test on a few of the images, adjust the `scenes` and `versions` parameters in `config.eval_env_interface_args`.

# Other Things You May Need to Customize
If your application uses sensors different from our commonly used cameras and depth sensors, or you want to extract specific features from your sensory input, you will need to define a custom sensor module. The sensor module receives the raw observations from the environment interface and converts them into the CMP, which contains features at poses. For more details on converting raw observations into the CMP, see our [documentation on sensor modules](https://thousandbrainsproject.readme.io/docs/observations-transforms-sensor-modules).

If your application requires a specific policy to move through the environment or you have a complex actuator to control, you might want to implement a custom `MotorSystem` or `MotorPolicy` class. For more details on our existing motor system and policies, see our [documentation on Monty's policies](https://thousandbrainsproject.readme.io/docs/policy).

Writing those custom classes works the same way as it does for the `EnvironmentInterface` and `EmbodiedEnvironment` classes. For general information, see our documentation on [customizing Monty](https://thousandbrainsproject.readme.io/docs/customizing-monty).

# Conclusion
This tutorial was a bit more text than practical code. This is because every application is different, and we try to convey the general principles here. The first step for any application is to think about if and how the task can be phrased as a sensorimotor environment. What is Monty's action space? How is movement defined? How does it change observations? How do movement and sensation determine the sensor's location and orientation in space? This will then help you figure out how to define a custom `EmbodiedEnvironment` and `EnvironmentInterface` and their associated `__init__`, `step`, `get_state`,`reset`, `pre_episode`, and `post_episode` methods. If you run into issues customizing Monty to your application, please come over to our [Discourse Forum](https://thousandbrains.discourse.group/) and ask for help!
