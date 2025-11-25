---
title: Running Your First Experiment
---

# Introduction
In this tutorial we will introduce the basic mechanics of Monty experiment configs, how to run them, and what happens during the execution of a Monty experiment. Since we will focus mainly on the execution of an experiment, we'll configure and run the simplest possible experiment and walk through it step-by-step. Please have a look at the next tutorials for more concrete examples of running the code with our current graph learning approach.

> [!NOTE]
> **Don't have the YCB Dataset Downloaded?**
>
> Most of our tutorials require the YCB dataset, including this one. Please follow the instructions on downloading it [here](../getting-started.md#41-download-the-ycb-dataset).
>

# Setting up the Experiment Config

> [!NOTE]
>
> Below instructions assume you'll be running an experiment within the checked out `tbp.monty` repository. This is the recommended way to start. Once you are familiar with Monty, if you'd rather setup your experiment in your own repository, then take a look at [Running An Experiment From A Different Repository](./running-an-experiment-from-a-different-repository.md).

Monty uses [Hydra](https://hydra.cc/) for configuration. The `conf` directory serves as the root configuration directory, and experiment configs are located under `conf/experiment`. The experiment config for this tutorial is located at `conf/experiment/tutorial/first_experiment.yaml` which contains the following:

```yaml
# To test your env and help you familiarize yourself with the code, we'll run the simplest possible
# experiment. We'll use a model with a single learning module as specified in
# monty_config. We'll also skip evaluation, train for a single epoch for a single step,
# and only train on a single object, as specified in config and train_env_interface_args.

# The order of the defaults matters. The defaults are applied in order, from top to bottom.
defaults:
  # We use the pre-defined /experiment/config/supervised_pretraining values for the config (@config)
  - /experiment/config/supervised_pretraining@config
  # We use the pre-defined /experiment/config/logging/pretrain values for the logging (@config.logging)
  - /experiment/config/logging/pretrain@config.logging
  # We use the pre-defined /experiment/config/monty/patch_and_view values for the monty_config (@config.monty_config)
  - /experiment/config/monty/patch_and_view@config.monty_config
  # Set up the environment and agent.
  # We use the pre-defined /experiment/config/environment/patch_view_finder_mount_habitat values for
  # the env_interface_config (@config.env_interface_config)
  - /experiment/config/environment/patch_view_finder_mount_habitat@config.env_interface_config
  # We use the pre-defined /experiment/config/environment_interface/per_object values for
  # the train_env_interface_args (@config.train_env_interface_args)
  - /experiment/config/environment_interface/per_object@config.train_env_interface_args

# The top-level _target_ is the experiment we want to run.
_target_: tbp.monty.frameworks.experiments.pretraining_experiments.MontySupervisedObjectPretrainingExperiment
# We override some of the config values we got from the pre-defined /experiment/config/supervised_pretraining.
config:
  do_eval: false
  max_train_steps: 1
  n_train_epochs: 1
  # We override the run_name in config.logging to be specific to our experiment.
  logging:
    run_name: first_experiment
  train_env_interface_class: ${monty.class:tbp.monty.frameworks.environments.embodied_data.EnvironmentInterfacePerObject}
  # We override the train_env_interface_args to be specific to our experiment.
  train_env_interface_args:
    object_names:
      - mug
    # We use the pre-defined PredefinedObjectInitializer to initialize the object.
    object_init_sampler:
      # Since this is a _target_, Hydra will instantiate this class using the _target_ and any
      # additional arguments you provide.
      _target_: tbp.monty.frameworks.config_utils.make_env_interface_configs.PredefinedObjectInitializer
```

# Running the Experiment

Monty experiments are run with `run.py` (or `run_parallel.py`) located directly under `tbp.monty`. To run the experiment defined above, `cd` into `tbp.monty`, make sure the `tbp.monty` environment has been activated (via `conda activate tbp.monty`), and enter

```shell
python run.py experiment=tutorial/first_experiment
```

The `experiment` argument is determined by the location of the experiment config, relative to the `conf/experiment` directory. This experiment is named `tutorial/first_experiment` since the config is located at `conf/experiment/tutorial/first_experiment.yaml`.

# What Just Happened?

Now that you have run your first experiment, let's unpack what happened. This first section involves a lot of text, but rest assured, once you grok this first experiment, the rest of the tutorials will be much more interactive and will focus on running experiments and using tooling. This first experiment is virtually the simplest one possible, but it is designed to familiarize you with all the pieces and parts of the experimental workflow to give you a good foundation for further experimentation.

Experiments are implemented as Python classes with methods like `train` and `evaluate`. In essence, `run.py` loads a config and calls `train` and `evaluate` methods if the config says to run them. **Notice that `first_experiment` has `do_eval` set to `false`, so `run.py` will only run the `train` method.**

## Experiment Structure: Epochs, Episodes, and Steps

One epoch will run training (or evaluation) on all the specified objects.  An epoch generally consists of multiple episodes, one for each object, or for each pose of an object in the environment. An episode is one training or evaluating session with one single object. This episode consists of a sequence of steps. What happens in a step depends on the particular experiment, but an example would be: shifting the agent's position, reading sensor inputs, transforming sensor inputs to features, and adding these features to an object model. For more details on this default experiment setup see [this section from the Monty documentation](../../how-monty-works/experiment.md).

If you examine the `MontyExperiment` class, the parent class of `MontySupervisedObjectPretrainingExperiment`, you will notice that there are related methods like `{pre,post}_epoch`, and `{pre,post}_episode`. **With inheritance or mixin classes, you can use these methods to customize what happens before during and after each epoch, or episode.** For example, `MontySupervisedObjectPretrainingExperiment` reimplements `pre_episode` and `post_epoch` to provide extra functionality specific to pretraining experiments. Also notice that each method contains calls to a logger. Logger classes can also be customized to log specific information at each control point. Finally, we save a model with the `save_state_dict` method at the end of each epoch. All told, the sequence of method calls goes something like

- `MontyExperiment.train` (loops over epochs)
  - Do pre-train logging.
  - `MontyExperiment.run_epoch` (loops over episodes)
    - `MontyExperiment.pre_epoch`
      - Do pre-epoch logging.
    - `MontyExperiment.run_episode` (loops over steps)
      - `MontyExperiment.pre_episode`
        - Do pre-episode logging.
      - `Monty.step`
      - `MontyExperiment.post_episode`
        - Update object model in memory.
        - Do post-episode logging
    - `MontyExperiment.post_epoch`
      - `MontyExperiment.save_state_dict`.
      - Do post-epoch logging.
  - Do post-train logging.

and **this is exactly the procedure that was executed when you ran `python run.py experiment=tutorial/first_experiment`.** (Please note that we're writing `MontyExperiment` in the above sequence rather than `MontySupervisedObjectPretrainingExperiment` for the sake of generality). When we run Monty in evaluation mode, the same sequence of calls is initiated by `MontyExperiment.evaluate` minus the model updating step in `MontyExperiment.post_episode`. See [here](../../how-monty-works/experiment.md) for more details on epochs, episodes, and steps.

## Model

The model is specified in the `config.monty_config` field of the `first_experiment` pre-defined in `/experiment/config/monty/patch_and_view`. Yes, that's a config within a config. The reason for nesting configs is that the model is an ensemble of LearningModules (LMs), and SensorModules (SMs), each of which could potentially have their own configuration as well. For more details on configuring custom learning or sensor modules see [this guide](../customizing-monty.md).

For now, we will start with one of the simpler and most common versions of this complex system. The `/experiment/config/monty/patch_and_view` configuration has fields `learning_module_configs` and `sensor_module_configs` where each key is the name of an LM (or SM resp.), and each value is the full config for that model component. **Our first model has one LM and two SMs**. Why two SMs and only 1 LM? One SM provides the LM with processed observations, while the second SM serves as our experimental probe and is used solely to initialize the agent at the beginning of the experiment.

Note that the `sm_to_agent_dict` field of the model config maps each SM to an "agent" (i.e. a moveable part), and only a single agent is specified, meaning that our model has one moveable part with one sensor attached to it. In particular, it has an RGBD camera attached to it. The mapping is specified using the `monty.agent_id` resolver: `${monty.agent_id:agent_id_0}` which ensures that we pass the `AgentID` type (specifically `AgentID("agent_id_0")`) to the dictionary.

## Steps

By now, we know that an experiment relies on `train` and `evaluate` methods, that each of these runs one or more `epochs`, which consists of one or more `episodes`, and finally each `episode` repeatedly calls `model.step`. Now we will start unpacking each of these levels, starting with the innermost loop over `steps`.

In `/experiment/config/monty/patch_and_view`, notice that the model class is specified as `${monty.class:tbp.monty.frameworks.models.graph_matching.MontyForGraphMatching}` (it uses the `monty.class` resolver that passes the Python class itself to the config), which is a subclass of `tbp.monty.frameworks.models.monty_base.MontyBase`, which in turn is a subclass of `tbp.monty.frameworks.models.abstract_monty_classes.Monty`. In the abstract base class `Monty`, you will see that there are two template methods for two types of steps: `_exploratory_step` and `_matching_step`. In turn, each of these steps is defined as a sequence of calls to other abstract methods, including `_set_step_type_and_check_if_done`, which is a point at which the step type can be switched. The conceptual difference between these types of steps is that **during exploratory steps, no inference is attempted**, which means no voting and no keeping track of which objects or poses are possible matches to the current observation. Each time `model.step` is called in the experimental procedure listed under the "Episodes and Epochs" heading, either `_exploratory_step` or `_matching_step` will be called. In a typical experiment, training consists of running `_matching_step` until a) an object is recognized, or b) all known objects are ruled out, or c) a step counter exceeds a threshold. Regardless of how matching-steps is terminated, the system then switches to running exploratory step so as to gather more observations and build a more complete model of an object.

You can, of course, customize step types and when to switch between step types by defining subclasses or mixins. To set the initial step type, use `model.pre_episode`. To adjust when and how to switch step types, use `_set_step_type_and_check_if_done`.

**In this particular experiment, `n_train_epochs` was set to 1, and `max_train_steps` was set to 1. This means a single epoch was run, with one matching step per episode**. In the next section, we go up a level from the model step to understand episodes and epochs.

## Environment Interface

In the config for first_experiment, there is a comment that marks the start of environment and agent setup. Now we turn our attention to everything below that line, as this is where episode specifics are defined.

The environment interface class is the way we interact with a simulation environment. The objects within an environment are assumed to be the same for both training and evaluation (for now), hence only one (class, args) pairing is needed. Note however that object orientations, as well as specific observations obtained from an object, will generally differ across training and evaluation.

The environment interface is basically the API between the environment and the model. Its job is to sample from the environment and return observations to the model (+initialize and reset the environment). Note that the next observation is decided by the last action, and the actions are selected by a `motor_system`. This motor system is shared by reference with the model. By changing the actions, the **model** controls what it observes next, just as you would expect from a sensorimotor system.

Now, finally answering our question of what happens in an episode, notice that our config uses a special type of environment interface: `EnvironmentInterfacePerObject` (note that this is a subclass of `EnvironmentInterface` which is kept as general as possible to allow for flexible subclass customization). As indicated in the docstring, this environment interface has a list of objects, and at the beginning / end of an episode, it removes the current object from the environment, increments a (cyclical) counter that determines which object is next, and places the new object in the environment. The arguments to `EnvironmentInterfacePerObject` determine which objects are added to the environment and in what pose. **In our config, we use a single list with one YCB object, a mug.**

## Final Notes on the Model

To wrap up this tutorial, we'll cover a few more details of the model. Recall that `sm_to_agent_dict` assigns each SM to a moveable part (i.e. an "agent"). The action space for each moveable part is in turn defined in the `motor_system_config` part of the model config. Once an action is executed, the agent moves, and each sensor attached to that agent (here just a single RGBD sensor) receives an observation. Just as `sm_to_agent_dict` specifies which sensors are attached to which agents, in `/experiment/config/monty/patch_and_view` the field `sm_to_lm_matrix` specifies for each LM which SMs it will receive observations from. Thus, observations flow from agents to sensors (SMs), and from SMs to learning modules (LMs), where all actual modeling takes place in the LM. Near the end of `model.step` (remember, this can be either `matching_step` or `exploratory_step`), the model selects actions and closes the loop between the model and the environment. Finally, at the end of each epoch, we save a model in a directory specified by the `config.logging.output_dir` configuration field (pre-defined in `/experiment/config/logging/defaults` in our config).

# Summary

That was a lot of text, so let's review what all went into this experiment.

- We ran a `MontyExperiment` using `run.py`
- We went through the `train` procedure with one epoch
- The epoch looped over a list of objects of length 1 - so a single episode was run
- The max steps was set to 1, so all told, we took one single step on one single object
- Our model had a single agent with a single RGBD camera attached to it
- During `model.step`, `matching_step` was called and one SM received one observation from the environment
- The model selected the next action
- We saved our model at the end of the epoch

Congratulations on completing your first experiment! Ready to take the next step? Learn the ins-and-outs of [pretraining a model](./pretraining-a-model.md).
