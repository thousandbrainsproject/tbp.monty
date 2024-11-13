---
title: Running Your First Experiment
---
Let's try to set up a first experiment config for a simplified toy experiment and run it. We will then walk step-by-step through what happened.

> 🚧 TODO: Maybe make this one of the benchmark experiments or make the analysis based on this config?
>
> At least make it a graph matching demo (maybe just 1-2 objects).

# Setting up the Experiment Config

To go along, copy this code into a file (for example called `first_experiment.py`). Save this file in the `benchmarks/configs/` folder.

```python
from tbp.monty.frameworks.config_utils.config_args import (
    SingleCameraMontyConfig,
    LoggingConfig
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    ExperimentArgs,
    SinglePTZHabitatDatasetArgs,
    get_env_dataloader_per_object_by_idx,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import MontyExperiment

#####
# To test your env and familiarize with the code, we'll run the simplest possible
# experiment. We'll use a model with a single learning module as specified in
# monty_config. We'll also skip evaluation, train for a single epoch for a single step,
# and only train on a single object, as specified in experiment_args and train_dataloader_args.
#####

first_experiment = dict(
    experiment_class=MontyExperiment,
    logging_config=LoggingConfig(),
    experiment_args=ExperimentArgs(
        do_eval=False,
        max_train_steps=1,
        n_train_epochs=1,
    ),
    monty_config=SingleCameraMontyConfig(),
    # Data{set, loader} config
    dataset_class=ED.EnvironmentDataset,
    dataset_args=SinglePTZHabitatDatasetArgs(),
    train_dataloader_class=ED.EnvironmentDataLoaderPerObject,
    train_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=1),
    eval_dataloader_class=ED.EnvironmentDataLoaderPerObject,
    eval_dataloader_args=get_env_dataloader_per_object_by_idx(start=0, stop=1),
)


CONFIGS = dict(
    first_experiment=first_experiment,
)
```

Next you will need to add the following lines to the `benchmarks/configs/__init__.py` file:

```python
from .first_experiment import CONFIGS as FIRST_EXPERIMENT

# Put this line after CONFIGS is initialized
CONFIGS.update(FIRST_EXPERIMENT)
```

# Running the Experiment

To run this experiment you just defined, you can now simply navigate to the `benchmarks/` folder and call the `run.py` script with the experiment name as the `-e` argument.

```shell
cd benchmarks
python run.py -e first_experiment
```

# What Just Happened?

Now that you have run your first experiment, let's unpack what happened. This first section involves a lot of text, but rest assured, once you grok this first experiment, the rest of the tutorial will be much more interactive and will focus on running expeirments and using tooling. This first experiment is virtually the simplest one possible, but it is designed to familiarize you with all the pieces and parts of the experimental workflow to give you a good foundation for further experimentation.

Experiments are implemented as Python classes with methods like `train` and `evaluate`. In essence, `run.py` loads a config, and calls `train` and `evaluate` methods if the config says to run them. **Notice that `first_experiment` has `do_eval` set to `False`, so `run.py` will only run the `train` method.**

## Experiment Structure: Epochs, Episodes, and Steps

One epoch will run training (or evaluation) on all the specified objects.  An epoch generally consists of multiple episodes, one for each object, or for each pose of an object in the environment. An episode is one training or evaluating session with one single object. This episode consists of a sequence of steps. What happens in a step depends on the particular experiment, but an example would be: shifting the agent's position, reading sensor inputs, transforming sensor inputs to features, and adding these features to an object model. For more details on this default experiment setup see [this section from the Monty documentation](../../how-monty-works/experiment.md).

If you examine the `MontyExperiment` class, you will also notice that there are related methods like `{pre,post}_epoch`, and `{pre,post}_episode`. **With inheritance or mixin classes, you can use these methods to customize what happens before during and after each epoch, or episode. **Also notice that each method contains calls to a logger. Logger classes can also be customized to log specific information at each control point. Finally, we save a model with the `save_state_dict` method at the end of each epoch. All told, the sequence of method calls goes something like

- train (loop over epochs)
  - pre_train logging
  - run_epoch (loop over episodes)
    - pre_epoch
      - pre_epoch logging
    - run_episode (loop over steps)
      - pre_episode
        - pre_episode logging
      - model.step()
      - post_episode
        - post_episode logging
    - post_epoch
      - save_state_dict
      - post_epoch logging
  - post_train logging

and **this is exactly the procedure that was executed when you ran `python run.py -e first_experiment`.**

> 🚧 TODO: Make a nicer looking version of this
>
> Make this look more "cody" instead of bullet points.

## Model

The model is specified in the `monty_config` field of the `first_experiment` config, as `SingleCameraMontyConfig` which is in turn defined within `src/tbp/monty/frameworks/config_utils/config_args.py`. Yes, that's a config within a config. The reason for nesting configs is that the model is an ensemble of LearningModules (LMs), and SensorModules (SMs), each of which could potentially have their own configuration as well. For more details on configuring custom learning or sensor modules see [this guide](../customizing-monty.md).

For now, we will start with the simplest vesion of this complex system. The `SingleCameraMontyConfig` dataclass has fields `learning_module_configs` and `sensor_module_configs` where each key is the name of an LM (or SM resp.), and each value is the full config for that model component. **Our first model has only one LM and one SM**. Note that the `sm_to_agent_dict` field of the model config maps each SM to an "agent" (i.e. a moveable part), and only a single agent is specified, meaning that our model has one moveable part with one sensor attached to it. In particular, it has an RGBD camera attached to it.

## Steps

By now, we know that an experiment relies on `train` and `evaluate` methods, that each of these runs one or more `epochs`, which consists of one or more `episodes`, and finally each `episode` repeatedly calls `model.step`. Now we will start unpacking each of these levels, starting with the innermost loop over `steps`.

In `SingleCameraMontyConfig`, notice that the model class is specified as `MontyBase` (`src/tbp/monty/frameworks/models/monty_base`), which is a subclass of an abstract class defined in `src/tbp/monty/frameworks/models/abstract_monty_classes`. In here you will see that there are two template methods for two types of steps: `_exploratory_step` and `_matching_step`. In turn, each of these steps is defined as a sequence of calls to other abstract methods, including `_set_step_type_and_check_if_done`, which is a point at which the step type can be switched. The conceptual difference between these types of steps is that **during exploratory steps, no inference is attempted**, which means no voting and no keeping track of which objects or poses are possible matches to the current observation. Each time `model.step` is called in the experimental procedure listed under the "Episodes and Epochs" heading, either `_exploratory_step` or `_matching_step` will be called. In a typical experiment, training consists of running `_matching_step` until a) an object is recognized b) all known objets are ruled out, or c) a step counter exceeds a threshold. Regardless of how matching-steps is terminated, the system then switches to running exploratory step so as to gather more observations and build a more complete model of an object.

You can, of course, customize step types and when to switch between step types by defininig subclasses or mixins. To set the initial step type, use `model.pre_episode`. To adjust when and how to switch step types, use `_set_step_type_and_check_if_done`.

**In this particular experiment, `n_train_epochs` was set to 1, and `max_train_steps` was set to 1. This means a single epoch was run, with one matching step per episode**. In the next section, we go up a level from the model step to understand episodes and epochs.

## Data{set, loader}

> 🚧 TODO: Update this Section after Refactor
>
> We are planning to refactor dataset and dataloader to become one class and have a name that expresses the iteractiveness more.

In the config for first_experiment, there is a comment that marks the start of data configuration. Now we turn our attention to everything below that line, as this is where episode specifics are defined.

The term dataset is used loosely here; **the dataset class in this experiment is technically a whole simulation environment**. The objects within an environment are assumed to be the same for both training and evaluation (for now), hence only one (class, args) pairing is needed. Note however that object orientations, as well as specific observations obtained from an object, will generally differ across training and evaluation. The term dataset is used in keeping with the traditional ML meaning; it is the thing you sample from in order to train a model.

So, if the data**set** is an environment, what is a data**loader**? Again, in keeping with the PyTorch use of the term, the dataloader is basically the API between the dataset and the model. Its job is to sample from the dataset and return observations to the model. Note that the next observation is decided by the last action, and the action is selected by a `motor_system`. This motor system is shared by reference with the model. By changing the action, the **model** controls what it observes next just as you would expect from an embodied agent.

 Now, finally answering our question of what happens in an episode, notice that our config uses a special type of dataloader: `EnvironmentDataLoaderPerObject` (note that this is a subclass of `EnvironmentDataLoader` which is kept as general as possible to allow for flexible subclass customization). As indicated in the docstring, this dataloader has a list of objects, and at the beginning / end of an episode, it removes the current object from the environment, increments a (cyclical) counter that determines which object is next, and places the new object in the environment. The arguments to `EnvironmentDataLoaderPerObject` determine which objects are added to the environment and in what pose. **In our config, we use a single list with one YCB object**.

## Final Notes on the Model

To wrap up this tutorial, we'll cover a few more details of the model. Recall that `sm_to_agent_dict` assigns each SM to a moveable part (i.e. an "agent"). The action space for each moveable part is in turn defined in the `motor_system_config` part of the model config. Once an action is executed, the agent moves, and each sensor attached to that agent (here just a single RGBD sensor) receives an observation. Just as `sm_to_agent_dict` specifies which sensors are attached to which agents, in `src/tbp/monty/frameworks/config_utils/config_args` the `MontyConfig` field `sm_to_lm_matrix` specifies for each LM which SMs it will receive observations from. Thus, observations flow from agents to sensors (SMs), and from SMs to LMs, where all actual modeling takes place in the LM. Near the end of `model.step` (remember, this can be either `matching_step` or `exploratory_step`), the model calls `decide_location_for_movement` which selects an action and closes the loop between the model and the environment. Finally, at the end of each epoch, we save a model in a directory specified by the `ExperimentArgs` field of the model config.

# Summary

That was a lot of text, so let's review what all went into this experiment.

- We ran a `MontyExperiment` using `run.py`
- We went through the `train` procedure with one epoch
- The epoch looped over a list of objects of length 1 - so a single episode was run
- The max steps was set to 1, so all told, we took one single step on one single object
- Our model had a single agent with a single RGBD camera attached to it
- During model.step, `matching_step` was called and one SM received one observation from the environment
- The `decide_location_for_movement` method was called
- We saved our model at the end of the epoch

Congratulations on completing your first experiment! Ready to take the next step? Learn the ins-and-outs of [pretraining a model](./pretraining-a-model.md).
