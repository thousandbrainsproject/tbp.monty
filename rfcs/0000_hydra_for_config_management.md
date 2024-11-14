- Start Date: 2024-09-15
- RFC PR: (leave this empty, it will be filled in after RFC is merged)

NOTE: While this RFC process is document-based, you are encouraged to also include visual media to help convey your ideas.

# Summary

> A brief explanation of the feature.

This RFC proposes integrating the [Hydra](https://hydra.cc/docs/intro/) configuration framework into the Monty project to enhance how configurations are managed and utilized. Hydra offers a **dynamic** and **hierarchical** approach to configuration management, allowing for modular configurations and command-line overrides.

# Motivation

> Why are we doing this? What use cases does it support? What is the expected outcome? Which metrics will this improve? What capabilities will it add?
>
> Please focus on explaining the motivation so that if this RFC is not accepted, the motivation could be used to develop alternative solutions. In other words, enumerate the constraints you are trying to solve without coupling them too closely to the solution you have in mind.

Currently, configurations in Monty are somewhat complex for new users and spread across several files (e.g. `frameworks/config_utils/config_args.py` for monty_configs, `frameworks/experiments/experiment_classes/monty_experiment.py` for experiment_configs, etc). This leads to difficulties in maintenance, ensuring consistency, and new contributors may find it challenging to understand the configuration structure, which may discourage contributions.

The expected outputs by utilizing Hydra for configuration management are that it will allow for: (1) easy and systematic adjustments to manage different experiments and (2) minimize errors caused by misconfigurations.

# Guide-level explanation

> Explain the proposal as if it was already included in Monty and you were teaching it to another Monty user. That generally means:
>
> - Introducing new named concepts.
> - Explaining the feature largely in terms of examples.
> - Explaining how Monty developers should *think* about the feature and how it should impact the way they use Monty. It should explain the impact as concretely as possible.
> - If applicable, provide sample error messages, deprecation warnings, or migration guidance.
> - If applicable, describe the differences between teaching this to existing Monty users and new Monty users.
> - If applicable, include pictures or other media if possible to visualize the idea.
> - If applicable, provide pseudo plots (even if hand-drawn) showing the intended impact on performance (e.g., the model converges quicker, accuracy is better, etc.).
> - Discuss how this impacts the ability to read, understand, and maintain Monty code. Code is read and modified far more often than written; will the proposed feature make code easier to maintain?
>
> Keep in mind that it may be appropriate to defer some details to the [Reference-level explanation](#reference-level-explanation) section.
>
> For implementation-oriented RFCs, this section should focus on how developer contributors should think about the change and give examples of its concrete impact. For administrative RFCs, this section should provide an example-driven introduction to the policy and explain its impact in concrete terms.

## Introduction to Hydra in Monty

Monty leverages the [**Hydra** package](https://hydra.cc/) as a configuration framework to manage all experiment configurations. Hydra allows you to compose configurations dynamically, making it easier to manage complex experiments and switch between setups without modifying code. With Hydra, configurations are organized hierarchically and can be easily overridden from the command line or through configuration files.

## Getting Started with Hydra in Monty

### Basic Structure

In Monty, configuration files are stored in the `configs` folder:

```
nupic.monty
|-- docs
|-- src
|-- configs
|   |-- train.yaml
|   |-- eval.yaml
|   |-- experiment
|   |   |-- benchmarks
|   |   |   |-- randrot_10distinctobj_surf_agent.yaml
|   |   |-- monty_capabilities
|   |   |   |-- 20240918_robustness.yaml
|   |-- loggers
|   |   |-- csv.yaml
|   |   |-- many_loggers.yaml
|   |   |-- wandb.yaml
|   |-- monty
|   |   |-- monty_base.yaml
|   |   |-- sensor_modules
|   |   |   |-- distant_sm.yaml
|   |   |   |-- surface_sm.yaml
|   |   |-- learning_modules
|   |   |   |-- base_lm.yaml
|   |   |   |-- graph_lm.yaml
|   |   |   |-- evidence_graph_lm.yaml
|   |   |-- motor_system
|   |   |   |-- base_policy.yaml
|   |   |   |-- informed_policy.yaml
|   |   |   |-- surface_policy.yaml
|   |   |   |-- ...
|   |-- data
|   |   |-- ycb.yaml
|   |-- hooks
|   |   |-- pre_episode_explore.yaml
|   |-- paths
|   |   |-- default.yaml # help set project paths, e.g. MONTY_DIR
|   |-- extras
|   |   |-- default.yaml # set extra utilities, such as whether to print configs, ignore warnings, etc.
```

(**Note**: the above is an example of what it may look like in terms of organization but definitely not comprehensive)

### Writing Configuration Files

Each configuration file defines a set of parameters for a specific component. For example, `distant_sm.yaml` under the Sensor Module directory may look like:

```yaml
# configs/monty/sensor_modules/distant_sm.yaml
_target_: src.nupic.monty.frameworks.models.sensor_modules.HabitatDistantPatchSM
sensor_module_id: 0
features: 
    - "on_object"
    - "rgba"
    - "point_normal"
    - "mean_curvature"
save_raw_obs: false
pc1_is_pc2_threshold: 10
noise_params: null
process_all_obs: false
```

### Composing Configurations

For experiments, we can compose or put together multiple configurations as below:

```yaml
# configs/experiments/example_experiment.yaml
# let's say its an experiment that uses all 77 objects in ycb instead of default of 10 objects
defaults:
    - monty: monty_base
    - data: ycb # default uses 10 objects
    - logger: wandb

# below we can include configs to override default for this particular experiment
data:
  ycb:
    num_target_objects: 77
```

## Overriding Configurations

One of Hydra's powerful features is the ability to override configurations from the command line.

Examples:

```bash
python run.py experiment=my_experiment_1 
python run.py experiment=my_experiment_2 data.ycb.num_target_objects=50 model.sensor_modules.distant_sm.features="on_object"
```

**Note:** Hydra will save all configurations including ones passed by the command line. However, for benchmarks I recommend having a configuration file written rather than passing arguments each time.

## Benefits for New and Existing Users

- **Separation of Concerns**:

Hydra enables us to alter a YAML configuration file in a directory separate from the source code directory (with the additional option of providing a command-line argument override). When someone attempts to use Monty in their project, they will need only focus on creating a YAML configuration file to execute Monty, perhaps without ever opening an IDE.

- **Modular and Flexible Configuration Management**:  
  
Hydra’s compositional design allows for breaking configurations into multiple reusable and modular files. This makes it easier to manage large, complex systems by swapping components without duplicating configuration files.

- **Dynamic Configuration Overriding**:  

Hydra enables dynamic overrides, allowing users to modify configurations via the command line without editing the YAML files directly. This flexibility improves experimentation speed and reduces manual work.

For example, the following command will run the experiment 5 times with different seeds:

```bash
python run.py -m seed=1,2,3,4,5 
```

The `-m` flag allows Hydra to perform a **multirun**, and will execute the script five times with different seeds.

- **Seamless Experimentation**:  

Hydra’s support for configuration and multiple configuration variants simplifies experimentation. Users can quickly experiment with different parameteres without manual reconfiguration.

## Migration Guidance for Existing Users

- Existing configurations will need to be converted into YAML files compatible with Hydra.
- Updating the `run.py` script to use Hydra's initializations.

```python
# run.py
# Copyright (C) 2024 Numenta Inc. All rights reserved.

# The information and source code contained herein is the
# exclusive property of Numenta Inc. No part of this software
# may be used, reproduced, stored or distributed in any form,
# without explicit written authorization from Numenta Inc.

import hydra
from omegaconf import DictConfig

@hydra.main(version_base=1.3, config_path="configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    run_experiment(cfg)

if __name__ == "__main__":
    main()
```

>[!NOTE]
> **What is omegaconf?**
>
> OmegaConf is another package that Hydra builds upon. Hydra uses OmegaConf's DictConfig as its default configuration object.
>
> **What is DictConfig?**
>
> DictConfig is a class from OmegaConf that is a nested dictionary which allows access to config params using both key-based (i.e. `cfg["param"]`) and attribute-based (i.e. `cfg.param`).

## Error Handling and Debugging

Hydra provides clear error messages when configurations are missing or incorrectly specified.

For example, if you try to use a non-existent configuration:

```bash
python run.py model.sensor_modules=unknown_sensor
```

Hydra will output:

```
ValueError: Could not load configuration `unknown_sensor`. Available options are: distant_sm, surface_sm
```

# Reference-level explanation

> This is the technical portion of the RFC. Explain the design in sufficient detail that:
>
> - Its interaction with other features is clear.
> - It is reasonably clear how the feature would be implemented.
> - Corner cases are dissected by example.
>
> The section should return to the examples from the previous section and explain more fully how the detailed proposal makes those examples work.

In this section, I will provide a more detailed technical explanation of how Hydra will be integrated into Monty.

## Overview of the Integration Process

1. **Configuration Structure**: Defining a hierarchical and modular configuration directory. This process will not impact current workflow.
2. **Codebase Modification**: Updating Monty's scripts, primarily those related to **initializations** will need to be updated. The command to initialize an object by Hydra is:

```python
model: Monty = hydra.utils.instantiate(cfg.model)
```

3. **Compatibility**: The most time-consuming step will be ensuring that Monty performance remains the same after using Hydra for configuration. Before implementing Hydra, we should have all benchmark results saved, then compare to these benchmarks after migrating to Hydra.

**Note**: Hydra itself will have little to no impact on performance related to executing code or startup. By "performance", I mean the performance of the experiments - manual refactoring may cause unintentional misconfigurations.

## Configuration Structure

In the Guide-level section, I have shown a simple example of what the configuration would look like for a Sensor Module:

```yaml
# configs/monty/sensor_modules/distant_sm.yaml
_target_: src.nupic.monty.frameworks.models.sensor_modules.HabitatDistantPatchSM
sensor_module_id: 0
features: ["on_object", "rgba", "point_normal", "mean_curvature"]
save_raw_obs: false
pc1_is_pc2_threshold: 10
noise_params: null
process_all_obs: false
```

**Note**: Hydra's `_target_` syntax allows for dynamic instantiation of classes.

A more hierarchical example, such as `monty_base.yaml`, which will have a sensor module(s) and others (e.g. learning_modules, motor_systems) may look like:

```yaml
# configs/monty/base.yaml

defaults:
    - sensor_modules: distant  # Reference to the distant sensor module configuration
    - learning_modules: base  # Reference to the base learning module configuration
    - motor_system: base   # Reference to the base motor policy configuration

monty:
  sensor_modules:
    - ${sensor_modules.sm1}
    - ${sensor_modules.sm2}
  learning_modules:
    - ${learning_module}
  motor_system:
    policy: ${motor_system}

# Define specific parameters for each sensor module instance

sensor_modules:
  sm1:
    _target_: src.nupic.monty.frameworks.models.sensor_modules.HabitatDistantPatchSM
    sensor_module_id: 0
    features:
      - on_object
      - rgba
    save_raw_obs: false
    pc1_is_pc2_threshold: 10
    noise_params: null
    process_all_obs: false

  sm2:
    _target_: src.nupic.monty.frameworks.models.sensor_modules.HabitatDistantPatchSM
    sensor_module_id: 1
    features:
      - point_normal
      - mean_curvature
    save_raw_obs: false
    pc1_is_pc2_threshold: 15
    noise_params: null
    process_all_obs: true
```

## Handling Corner Cases

### Conflict Resolution

When multiple configurations specify the same parameter, Hydra follows a defined precedence:

1. **Command-Line Overrides**: Highest priority.
2. **Configuration Files**: Based on the order specified in the `defaults` list.

Within the same configuration file, the last configuration specified takes precedence over earlier ones.

For example:

```yaml
# config.yaml  
defaults:  
  - config_a  
  - config_b  # This has higher precedence than config_a  

# config_a.yaml  
parameter: "from_a"  

# config_b.yaml  
parameter: "from_b"  
```

Parameter will be set to `from_b`.

3. **Default Values in Code**: Lowest priority.

## Dynamic Parameters

For parameters that depend on runtime information or computations, Hydra supports variables and evaluating using the `${...}` syntax.

Example:

```yaml
parameter: ${oc.env:ENV_VARIABLE, default_value}
```

Here, `oc` stands for OmegaConf; this uses OmegaConf's environment variable resolver, and will fall back on `default_value` if `ENV_VARIABLE` is not set. It is possible to create custom resolvers using Hydra, for more details, please refer to [OmegaConf's Resolver documentation](https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html).

### Interaction with Other Features

Hydra's configuration can be easily integrated with existing logging tools (e.g. Weights & Biases).

### Running an Experiment

```bash
python run.py experiment=my_experiment
```

# Drawbacks

> Why should we *not* do this? Please consider:
>
> - Implementation cost, both in terms of code size and complexity
> - Whether the proposed feature can be implemented outside of Monty
> - The impact on teaching people Monty
> - Integration of this feature with other existing and planned features
> - The cost of migrating existing Monty users (is it a breaking change?)
>
> There are tradeoffs to choosing any path. Please attempt to identify them here.

## Implementation Cost and Complexity

- **Refactoring Effort**: Migrating to Hydra requires substantial changes to the existing codebase. Configuration files need to be written in YAML format, and scripts must be updated to integrate with Hydra's initialization.
- **Learning Curve for Contributors**: Unlike Python's built-in dictionaries, contributors will need to become familiar with Hydra's syntax (e.g. defaults, overriding) and concepts (e.g. composing), which could slow down development initially.

## Whether it can be implemented outside of Monty

Sorry, I'm not exactly sure what this means...maybe this is more for a feature?

## Impact on Teaching and Onboarding

- **Steeper Learning Curve for New Users**: Like other dependencies (e.g. `habitat-sim`), new users will need to learn both Monty and Hydra simultaneously, which may overwhelm beginners.

## Cost of Migrating Existing Monty Users

- Transitioning to Hydra will introduce breaking changes. Users will need to adapt their current configurations to the new system, which will be inconvenient and time-consuming.

## Trade-offs and Considerations

- **Dependency Risks**: Like with other dependencies, relying on an external framework introduces risks related to dependency management (e.g. version conflicts, security vulnerabilities).

# Rationale and alternatives

> - Why is this design the best in the space of possible designs?
> - What other designs have been considered, and what is the rationale for not choosing them?
> - What is the impact of not doing this?

## Why is this design the best in the space of possible designs?

How Hydra stands out in the space of possible designs:

1. **Dynamic and Hierarchical Configuration**: Hydra allows for dynamic configuration from multiple sources, meaning we can easily mix and match different components (e.g. 1 SM + 1 LM for experiment1, 1 SM + 5 LM for experiment2) without modifying the underlying Python code and managing YAML files. In Hydra, configurations are organized hierarchically, which reflects the current hierarchy of Monty Codebase (i.e. `MontyExperiment` --> `Monty` --> `Sensor Modules`, `Learning Modules`, etc.)
2. **Minimizing Misconfigurations**: A centralized and unified configuration system makes it easier to maintain and assist users from supplying invalid configurations.
3. **Rich Feature Set**: Hierarchical, Composable, Dynamic, reduces duplicate codes and boilerplates
4. **Active Development and Support**: Hydra's GitHub is very active.

## What other designs have been considered, and what is the rationale for not choosing them?

1. Other Configuration Frameworks

- Other frameworks include [ConfigArgParse](https://github.com/bw2/ConfigArgParse), [YACS](https://github.com/rbgirshick/yacs), and [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/).
- **ConfigArgParse**: A drop-in replacement for argparse that allows options to also be set via config files and/or environment variables.
- **YACS**: Yet Another Configuration System
- **OmegaConf**: OmegaConf is a YAML based hierarchical configuration system, with support for merging configurations from multiple sources (files, CLI argument, environment variables) providing a consistent API regardless of how the configuration was created.

Rationale for not choosing:

1. ConfigArgParse
   1. Does not support hierarchical configurations
2. YACS (Yet Another Configuration System)
   1. Not being actively developed (last update 4 years ago), Apache 2.0 License (not sure if this will be a problem. Hydra has MIT license.)
3. OmegaConf
   1. Hydra builds on top of OmegaConf. OmegaConf is great for simpler projects, and Hydra is great when we want to pull configs from multiple sources (e.g. keeping separate configs for sensor_modules, learning_modules, etc. and putting them together).

## What is the impact of not doing this?

1. **Persistent Difficulties in Understanding Monty for New Users**: There is a steep learning curve with current configurations which may discourage new contribuations.
2. **Maintenance Difficulties**: Without a centralized and standardized configuration system, maintaining consistency and updating configurations remains cumbersome.
3. **Reduced Experimentation Efficiency**: Adjusting configurations for different experiments would lead to making changes to the code-base across **multiple** files, making it harder to compare or introduce errors.

# Prior art and references

> Discuss prior art, both the good and the bad, in relation to this proposal.
> A few examples of what this can include are:
>
> - References
> - Does this functionality exist in other frameworks, and what experience has their community had?
> - Papers: Are there any published papers or great posts that discuss this? If you have some relevant papers to refer to, this can serve as a more detailed theoretical background.
> - Is this done by some other community and what were their experiences with it?
> - What lessons can we learn from what other communities have done here?
>
> This section is intended to encourage you as an author to think about the lessons from other frameworks and provide readers of your RFC with a fuller picture.
> If there is no prior art, that is fine. Your ideas are interesting to us, whether they are brand new or adaptations from other places.
>
> Note that while precedent set by other frameworks is some motivation, it does not on its own motivate an RFC.
> Please consider that Monty sometimes intentionally diverges from common approaches.

## Use of Hydra in Other Communities

Hydra was developed by FAIR and has been primarily used in AI/ML communities to manage experiments testing various hyperparameters. Personally, I have been introduced to Hydra when I started using [Pytorch Lightning-Hydra Template](https://github.com/ashleve/lightning-hydra-template/tree/main).

# Unresolved questions

> Optional, but suggested for first drafts.
>
> What parts of the design are still TBD?

# Future possibilities

> Optional.
>
> Think about what the natural extension and evolution of your proposal would
> be and how it would affect Monty and the Thousand Brains Project as a whole in a holistic way.
> Try to use this section as a tool to more fully consider all possible
> interactions with the Thousand Brains Project and Monty in your proposal.
> Also consider how this all fits into the future of Monty.
>
> This is also a good place to "dump ideas" if they are out of the scope of the
> RFC you are writing but otherwise related.
>
> If you have tried and cannot think of any future possibilities,
> you may simply state that you cannot think of anything.
>
> Note that having something written down in the future-possibilities section
> is not a reason to accept the current or a future RFC; such notes should be
> in the section on motivation or rationale in this or subsequent RFCs.
> The section merely provides additional information.
