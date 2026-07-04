---
title: Improve Terminology Including Directory and Experiment Names
description: There are several instances where terminology in the code could be clarified, including in the directories organizing tbp.monty, and our experiment configs.
rfc: required
estimated-scope: medium
improved-metric: learning-experience
output-type: PR
skills: monty
contributor: 
status: evergreen
---

There are several instances where the terminology in `tbp.monty` could be improved for clarity. Exising confusion often relates to the progressive development of the platform, where clearer terminology has emerged in our research meetings over time, but is not yet reflected in the code.

Note that the below are examples alongside potential proposals. However, the aim of an RFC would be to clarify the best approach to any terminology change.

### Policies, Goals, and Actions
One concrete example is how we refer to goals, actions, and policies. Currently we have "Goal Generators" in SMs and LMs, and "Policies" in the Motor System. We also have "Goals" that are passed to the Motor System, but in our research meetings, we now refer to Goal States vs. Target Poses. This latter distinction has helped clarify when something is about changing a state, vs. specifying a spatial target. A potential resolution might be to remove the term "Generator", and use something like:
- Goal-State Policies:
    - Exist in LMs.
    - Produce Goal States, which only go to other LMs, and specify the desired *state* of an object.
- Target-Pose Policies:
    - Can exist in SMs or LMs; as such, the one form of policy that can be either model-free (if within an SM) or model-based (if within an LM).
    - The equivalent of the current Goal Generators (hypothesis testing policy and SM salience policy).
    - Pass a Target Pose to the Motor System (i.e. specify a location and orientation, but not an object state).
- Action Policies
    - The outputs of the Motor System in a simulator or other embodiment.
    - Specify low-level motor primitives.

### "Models"
The term model is highly overloaded in Monty - it can refer to Monty as a whole, as well as the models that a Learning Module in Monty learns. In addition, the primary directory where most of the key Monty code exists is `frameworks/models`, which could be interpreted in multiple ways. This is particularly confusing for someone coming across Monty for the first time.

Formalizing the notion that models refer to the representations learned within an LM would be consistent with other terminology we use widely, like "model-based vs model-free" policies, and ["out of model movements"](../learning-module-improvements/use-out-of-model-movements.md). This would also discourage us from referring to "training a Monty model" (in reference to a Monty system), which is language more appropriate for a classical machine learning perspective, but not for the longer-term vision of Monty as a continually learning system.

After making this change, the directories for Monty could likely be improved, including removing `frameworks/models`. A follow-on task could be to reorganize and break up how the different components of Monty are specified within this directory.

### Log Outputs and Experiment Names

Experiment results are logged to directories with complex names. For example

```
/user_name/tbp/results/monty/projects/evidence_eval_runs/logs/your_experiment
```

could likely be simplified to

```
/user_name/tbp/results/monty/your_experiment
```

(the longer directories we currently use relate to a time when multiple, parallel visions of Monty were being worked on at Numenta)

Similarly, many of our experiments have inconsistent names. Focusing on our pretraining experiments, only some of these specify "supervised" (even though many are supervised), and some specify the agent used ("only_surf...") while others do not ("supervised_pre_training_base"). This relates to [making our configs easier to use](./make-configs-easier-to-use.md).

