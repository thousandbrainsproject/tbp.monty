---
title: Motor System
---

Monty's motor system is contained within the a `MotorSystem` object. The motor system's job is to generate `Action` objects to be consumed by a simulator, robot, etc.


The motor system's primary component is a `MotorPolicySelector` which itself contains one or more `MotorPolicy` objects. The `MotorPolicySelector` is responsible for choosing the policy given any information it has available, which includes goals, proprioceptive information, raw observations, and CMP-compliant data. If goals are not provided, the `MotorPolicySelector` will still select an appropriate default policy. `MotorPolicy` can be considered the "terminal" object in the sense that they are the only objects within the motor system that produce actions. In contrast, the policy selector mediates action-generation by trafficking data in and out of the policies it chooses.

Currently, we have defined two motor policy selectors:
  - `SinglePolicySelector`: This is the trivial selector. Configured with a single policy, it merely moves data in and out of its sole policy. This is most often used during training (e.g., with the `NaiveSpiralScan` motor policy) or with the surface agent configurations.
  - `DistantPolicySelector`: This selector has three policies it can choose from. If the selector has a goal that was sent from a learning module, it will choose the `JumpToGoal` policy which repositions the agent and sensors in the environment so that Monty will be able to observe the target location. If there are no LM-derived goals but there are goals from an SM (e.g. `SalienceSM`), then this selector will saccade to the target location with the `LookAtGoal` policy. When there are no goals present, it will default to the `InformedPolicyRandomWalk`, which merely moves around at random.

For more detailed information on how our policies work, see [Policy](motor-system/policy.md).
