---
title: Decompose Goals into Subgoals & Communicate
---

This will be most relevant when we begin implementing policies that change the state of the world, rather than just those that support efficient sensing and inference.

One example task we imagine is setting a dinner table. At the higher level of the system, a learning module that models dinner-tables would receive the goal-state to have the table in the "set for eating" state. This might be a vision-based LM that can use it's direct motor-output to saccade around the scene, and infer whether the table is set, but cannot actually move objects.

It might perceive, for example, that the fork is not in the correct location for the learned model of a set dinner table. As such, it could pass a goal-state to the LM that models forks to be in the required pose in body-centric coordinates. The fork modeling-LM, which has information about the morphology of the fork, could then send goal-states directly to the motor-system, or to an LM that controls actuators like a hand. In either case, the ultimate aim is to apply pressure to the fork such that it achieves the desired goal state of being in the correct location.

To set the entire dinner table, the higher-level LM would send out the sub-goal of the fork in the correct position, before moving on to other components of the table object, such as setting the position of the knife.

In the above example, neither the dinner-table, fork, nor hand LMs have sufficient knowledge to complete the task on their own. Instead, it must be decomposed into a series of sub-goal states.

How exactly we define the goal-states that carry out the practical process of applying pressure to move the fork is still a point of discussion, and so an early implementation might assume that a sub-cortical policy is already known that can move objects around the scene, based on a receive goal-state. Alternatively, we might begin with a simpler task such as pressing a button or key, where the motor policy simply needs to apply force at a specific location.

Actually learning the causal relationships between states in low-level objects and high-level objects is also an aspect we are still developing ideas for. However, we know that these will be formed via hierarchical connections between LMs, similar to the [Top Down Connections Used for Sensory Prediction](../cmp-hierarchy-improvements/add-top-down-connections.md).