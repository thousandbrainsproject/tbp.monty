---
title: Interpret Goal States in Motor System & Switch Policies
description: Enable the motor system to switch between different ways to achieve goals.
rfc: optional
estimated-scope: unknown
improved-metric: numsteps, goal-policy
output-type: PR, monty-feature
skills: python, monty, refactoring
contributor: tristanls-tbp
status: scoping
---

We would like to implement a state-switching mechanism where the motor system can switch the policy that it is executing, depending on the goal state it receives.

For example, we might want to do a semi-random walk when the motor system is not receiving goal states. We may want to switch to a saccade policy (rotating the sensor without moving it's position) when the sensor module sends a goal based on detected salient features. We may want to switch to a "JumpToGoalState" policy when the motor system receives a goal from the learning module that includes both a desired location and orientation (i.e. move the agent and rotate it)

This task also relates to [Enable Switching Between Learning and Inference-Focused Policies](./implement-switching-between-learning-and-inference-focused-policies.md).