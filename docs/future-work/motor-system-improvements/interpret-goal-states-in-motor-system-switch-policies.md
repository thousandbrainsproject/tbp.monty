---
title: Interpret Goal States in Motor System & Switch Policies
---

We would like to implement a state-switching mechanism where an LM (or multiple LMs) can pass a goal-state to the motor system to switch the model-free policies that it is executing.

For example, we might like to perform a thorough, random walk in a small region if the observations are noisy and we would like to sample them densely. Alternatively, we might like to move quickly across the surface of an object, spending little time in a given region.

This task also relates to [Enable Switching Between Learning and Inference-Focused Policies](./implement-switching-between-learning-and-inference-focused-policies.md).