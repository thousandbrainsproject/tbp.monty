---
title: Implement Switching Between Learning and Inference-Focused Policies
description: Dynamically switch the way goals are selected based on more high level goals and confidence of the system.
rfc: required
estimated-scope: unknown
improved-metric: goal-policy, learning, accuracy, numsteps
output-type: RFC, monty-feature, PR
skills: python, monty, refactoring
contributor: 
status: open
---

> [!NOTE]
> Currently we have policies in several different places (SM goal generator, LM goal generator, goal state selector, motor system) and this item here might relate more to LM goal state generator improvements as opposed to the motor system. This item needs to be scoped more and might be moved in category.

Currently, a Monty system cannot flexibly switch between a learning-focused policy (such as the naive scan policy) and an inference-focused policy. Enabling LMs to guide such a switch based on their internal models, and whether they are in a matching or exploration state, would be a useful improvement.

This would be a specific example of a more general mechanism for switching between different policies, as discussed in [Switching Policies via Goal States](interpret-goal-states-in-motor-system-switch-policies.md).

Similarly, an LM should be able to determine the most appropriate *model-based* policies to initialize, such as the hypothesis-testing policy vs. a [model-based exploration policy](model-based-exploration-policy.md).
