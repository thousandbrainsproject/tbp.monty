---
title: Implement Switching Between Learning and Inference-Focused Policies
---

Currently, a Monty system cannot flexibly switch between a learning-focused policy (such as the naive scan policy) and an inference-focused policy. Enabling LMs to guide such a switch based on their internal models, and whether they are in a matching or exploration state, would be a useful improvement.

This would be a specific example of a more general mechanism for switching between different policies, as discussed in [Switching Policies via Goal States](interpret-goal-states-in-motor-system-switch-policies.md).

Similarly, an LM should be able to determine the most appropriate *model-based* policies to initialize, such as the hypothesis-testing policy vs. a [top-down exploration policy](top-down-exploration-policy.md).