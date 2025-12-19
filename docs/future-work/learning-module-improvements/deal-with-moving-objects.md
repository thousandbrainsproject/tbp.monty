---
title: Deal with Moving Objects
description: Use sensed local flow or predicted object movement to compensate the hypothesized location on object.
rfc: required
estimated-scope: large
improved-metric: dynamic, real-world
output-type: prototype, monty-feature, PR
skills: python, research, monty
contributor: 
status: open
---
> [!NOTE]
> This work relates to first being able to [Detect Local and Global Flow](../../future-work/sensor-module-improvements/detect-local-and-global-flow.md). For an improved version it would also depend on [Modeling Object Behaviors](../../theory/object-behaviors.md). It also involves [setting up an environment with moving objects](../environment-improvements/object-behavior-test-bed.md).

Our current idea is to then use this information to model the state of the object, such that beyond its current pose, we also capture how it is moving as a function of time. This information can then be made available to other learning modules for voting and hierarchical processing.
