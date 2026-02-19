---
title: Less Dependency on First Observation
description: Currently we only initialize hypotheses on the first step of an episode which make us dependent on the first observation having low noise .
rfc: 
estimated-scope: large
improved-metric: noise, multi-object
output-type: prototype, monty-feature, PR, analysis
skills: python, research, monty
contributor: ramyamounir
status: completed
hidden: true
---

In the current implementation, Monty initializes its entire hypothesis space during the first step of an episode.
This design makes the system highly dependent on the quality of that initial observation.
Any sensor noise, occlusion, or partial view influences all hypotheses generated for the remainder of the episode.
The model also becomes tightly anchored to the pose sensed from this single observation, which limits its ability to adapt when Monty moves onto a different objects, e.g., during compositional model inference, or when the pose of the object changes over time, e.g., during a behavior.

A more flexible approach is to remove the special sampling procedure of the first step entirely.
Hypotheses should be initialized, expanded, or replaced at any point during episode, based on evidence scores rather than episode boundaries.

> [!NOTE]
> This has been implemented as part of the `BurstSamplingHypothesesUpdater`. Burst sampling removes the special first-step initialization entirely; instead, hypotheses are sampled through bursts triggered by evidence slopes at any point during inference. This means the system is no longer anchored to a single initial observation. See the [burst sampling](../../how-monty-works/learning-module/evidence-based-learning-module.md#burst-sampling) documentation for details.
