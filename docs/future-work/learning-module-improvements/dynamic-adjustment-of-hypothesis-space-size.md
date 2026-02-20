---
title: Dynamic Adjustment of Hypothesis Space Size
description: Dynamically delete and resample hypotheses throughout the course of an episode.
rfc: https://github.com/thousandbrainsproject/tbp.monty/blob/main/rfcs/0013_resampling_dynamic_adjustment.md
estimated-scope: large
improved-metric: speed, pose, accuracy, noise, multi-object
output-type: prototype, PR, monty-feature, publication
skills: python, research, monty
contributor: ramyamounir
status: completed
---

The current Monty system initializes a fixed hypothesis space at the start of each episode and maintains it until the episode completes.
While this approach simplifies hypotheses updating, it introduces several constraints that limit adaptability and scalability.
Most importantly, it ties inference to episode boundaries and prevents Monty from adapting its hypothesis space as new observations are collected.

There are several important limitations in the current `DefaultHypothesesUpdater` design:

- **Fixed hypothesis space for the full episode.**
A fixed hypothesis space implies a supervised notion of an episode's start and end.
Monty initializes hypotheses at the beginning and deletes them at the end, which prevents true continuous inference.
Ideally, Monty should operate without the concept of episodes, able to explore, move across objects, switch contexts, and update its hypotheses in an online unsupervised manner.

- **Dependence on the first observation.**
Monty relies heavily on the initial observation to sample its hypotheses.
As a result, the entire hypothesis space is anchored to the morphological features present in that first observation.
This makes it difficult for Monty to handle changes in object identity within a compositional model, or to accommodate pose changes that happen during interactions and behaviors.
Sensor noise in the first observation can also affect all the generated hypotheses from this single step.
More context for this future work item is provided [here](https://thousandbrainsproject.readme.io/docs/less-dependency-on-first-observation)

- **Inefficiency from maintaining unlikely hypotheses.**
Since all hypotheses remain stored throughout the episode, even unlikely hypotheses require some computation and memory.
This overhead becomes significant as Monty learns about more objects.
Some optimizations exist, such as updating only the top hypotheses based on confidence thresholds (i.e., `x_percent_threshold`), but the overall hypothesis space remains fixed and cannot shrink or grow.

The goal of this future work item is to replace the fixed hypothesis space with a dynamic hypothesis space.
The system should be able to identify and delete hypotheses that fail to accumulate evidence, while also sampling new hypotheses that are informed by more recent observations, when needed.
This would enable a shift toward continuous, episode-free inference where the system can navigate and understand its environment in a more natural way.

> [!NOTE]
> This feature has been implemented as the `BurstSamplingHypothesesUpdater`. See the [burst sampling](../../how-monty-works/learning-module/evidence-based-learning-module.md#burst-sampling) documentation for details.
