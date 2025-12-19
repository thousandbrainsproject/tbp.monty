---
title: Test Particle-Filter-Like Resampling of Hypothesis Space
description: Resample new hypotheses dynamically throughout an experiment.
rfc: https://github.com/thousandbrainsproject/tbp.monty/blob/main/rfcs/0009_hypotheses_resampling.md
estimated-scope: large
improved-metric: accuracy, speed, multi-object
output-type: prototype, experiments, analysis, PR
skills: python, monty, research
contributor: ramyamounir
status: completed
---

> [!NOTE]
> This feature has been completed and can be tested by using the `ResamplingHypothesesUpdater` (instead of the `DefaultHypothesesUpdater`) as done in the [unsupervised inference experiments](../../overview/benchmark-experiments.md#unsupervised-inference).
> Followup items are:
> - [Dynamic adjustment of hypothesis space size](../learning-module-improvements/dynamic-adjustment-of-hypothesis-space-size.md)
> - [Sample offspring hypotheses](../learning-module-improvements/sample-offspring-hypotheses.md)
> - [Re-anchor hypotheses for robustness to noise and distortions](../learning-module-improvements/re-anchor-hypotheses.md)
> - [Less dependency on first observation](../learning-module-improvements/less-dependency-on-first-observation.md)
> If you have experience with ideas around this (especially related to SLAM), feel free to reach out!

In order to make better use of the available computational resources, we might begin by sampling a "coarse" subset of possible hypotheses across objects at the start of an episode. As the episode progresses, we could re-sample regions that have high probability, in order to explore hypotheses there in finer detail. This would serve the purpose of enabling us to have broad hypotheses initially, without unacceptably large computational costs. At the same time, we could still develop a refined hypothesis of the location and pose of the object, given the additional sampling of high-probability regions.

Furthermore, when the evidence values for a point in an LM's graph falls below a certain threshold, we generally stop testing it. Related to this, the initial feature pose detected when the object was first sensed determines the pose hypotheses that are initialized. We could therefore implement a method to randomly initialize a subset of rejected hypotheses, and then test these. This relates to [Less Dependency on First Observation](less-dependency-on-first-observation.md).

This work could also tie in with the ability to [Use Better Priors for Hypothesis Initialization](../learning-module-improvements/use-better-priors-for-hypothesis-initialization.md), as these common poses could be resampled more frequently.
