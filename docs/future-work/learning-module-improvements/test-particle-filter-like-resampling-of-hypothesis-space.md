---
title: Test Particle-Filter-Like Resampling of Hypothesis Space
---

In order to make better use of the available computational resources, we might begin by sampling a "coarse" subset of possible hypotheses across objects at the start of an episode. As the episode progresses, we could re-sample regions that have high probability, in order to explore hypotheses there in finer detail. This would serve the purpose of enabling us to have broad hypotheses initially, without unacceptably large computational costs. At the same time, we could still develop a refined hypothesis of the location and pose of the object, given the additional sampling of high-probability regions.

Furthermore, when the evidence values for a point in an LM's graph falls below a certain threshold, we generally stop testing it. Related to this, the initial feature pose detected when the object was first sensed determines the pose hypotheses that are initialized. We could therefore implement a method to randomly initialize a subset of rejected hypotheses, and then test these. This relates to [Less Dependency on First Observation](less-dependency-on-first-observation.md).

This work could also tie in with the ability to [Use Better Priors for Hypothesis Initialization](../learning-module-improvements/use-better-hypothesis-priors.md), as these common poses could be resampled more frequently.
