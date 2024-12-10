---
title: Test Particle-Filter-Like Resampling of Hypothesis Space
---

When the evidence values for a point in an LM's graph falls below a certain threshold, we generally stop testing it. Furthermore, the initial feature pose detected when the object was first sensed determines the pose hypotheses that are initialized.

We would like to implement a method to randomly initialize a subset of rejected hypotheses, and then test these. This relates to [Less Dependency on First Observation](less-dependency-on-first-observation.md).

This work could also tie in with the ability to [Use Better Priors for Hypothesis Initialization](../learning-module-improvements/use-better-hypothesis-priors.md), as these common poses could be resampled more frequently.
