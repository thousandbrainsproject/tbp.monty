---
title: Test Particle-Filter-Like Resampling of Hypothesis Space
---

When the evidence values for a point in an LM's graph falls below a certain threshold, we generally stop testing it. Furthermore, depending on the initial feature pose detected when the object was first sensed, this will determine the pose hypotheses that are initialized.

We would like to implement a method to randomly initialize a subset of rejected hypotheses, and then test these. This relates to [Less Dependency on First Observation](less-dependency-on-first-observation.md).

This work could also tie in with TODO making use of frequently observed poses to bias the initialization of new hypotheses.