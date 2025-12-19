---
title: Improve Bounded Evidence Performance
description: If we don't allow evidence to grow infinitely, memory fades. Develop mechanisms to deal with this.
rfc: optional
estimated-scope: medium
improved-metric: multi-object
output-type: prototype, experiments, analysis
skills: python, research, monty
contributor: ramyamounir, scottcanoe
status: paused
---

In the current model, evidence grows infinitely as Monty compares each hypothesis’s predictions with the observed features at every matching step.
Hypotheses accumulate evidence proportional to their similarity to the observed features, which produces unbounded scores.
Unbounded accumulation of evidence is undesirable for several reasons.

First, Unbounded accumulation introduces a strong temporal bias. 
Hypotheses that are initialized earlier in an episode have more opportunities to accumulate evidence and can dominate the hypothesis space simply due to their age.
This behavior becomes problematic when new hypotheses are sampled later in time.
Newly sampled hypotheses are disadvantaged and may fail to out-compete existing ones, even if they are more consistent with recent observations.

Second, when resampling of hypotheses is allowed, unbounded evidence values become difficult to interpret and reason about.
As evidence grows arbitrarily large, the absolute magnitude of a score becomes less meaningful, and comparisons depend increasingly on the hypotheses age values rather than consistency with recent observations.
Bounded scores are easier to work with as they allow downstream components to rely on known numerical ranges, simplifying threshold selection and enabling more stable decision rules for deleting and resampling hypotheses.

Bounding evidence through methods such as [exponential moving averages](https://youtu.be/A1cOwvZpgjU?si=1oyk6-BTij6TaRdG&t=1668), weighted averages, or normalization by hypothesis age can keep scores within a bounded range.
However, these techniques compress historical information and cause memory to fade over time.
Monty loses the influence of important past features, such as a mug handle, as a result, recognition can become less accurate in long sequences.

A promising direction is to pair bounded evidence with [saliency driven saccades](https://thousandbrainsproject.readme.io/docs/implement-efficient-saccades-driven-by-model-free-and-model-based-signals) or other attention mechanisms.
If the system consistently revisits the key discriminative features that are important for recognition, and does so efficiently, then those features remain present in recent history.

This is still an open problem. Ideas around this issue are welcome.

