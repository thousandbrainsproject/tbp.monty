---
title: Improve Bounded Evidence Performance
description: If we don't allow evidence to grow infinitely, memory fades. Develop mechanisms to deal with this, including with more neuron-like integration properties.
rfc: optional
estimated-scope: medium
improved-metric: multi-object
output-type: prototype, experiments, analysis
skills: python, research, monty
contributor: ramyamounir, scottcanoe
status: paused
---

In the current evidence-based learning module, evidence grows infinitely as Monty compares each hypothesis’s predictions with the observed features at every matching step.
Hypotheses accumulate evidence proportional to their similarity to the observed features, which produces unbounded scores.
Unbounded accumulation of evidence is undesirable for several reasons.

## The Challenges and Expected Improvements

### A Temporal Bias Limits Improved Hypotheses

First, unbounded accumulation introduces a strong temporal bias. 
Hypotheses that are initialized earlier in an episode have more opportunities to accumulate evidence and can dominate the hypothesis space simply due to their age.
This behavior becomes problematic when new hypotheses are sampled later in time.
Newly sampled hypotheses are disadvantaged and may fail to out-compete existing ones, even if they are more consistent with recent observations.
A concrete outcome that we could measure for is that the most likely hypothesis (MLH) that Monty infers approaches the "theoretical limit" (see `TheoreticalLimitLMLoggingMixin`), i.e. the error associated with the best hypothesis that Monty has initialized. Unbound evidence currently means that an older, "good enough" hypothesis will often remain the MLH even where the theoretical limit is lower following sampling of new hypotheses (TODO link to burst sampling). Note that a complication of this is that standard measures of rotation error will fail to account for objects that have symmetry, which Monty detects (TODO link). As such, an additional change may be to modify the theoretical limit to use alternative measures of rotation error that are not biased by symmetry, such as the Chamfer Distance metric used in our Thousand Brains systems paper (TODO link). 

### Evidence Interpretation and Hyperparameter Tuning are Challenging

A second issue is that, when resampling of hypotheses is allowed, unbounded evidence values become difficult to interpret and reason about.
As evidence grows arbitrarily large, the absolute magnitude of a score becomes less meaningful, and comparisons depend increasingly on the hypotheses age values rather than consistency with recent observations.
Bounded scores are easier to work with as they allow downstream components to rely on known numerical ranges, simplifying threshold selection and enabling more stable decision rules for deleting and resampling hypotheses.

## Potential Solutions

Bounding evidence through methods such as [exponential moving averages](https://youtu.be/A1cOwvZpgjU?si=1oyk6-BTij6TaRdG&t=1668), weighted averages, or normalization by hypothesis age can keep scores within a bounded range.
However, these techniques compress historical information and cause memory to fade over time.
Monty loses the influence of important past features, such as a mug handle, as a result, recognition can become less accurate in long sequences.

A promising direction is to pair bounded evidence with [saliency-driven saccades](https://docs.thousandbrains.org/docs/implement-efficient-saccades-driven-by-model-free-and-model-based-signals) or other attention mechanisms, which we now have a working version of (TODO link to vocus).
If the system consistently revisits the key discriminative features that are important for recognition, and does so efficiently, then those features remain present in recent history.

## A Biological Connection and Other Benefits

It is worth highlighting that this Future Work item bears some interesting parallels to Monty more closely modeling the brain. In particular, neurons in the brain can be well described by the leaky-integrate-and-fire (LIF) model (TODO link and image below). LIF neurons gradually accumulate incoming spikes, up to a limit at which they spike and reset their membrane potential. At the same time, they constantly lose current (hence "leaky") towards a resting membrane potential. We believe something like a "Cumulative Moving Average neuron" (CMA neuron) could well approximate the dynamics we are interested (evidence upper-bounded and temporal decay) that are captured in LIF neurons.

The above biological framing may help us develop other useful properties. For example, "spiking" in CMA neurons (i.e. reaching a particular threshold) could be a basis for communicating representations further up in the hierarchy or via voting, including informing associative learning between LMs. At the same time, hypotheses in Monty can be thought of as "neuron instances" - unlike biological brains that must use a fixed capacity of physical neurons, we have the benefit that we can initialize new neural instances as required. This enables us to trade off computational intensity for parallelization and faster inference speed, and enabling a new kind of neuron that borrows from the strengths of silicon and biological systems.