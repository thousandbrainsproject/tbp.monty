---
title: Unsupervised Learning with Burst Sampling
description: Enable and test unsupervised learning when combined with burst sampling.
rfc: optional
estimated-scope: medium
improved-metric: learning, accuracy, numsteps
output-type: experiments, analysis, PR
skills: python, research, monty
contributor: 
status: open
---

Burst sampling represents a major improvement in how hypotheses are created in a continuous manner within Monty, and should generally be a part of all future LM configurations. Unfortunately, using burst sampling in the setting of unsupervised learning (which should also become the long-term default of our LMs) is non-trivial. The good news is that with the right approach, burst sampling might actually complement, rather than hinder, unsupervised learning.

The current indication for an LM to create a new reference frame during unsupervised learning is if the evidence for all known objects is *negative* following a minimum number of matching steps. This is a conservative threshold for initializing a new reference frame, but more importantly, it is not clear whether we would ever reach this condition with burst sampling active.

In particular, burst sampling will create new hypotheses if the existing ones are performing poorly. Following a few steps, many of these hypotheses will develop at least some positive evidence before they are deleted (as the local properties of distinct objects are often similar - think about how a 1cm patch of an apple isn't very different from a 1cm patch of a bowling ball). Assuming the object is truly unfamiliar, more burst sampling will occur, and we risk never reaching a condition where all hypotheses are negative.

Instead, it would be worth using burst sampling itself as an indication for creating a new reference frame. In particular, if burst sampling continues for a prolonged period of time, this suggests that the existing models are insufficient (despite bursting), and so a new model is necessary.

Following the above change, the aim would be to switch our unsupervised learning benchmarks to use burst sampling, and evaluate its performance. It will likely be necessary to optimize how many bursts are necessary to trigger the creation of a new reference frame. As burst sampling enabled significantly better accuracy and shorter episodes in our baseline benchmarks, we may observe a similar effect when performing inference following unsupervised learning.