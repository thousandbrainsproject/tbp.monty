---
title: Use Out of Model Movements
description: Ensure that off-object observations are processed by LMs, resulting in evidence updates.
rfc: (see section 2) https://github.com/thousandbrainsproject/tbp.monty/pull/366/files
estimated-scope: medium
improved-metric: speed, numsteps
output-type: experiments, analysis, PR
skills: python, research, monty
contributor: hlee
status: paused
---

This task relates to [Use Off Object Observations](/use-off-object-observations.md), and so it is recommended that you read that description first.

Consider the situation described under using off-object observations. What if a hypothesis actually believes it moved out of the model's space? It might then receive an "off object" observation, or an observation from another object that happens to be there. This setup is shown diagrammatically below:

![Example of out of model movement](../../figures/future-work/out_of_model_movement.png)
*Example of a sensor movement resulting in the hypothesis moving out of the learned reference frame of an object.*

NOTE: In the past we have sometimes referred to this as an out of "reference frame" movement. However, it is most accurate to describe the reference frame as the representational space where a model is learned. As such, describing it as an out-of-model-movement (OOMM) is more accurate.

We believe the best approach is that the LM would retain it's hypothesis for a short period of time (perhaps 2-3 steps in the current, discrete version of Monty). If it then moves back into the reference frame before this time has elapsed, it can continue to accumulate evidence.

If a hypothesis stays outside of the reference frame for too long however, then the hypothesis should receive strong negative evidence. In the resampling version of Monty, this would effectively be handled by the evidence slope being sufficiently small that the hypothesis is deleted.

This approach ensures that a Learning Module is not expected to maintain a long-term memory of objects it is not currently on. At the same time, it means that we can move out of a reference frame, which can be a useful way to test other hypotheses (i.e. for objects where we do expect to find something there). By then moving back onto the object and continuing to accumulate evidence, certain hypotheses can continue to grow.

Other details
- If a MLH moves out of a reference frame and becomes "clamped" as a result, it should not continue to pass information up in a hierarchy of LMs. Intuitively, if we believe we are on a mug at location x, and then move to location y which is off of the mug, then we should not communicate to the next LM in the hierarchy that there is a mug at location y.

Note that any concept of object permanence - for example when moving back to where the object was, should be captured via hierarchy - a higher-level LM would have a scene-like representation, and could use top-down biasing to help the LM in question recall what object it had just been observing.

Potential "gotchas" to be aware of:
- Currently hypotheses receive negative evidence when they move out of the model; by making the proposed change, we may find that certain (incorrect) hypotheses are eliminated more slowly. However, in practice, the window for maintaining a hypothesis is so short that it will hopefully have a negligible impact on the number of matching steps for convergence. Moreover, this change may have already been necessary to enable Monty to accommodate more sparse models, which is one of our current aims (see [Use Models With Fewer Points](/use-models-with-fewer-points.md))
- We previously performed some early experiments evaluating the immediate elimination of hypotheses that undergo out-of-model-movements. We found that when there is no location noise, it can help reduce the hypothesis space without any negative impact on accuracy. However, as uncertainty about where the sensor is located is introduced, often good hypotheses were eliminated. We expect that the approach proposed here (clamp hypotheses for a few steps before deleting them) should mean this does not happen.

See also the linked RFC for potential considerations around unsupervised learning and incomplete models.