---
title: Use Out of Reference Frame Movements
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

Consider the situation described under using off-object observations. What if a hypothesis actually believes it moved out of the reference frame? If it receives an off-object observation, wouldn't this be consistent with its prediction? Should this then provide positive evidence for this hypothesis? This setup is shown diagramically below:

![Example of out of reference frame movement](../../figures/future-work/off_object_obs.png)
*Example of a sensor movement resulting in the hypothesis moving out of the learned reference frame of an object.*

On the surface (sorry...) this might appear like a good approach, however, as noted the hypothesis has moved out of the reference frame. An LM that knows about an object is not expected to have a long term memory about a scene-like representation of where that object is, so as soon as it moves out of an object's reference frame, then from that LM's perspective, the object does not exist at that location.

This is therefore a unique condition, because the LM should be able to immediately discount this hypothesis as invalid (or at least, unlikely), i.e., before any sensory observation has been received. To emphasize this distinction, we call this an *out-of-reference-frame-movement* (OORFM). This must also be handled differently from off-object observations because there may in fact be an "on-object" observation after such a movement. Whether the observation is on or off any object however, the hypothesis is not valid if it is not within the reference frame.

Note that any concept of object permanence - for example when moving back to where the object was, should be captured via hierarchy - a higher level LM would have a scene-like representation, and could use top-down biasing to help the LM in question recall what object it had just been observing. 

We have performed some early expiriments evaluating the elimination of hypotheses that undergo OORFMs. We found that when there is no location noise, it can help reduce the hypothesis space without any negative impact on accuracy. However, as uncertainty about where the sensor is located is introduced, there is a risk that good hypotheses are eliminated. This likely relates to our use in these experiments of the default matching distance to determine if a movement qualifies as an OORFM or not. As such, a more tolerant approach to classifying OORFMs will likely be required. We are likely to see some benefits in convergence speed with such a modified approach, however we have paused this work as it did not relate strongly to improving performance with compositional objects.

See also the linked RFC for potential considerations around unsupervised learning and incomplete models.