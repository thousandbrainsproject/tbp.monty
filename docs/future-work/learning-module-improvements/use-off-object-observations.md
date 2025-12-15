---
title: Use Off-Object Observations
description: Ensure that off-object observations are processed by LMs, resulting in evidence updates.
rfc: optional
estimated-scope: medium
improved-metric: numsteps, accuracy
output-type: experiments, analysis, PR
skills: python, research, monty
contributor: hlee
status: scoping
---

There are a few instances where Monty can move off of an object, sensing nothing/empty space (or at least no morphological features). For example, this can occur due to a saccade moving off the object and pointing at a void (such as the sky), or a touch-based agent leaving the surface of an object.

Currently we have methods to move the sensor back onto the object, however we do not make use of the information that there was no object at the perceived location before doing so. This is valuable information, as the absence of the object at a location will be inconsistent with some object and pose hypotheses - hypotheses that should therefore have their associated evidence reduced. This setup is shown in the diagram below.

![Example of off object observations](../../figures/future-work/off_object_obs.png)
*Example of a sensor moving off of an object and observing nothing.*

To address this, we need to update how these observations are processed such that:
1. These observations are formulated as an appropriate "null"-type observation - there is no surface to observe, and so there should be no morphological features. However, there would still be a location (e.g. where the finger tip is hovering, or an eye looking into the far distance), and there can be non-morphological features like color.
2. These observations are still passed to the learning module.
3. If a hypothesis predicts an observation, then the learning module's evidence update appropriately results in negative evidence, as no object model should ever store a "null" feature that would match this observation.

### What if a hypothesis itself has moved out of the model?

See [Use Out of Model Movements](./use-out-of-model-movements.md)

### What if the movement is just off the surface of the object?

This situation is shown in the below diagram:

![Example of off object observations near a surface](../../figures/future-work/off_object_near_surface.png)
*Example of a sensor moving off of an object and observing nothing when it is close to the object's surface.*

In this case, while some hypotheses on the surface of the object might be close (even within the neighborhood tolerance) to the sensors actual location, these will receive negative evidence, as they will predict being on the actual object. This is the desired behavior, as the LM should not predict that it is still on the object if it is no longer sensing it. However, we should keep this effect in mind when testing this as it might cause "almost correct" hypotheses to get a lot of negative evidence and thereby lead to some performance drops.

### Gotchas to Watch Out For

From an initial look, this change might appear relatively straightforward, however we have found ourselves repeatedly facing conceptual or practical issues when visiting this task. A few potential gotchas to consider:
1. We need to ensure that any change to how sensor modules pass these observations to an LM does not interfere with the existing policies. In particular, the distant agent's random saccade policy has an `undo_last_action` that moves the sensor back onto the object if it moves off of it. Similarly, after performing a hypothesis testing jump, a `handle_failed_jump` process reverses the last action. Ideally we would still perform these corrective actions, but only after the LMs have a chance to process the observation.
2. During learning, we want to ensure that observations associated with these "null" morphological features are not stored, i.e. are not learned as part of any model.
3. The FeatureChangeSM has additional logic for determining when to pass an observation to the LMs, and this needs to be accounted for in any changes.
4. The buffer uses information about off-object observations to filter data, so some elements here may need updating.
5. There is a [discontinued RFC on this topic](https://github.com/thousandbrainsproject/tbp.monty/pull/425/files) which has some useful discussion of the above points.

### Outcome Measures
- We anticipate that the primary benefit of this change will be that Monty can use off-object observations to more quickly eliminate hypotheses; this will result in fewer steps before convergence.
- However, it may also improve Monty's accuracy, at least on certain recognition tasks. In particular, consider the instance where Monty has two hypotheses, which are a number "1" and a number "7", or a letter "i" vs a short vertical bar "ı" without a dot. When the actual object observed is the one with fewer features / a smaller model (the "1" rather than the "7", or the i without a dot (ı) rather than "i"), then Monty currently cannot recognize the object. If it moves to the location where it might expect the upper bar of a seven, or the dot of an i, it will perceive nothing. Without processing this observation, Monty cannot distinguish the hypotheses. Thus, this future work item is necessary to resolve this issue. In addition however, resolving this setting will also require adjusting the [Use of Out of Model Movements](./use-out-of-model-movements.md) to ensure we briefly maintain the correct hypothesis (1 or ı, respectively). With these two changes in place, we expect the incorrect hypothesis for the larger model (7 or i) to receive negative evidence, while the correct hypothesis for the smaller model is maintained such that it can continue to accumulate evidence, and ultimately win out.

NOTE: The instance where Monty moves onto *another* object (for example, a distant wall, or an occluding robotic hand) is already handled by the existing matching process in Monty. In particular, this should result in unexpected sensory input if an LM believes it is still on the original object, and therefore will (correctly) result in negative evidence. This instance is therefore not considered the "off object" setting described above that we wish to resolve.