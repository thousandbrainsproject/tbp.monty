---
title: Model-Free Distant Agent Exploration Policy
description: A policy for the distant agent to rapidly visit salient regions, then systematically explore.
rfc: optional
estimated-scope: medium
improved-metric: learning, accuracy, speed
output-type: monty-feature, PR
skills: python, monty
contributor: 
status: open
---

We would like a model-free strategy for the distant agent to first rapidly explore salient regions of the visual input. In the absence of model-based signals, it should then begin systematically exploring the visual space.

As a starting point, the distant agent SMs can leverage a [salience-based goal generator](https://docs.thousandbrains.org/docs/salience-sm#saliencestrategy). This generates goals that rapidly move the sensor to salient regions. This goal generation can also leverage an [inhibition of return](https://docs.thousandbrains.org/docs/salience-sm#returninhibitor) strategy that reduces the likelihood of revisiting locations. 

The proposal is to implement a decay in the salience map such that we initially saccade to salient regions, but that these slowly become less interesting. As the salience decays, the policy will converge to a pure inhibition of return strategy. This would capture the intuitive practice of initially looking at interesting things, then exploring more uniformly, and this could continue until a model-based goal is received to move to a particular location.

This policy could be used at both learning and inference, giving us sparse, but still generalizable (cross-modality) models. In particular, one limitation of a salience-only strategy during learning and inference is that other modalities which do not follow the same policy at inference (such as the surface agent) will visit under-represented regions. At the same time, the policy proposed here could ensure that salient regions are more strongly represented at both learning and inference. This is particularly important in naturalistic settings, where sensory inputs are often sparse (e.g. objects scattered around a room).

Ideally there would also be a model-free *reset* for the salience map. For example, if we look at an entirely different part of the room, or the object being studied rotates, then the saliency map should become interesting again.

An aim would be to update the benchmarks to leverage sparser models than the current ones learned by the surface agent, even if that means an initial drop in accuracy. The key requirement is that the learned models should look reasonable (i.e. well sampled).