---
title: Better Model Surface Representation
description: Figure out and test a way to represent an object's surface better in our models and/or matching algorithm.
rfc: optional
estimated-scope: unknown
improved-metric: speed, pose, accuracy, noise
output-type: experiments, analysis, PR
skills: python, research, monty-beginner
contributor: 
status: open
---

> [!NOTE]
> This item relates to [using models with fewer points](use-models-with-fewer-points.md), which would be one of the ways to measure success here.

The naive way of testing whether a sensor is still on the object if a given hypothesis is correct is to apply the movement in the object's RF and, searching in the radius of the new location, determine whether there are points stored nearby in the model. However, this disregards the concept of surface, where we would want to have very little tolerance when the movement is going off the surface (i.e. in the direction of the surface normal) but allow for more tolerance if the next point along the surface is a bit further away. If we don't allow this second tolerance, we are forced to learn dense models where even a plain, even surface is represented with many points in the model because otherwise we won't find near enough neighbors.

![Example of problems with the naive approach of search. Sparsity in the model will lead to some search locations not having a nearby point stored in the model (red). If we increase the search radius, many points that should be off the object will be considered on object instead.](../../figures/future-work/search_naive.png)

This problem is partially addressed by the custom distance metric we use in the EvidenceGraphLM, which takes into account the direction of the sensed surface normal, as well as the amount of curvature, when searching for nearest neighbors in the model (read more [here](../../how-monty-works/learning-module/evidence-based-learning-module.md#search-radius-and-distance-weighting)). However, it only works up to a certain degree of sparsity in the models. It is also only an approximation of surface representation (only during inference, using spherical radius without distinction between going into vs. out of the surface, basic heuristic to account for curvature). There might be small tweaks that could be made to it (account for curvature better, test parameters more) or a larger rethinking of representing surface in the models.

![How we currently account for surface when searching the models. Depending on the amount of curvature sensed, the search radius will be narrowed more or less in the direction of the sensed surface normal. Points in the radius with a surface normal pointing in the opposite direction of the hypothesized normal will be ignored in the evidence calculation. This accounts for thin surfaces where points on both side of a surface will end up in the radius.](../../figures/future-work/search_incl_surface_rep.png)

For example, the existing method uses the surface curvature sensed at a point to adjust the search radius. However, we are not leveraging information stored in the models for this part. One could think about an inversion of how search operates - rather than the sensation determining the extent of the matching radius, learned characteristics of points and their surrounding surface could possibly dictate this.