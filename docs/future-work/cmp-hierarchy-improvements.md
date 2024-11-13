---
title: CMP & Hierarchy Improvements
description: Improvements we would like to add to the CMP or hierarchical information processing.
---
These are the things we would like to implement:

- [Figure out performance measure and supervision in heterarchy](cmp-hierarchy-improvements/figure-out-performance-measure-and-supervision-in-heterarchy.md) #infrastructure
- [Add top-down connections](cmp-hierarchy-improvements/add-top-down-connections.md) #numsteps
- [Add associative connections](cmp-hierarchy-improvements/add-associative-connections.md) #abstract #numsteps
- [Run & Analyze experiments with >2LMs in heterarchy testbed](cmp-hierarchy-improvements/run-analyze-experiments-with-2lms-in-heterarchy-testbed.md) #compositional
- [Run & Analyze experiments in multiobject environment looking at scene graphs](cmp-hierarchy-improvements/run-analyze-experiments-in-multiobject-environment-looking-at-scene-graphs.md) #multiobj
- [Test learning at different speeds depending on level in hierarchy](cmp-hierarchy-improvements/test-learning-at-different-speeds-depending-on-level-in-hierarchy.md) #learning #generalization
- [Send similarity encoding object ID to next level & test](cmp-hierarchy-improvements/send-similarity-encoding-object-id-to-next-level-test.md) #compositional

Please see the [instructions here](project-roadmap.md#how-you-can-contribute) if you would like to tackle one of these tasks.

> 📘 Heterarchy vs. Hierarchy
> 
> We sometimes use the term heterarchy to express the notion that, similar to the brain, connections in Monty aren't always strictly hierarchical. There are many long-range connections (voting) within the same hierarchical level and across level, as well as skip-connections. Also, every level in the "hierarchy" has a motor output (goal state) that it sends to the motor system.