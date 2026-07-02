---
title: Use Constrained Grid Object Models for All Benchmark Experiments
description: Evaluate and finetune the GridObjectModels to work with all of our benchmark experiments with new hierarchical configs.
rfc: optional
estimated-scope: medium
improved-metric: learning, generalization, scale
output-type: experiments, analysis
skills: research, monty
contributor: 
status: open
---

Most of our benchmark configs still rely on models that use the `DisplacementGraphLM` during learning. We have developed the [constrained object model](../../how-monty-works/learning-module/object-models.md#object-models) (`GridObjectModel` class), which has many notable advantages. For example, this learning module will likely be key to making progress in [unsupervised learning](./test-grid-object-models-for-unsupervised-learning.md), and so more accurately reflects a general learning module that can handle the challenges of learniing in the real world.

One issue preventing a simple switch to the constrained object models in all of our benchmarks is that a key parameter of the `GridObjectModel` is the physical scale of the reference frame that it contains, and therefore the objects it can effectively model (`max_size`). This relates to how we believe cortical columns will have a preferred scale (e.g., smaller scales for columns in V1, and larger scales for columns in IT cortex, just as grid cells in entorhinal cortex have different spatial scales). However, as most of our configs rely on a single LM, we are limited in our ability to handle different scales in a way that a hierarchical visual cortex could. This is problematic, as the scale of objects in YCB can vary significantly, from the small dice, to large food packages.

A suggested approach is to change our core benchmarks configs to leverage a hierarchical Monty system. Some experimentation will likely be necessary, but something on the order of three stacked LMs with progressively increasing spatial scales will likely be sufficient. Ideally these would match the different scales we use in the hierarchical Monty specified in our [compositional benchmarks](https://docs.thousandbrains.org/docs/benchmark-experiments#compositional-datasets), which already use the constrained object models (note that at the time of writing, these configs are still in flux). 

With the above change, we should be able to maintain reasonable performance on our YCB benchmarks while using the `GridObjectModel` class. Note that this work may require examining the convergence dynamics of Monty (in particular the parameter `min_lms_match`), as for certain objects, only 1 LM in the hierarchy may have a high-quality model.

This update could work in concert with [an improved exploration policy for the distant agent](../motor-system-improvements/model-free-distant-agent-exploration-policy.md). Early tests with salience-based learning policies suggest that these naturally support sampling in a scale-invariant way. In particular, this policy will perform smaller saccades to explore a smaller object such as a dice, and larger saccades to explore an object like a cereal box; this is in contrast to the naive scan and surface-agent learning policies, which will sample many more points for a larger object.

Finally, this change could enable a significant simplification of the Monty code-base by removing the `FeatureGraphLM` and `DisplacementGraphLM` classes. Even if some elements of the latter are cherry-picked for the purpose of a [hybrid node and edge-matching LM](./hybrid-node-and-edge-matching.md), this would still enable the removal >1k lines of unused code and dozens of associated unit tests.