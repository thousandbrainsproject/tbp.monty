---
title: Unsupervised Learning in a Hierarchy
description: Test and improve learning with multiple LMs stacked on top of each other without supervising them.
rfc: optional
estimated-scope: unknown
improved-metric: learning, compositional
output-type: experiments, analysis, PR, publication
skills: python, research, monty
contributor: 
status: open
---

Currently, training a hierarchical setup of Monty involves a complicated procedure of supervising the lower-level LMs to learn the child objects, followed by supervising the higher-level LM to learn the parent object. This is complex and unrealistic to scale. It also requires labeled datasets where child objects can be learned in isolation first.

A first step for this item would be to test what kind of models are learned if there is no supervision. The `max_model_size` parameter of the [constrained object models](../../how-monty-works/learning-module/object-models.md#object-models) can be leveraged to encourage the lower level LMs to learn smaller sub-components. 

In addition, models which [learn and forget at different rates](learning-and-forgetting-speed-parameter.md) could gradually disentangle objects from one another by only storing information that consistently appears together (i.e. the features of a child object at their relative arrangements).

Depending on the results, methods should be devised to improve Monty's ability to learn in this kind of scenario. This might involve testing other policies, such as model-free segmentation to stay on sub-components of an object. It might also involve changes to Monty's algorithm. 

This item relates to [testing grid object models for unsupervised learning](./test-grid-object-models-for-unsupervised-learning.md).