---
title: Test Grid Object Models for Unsupervised Learning
description: Evaluate and finetune the GridObjectModels in unsupervised settings.
rfc: optional
estimated-scope: medium
improved-metric: learning, generalization
output-type: experiments, analysis, publication
skills: research, monty
contributor: 
status: open
---

Since developing the [constrained object models](../../how-monty-works/learning-module/object-models.md#object-models) (`GridObjectModel` class), we have not had the chance to extensively test them. We have especially not tested them much in unsupervised learning scenarios and as a way to generalize common features of similar objects.

To test this, one could use our existing environments (e.g. YCB objects in Habitat) and unsupervised learning configs (like outline in [this tutorial](../../how-to-use-monty/tutorials/unsupervised-continual-learning.md)) as a starting point. From there, one would need to update configs to use the `GridObjectModel`, run experiments with different model parameters, and visualize the results.

As a next step, shortcomings and issue can be identified and potential solutions tested (ideally starting with just tweaking the existing parameters).

The existing datasets aren't ideal for testing the learning of categories of objects and generalizing. One would at the least have to update the way performance is measured to quantify improvements in those abilities. One could also [develop different datasets aimed more an testing categories](../environment-improvements/create-dataset-and-metrics-to-evaluate-categories-and-generalization.md), such as showing a collection of different types of mugs and trying to learn their commonalities in a model.

One may also explore the GridObjectModel in combination with a hierarchical Monty setup and test whether object parts naturally emerge (e.g. mug handle learned at lower level and full mug at higher level without explicit supervision). Testing this would likely deserve it's own future work item.

It would also be good to test how useful the `GridObjectModel` is for "forgetting" statistical irregularities. For instance, how fast a noisy observation will be forgotten (or never even added) or how different sets of model parameters lead to learning "that specific mug with a chip in it" or "any mug in my cabinet".