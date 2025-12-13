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

Additionally, one could test on different datasets, such as showing a collection of different types of mugs and trying to learn their commonalities in a model.

It would also be good to test how useful the `GridObjectModel` is for "forgetting" statistical irregularities. For instance, how fast a noisy observation will be forgotten (or never even added) or how different sets of model parameters lead to learning "that specific mug with a chip in it" or "any mug in my cabinet".