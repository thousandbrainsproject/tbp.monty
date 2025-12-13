---
title: Make Dataset to Test Compositional Objects
description: Create more complex compositional object meshes that we can load in our simulator and test Monty on.
rfc: optional
estimated-scope: medium
improved-metric: compositional
output-type: experiments, testbed, mesh-models
skills: python, 3d-modeling
contributor: scottcanoe, ramyamounir
status: evergreen
---

> [!TIP] We have implemented a first dataset to test compositional modeling!
> You can find the dataset descriptions and current performance on it in our [benchmark documentation](../../overview/benchmark-experiments.md#compositional-datasets).
> Nevertheless, we will eventually want to implement a more complex dataset as described below to test scene level representations and generalization.

To test compositional objects, we would like to develop a minimal dataset based on common objects (such as mugs and bowls) with logos on their surfaces. This will enable us to learn on the component objects in isolation, while moving towards a more realistic setting where the component objects must be disentangled from one another. The logo-on-surface setup also enables exploring interesting challenges of object distortion, and learning multiple location-specific associations, such as when a logo has a 90 degree bend half-way along its length.

It's worth noting that observing objects and sub-objects in isolation is often how compositional objects are learned in humans. For example, when learning to read, children begin by learning individual letters, which are themselves composed of a variety of strokes. Only when letters are learned can they learn to combine them into words. More generally, disentangling an object from other objects is difficult without the ability to interact with it, or see it in a sufficient range of contexts that its separation from other objects becomes clear.

We would eventually expect compositional objects to be learned in an unsupervised manner, such as that a wing on a bird is a sub-object, even though it may never have been observed in isolation. When this is consistently possible, we can consider more diverse datasets where the component objects may not be as explicit. At that time, the challenges described in [Figure out Performance Measure and Supervision in Heterarchy](../cmp-hierarchy-improvements/figure-out-performance-measure-and-supervision-in-heterarchy.md) will become more relevant.

In the future, we will move towards policies that change the state of the world. At this time, an additional dataset that may prove useful is a "dinner-table setting" with different arrangements of plates and cutlery. For example, the objects can be arranged in a normal setting, or aligned in a row (i.e. not a typical dinner-table setting). Similarly, the component objects can be those of a modern dining table, or those from a "medieval" time-period. As such, this dataset can be used to test the ability of Monty systems to recognize compositional objects based on the specific arrangement of objects, and to test generalization to novel compositions. Because of the nature of the objects, they can also be re-arranged in a variety of ways, which will enable testing policies that change the state of the world.

![Dinner table set](../../figures/future-work/dinner_variations_standard.png)
*Example of compositional objects made up of modern cutlery and plates.*

![Dinner table set](../../figures/future-work/dinner_variations_medieval.png)
*Example of compositional objects made up of medieval cutlery and plates.*

> [!NOTE]
> While we always want to test Monty on more complex and varied environments (hence the evergreen status), it can be useful to have more clearly defined items on our future work roadmap. If you start work on making a new compositional testbed, feel free to create a new future work page, outlining that specific testbed.