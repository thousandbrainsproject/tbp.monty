---
title: Create Dataset and Metrics to Evaluate Categories and Generalization
---

Datasets do not typically capture the flexibility of object labels based on whether an object belongs to a broad class (e.g. cans), vs. a specific instance of a class (e.g. a can of tomato soup).

Labeling a dataset with "hierarchical" labels, such that an object might be both a "can", as well as a "can of tomato soup" would be one approach to capturing this flexibility. Once available, classification accuracy could be assessed both at the level of individual object instances, as well as at the level of categories.

We might leverage crowd-sourced labels to ensure that this labeling is reflective of human perception, and not biased by our beliefs as designers of Monty. This also relates to the general problem fo [Multi-Label Classification](https://paperswithcode.com/task/multi-label-classification), and so there may be off-the-shelf solutions that we can explore.

Initially such labels should focus on morphology, as this is the current focus of Monty's recognition system. However, we would eventually want to also account for affordances, such as an object that is a chair, a vessel, etc. Being able to classify objects based on their affordances would be an experimental stepping stone to the true measure of the systems representations, which would be how well affordances are used to manipulate the world.