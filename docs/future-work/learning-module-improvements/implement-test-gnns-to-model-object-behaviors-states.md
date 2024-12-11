---
title: Implement & Test GNNs to Model Object Behaviors & States
---

We would like to test using local functions between nodes of an LM's graph to model object behaviors. In particular, we would like to model how an object evolves over time due to external and internal influences, by learning how nodes within the object impact one-another based on these factors. This relates to graph-neural networks, and [graph networks more generally](https://arxiv.org/pdf/1806.01261), however learning should rely on sensory and motor information local to the LM. Ideally learned relations will generalize across different edges, e.g. the understanding that two nodes are connected by a rigid edge vs. a spring.

As noted, all learning should happen locally within the graph, so although gradient descent can be used, we should not back-propagate error signals through other LMs. Please see our related policy on [using Numpy rather than Pytorch for contributions](../../contributing/style-guide#numpy-preferred-over-pytorch). For further reading, see our discussion on [Modeling Object Behavior Using Graph Message Passing](https://github.com/thousandbrainsproject/monty_lab/tree/main/object_behaviors#implementation-routes-for-the-relational-inference-model) in the Monty Labs repository.

We have a dataset that should be useful for testing approaches to this task, which can be found in [Monty Labs](https://github.com/thousandbrainsproject/monty_lab/tree/main/object_behaviors).

At a broader level, we are also investigating alternative methods for modeling object behaviors, including sequence-based methods similar to HTM, however we believe it is worth exploring graph network approaches as one (potentially complementary) approach. In particular, we may find that such learned edges are useful for frequently encountered node-interactions like basic physics, while sequence-based methods are best suited for idiosyncratic behaviors.