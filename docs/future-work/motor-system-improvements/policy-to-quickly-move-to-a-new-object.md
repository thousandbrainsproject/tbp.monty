---
title: Policy to quickly move to a new object
---

When exploring an environment with multiple objects (including components of a compositional object), it is beneficial to quickly move to a new object when the current one has been recognized, so as to rapidly build up a model of the outside world. 

It would be useful to have a policy that uses a mixture of model-free components (e.g. saliency map) and model-based components (learned relations of sub-objects to one another in a higher-level LM) to make a decision about where to move next in such an instance.

This therefore relates to both [Model-based policy to recognize an object before moving on to a new object](./model-based-policy-to-recognize-an-object-before-moving-on-to-a-new-object.md) and [Implement efficient saccades driven by model-free and model-based signals](../motor-system-improvements/implement-efficient-saccades-driven-by-model-free-and-model-based-signals.md).

Ideally, both this policy and the policy to remain on an object could be formulated together as a form of curiosity, where a learning module aims to reduce uncertainty about the world state.