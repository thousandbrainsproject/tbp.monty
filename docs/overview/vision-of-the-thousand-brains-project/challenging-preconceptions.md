---
title: Challenging Preconceptions
---
Several of the ideas and ways of thinking introduced in this document may be counter-intuitive to people used to the way of thinking prominent in current AI methods, including deep learning. For example, ideas about intelligent systems, learning, models, hierarchical processing, or action policies that you already have in mind might not apply to the system that we are describing. We therefore ask the reader to try and dispense with as many preconceptions as possible and to understand the ideas presented here on their own terms. We are happy to discuss any questions or thoughts that may arise from reading this document. Please reach out to us at [ThousandBrains@numenta.com](mailto:ThousandBrains@numenta.com).

Below, we highlight some of the most important differences between the system we are trying to build here and other AI systems.

- We are building a sensorimotor system. It learns by interacting with the world and sensing different parts of it over time. It does not learn from a static dataset. This is a fundamentally different way of learning than most leading AI systems today and addresses a (partially overlapping) different set of problems.

- We will introduce learning modules as the basic, repeatable modeling unit, comparable to a cortical column. An important thing to point out here is that none of these modeling units receives the full sensory input. For example in vision, there is no 'full image' anywhere. Each sensor senses a small patch in the world. This is in contrast to many AI systems today where the whole input is fed into a single model.

- Despite the previous point, each modeling system can learn complete models of objects and recognize them on its own. A single modeling unit should be able to perform all basic tasks of object recognition and manipulation. Using more modeling units makes the system faster and more efficient, and supports compositional and abstract representations, but a single learning module is itself a powerful system. In the single model scenario, inference always requires movement to collect a series of observations, in the same way that recognizing a coffee cup with one of your fingers requires moving across its surface.

- All models are structured by reference frames. An object is not just a bag of features. It is a collection of features at locations. The relative locations of features to each other is more important than the features themselves.