---
title: Test Learning at Different Speeds Depending on Level in Hierarchy
---

Our general view is that episodic memory and working memory in the brain leverage similar representations to those in learning modules, i.e. structured reference frames of discrete objects.

For example, the brain has a specialized region for episodic memory (the hippocampal complex), due to the large number of synapses required to rapidly form novel binding associations. However, we believe the core algorithms of the hippocampal complex follow the same principles of a cortical column (and therefore a learning module), with learning simply occurring on a faster time scale.

As such, we would like to explore adding forms of episodic and working memory by introducing high-level learning modules that learn information on extremely fast time scales relative to lower-level LMs. These should be particularly valuable in settings such as recognizing multi-object arrangements in a scene, and providing memory when a Monty system is performing a multi-step task. Note that because of the overlap in the core algorithms, LMs can be used largely as-is for these memory systems, with the only change being the learning rate.

It is worth noting that the `GridObjectModel` would be particularly well suited for introducing a learning-rate parameter, due to its constraints on the amount of information that can be stored.

As a final note, varying the learning rate across learning modules will likely play an important role in dealing with representational drift, and the impact it can have on continual learning. For example, we expect that low-level LMs, which partly form the representations in higher-level LMs, will change their representations more slowly.