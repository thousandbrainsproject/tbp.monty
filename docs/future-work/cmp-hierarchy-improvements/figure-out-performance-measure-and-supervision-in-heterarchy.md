---
title: Figure out Performance Measures and Supervision in Heterarchy
---

As we introduce hierarchy and leverage more unsupervised learning, representations will emerge at different levels of the system that may not correspond to any labels present in our datasets. For example, handles, or the head of a spoon, may emerge as object-representations in low-level LMs, even though the dataset only recognizes labels like "mug" and "spoon".

One approach to measure the "correctness" of representations in this setting might be how well a predicted representation aligns with the outside world. For example, while LMs are not designed to be used as generative models, we could visualize how well an inferred object graph maps onto the object actually present in the world. Quantifying such alignment might leverage measures such as differences in point-clouds. This would provide some evidence of how well the learned decomposition of objects corresponds to the actual objects present in the world.

See also [Make Dataset to Test Compositional Objects](../environment-improvements/make-dataset-to-test-compositional-objects.md).