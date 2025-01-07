---
title: Deal with Moving Objects
---

This work relates to first being able to [Detect Local and Global Flow](../../future-work/sensor-module-improvements/detect-local-and-global-flow.md).

Our current idea is to then use this information to model the state of the object, such that beyond its current pose, we also capture how it is moving as a function of time. This information can then be made available to other learning modules for voting and hierarchical processing.

This work also relates to [Modeling Object Behaviors and States](../../future-work/learning-module-improvements/implement-test-gnns-to-model-object-behaviors-states.md), as an object state might be quite simple (the object is moving in a straight line at a constant velocity), or more complex (e.g. in a "spinning" or "dancing" state). To pass such information via the Cortical Messaging Protocol, the former would likely be treated similar to pose (i.e. specific information shared, but limited in scope), while the latter would be shared more similar to object ID, i.e. via a summary representation that can be learned via association.
