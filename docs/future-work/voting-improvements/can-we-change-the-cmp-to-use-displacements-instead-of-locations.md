---
title: Can we change the CMP to use displacements instead of locations?
---

Movement is core to how LMs process and model the world. Currently, an LM receives an observation encoded with a body-centric location, and then infers a displacement in object-centric coordinates. Similarly, goal-states are specified as a target location in body-centric coordinates, which are then acted upon.

However, a more general formulation might be to use displacements as the core spatial information in the CMP, such that a specific location (in body-centric coordinates or otherwise) is never communicated outside of an LM or sensor module.

Such an approach might align well with adding information about flow (see [Detect Local and Global Flow](../sensor-module-improvements/detect-local-and-global-flow.md)), modeling moving objects (see [Deal With Moving Objects](../learning-module-improvements/deal-with-moving-objects.md)), and supporting abstract movements like the transition from grandchild to grandparent.