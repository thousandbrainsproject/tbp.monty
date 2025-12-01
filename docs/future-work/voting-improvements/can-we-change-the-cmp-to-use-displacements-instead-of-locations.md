---
title: Can we change the CMP to use displacements instead of locations?
description: Investigate using movement vectors instead of locations in the CMP and the implications it would have for Monty.
rfc: required
estimated-scope: medium
improved-metric: 
output-type: RFC
skills: monty
contributor: 
status: open
---

> [!NOTE]
> Although, there is reasonable evidence that the brain may be using movements instead of locations, it is unclear whether there are any computational benefits to doing this. Since we haven't identified concrete benefits, the `improved-metric` field is left open. A first step to tackling this task would be to think through any functional implications of this change.

Movement is core to how LMs process and model the world. Currently, an LM receives an observation encoded with a body-centric location, and then infers a displacement in object-centric coordinates. Similarly, goal-states are specified as a target location in body-centric coordinates, which are then acted upon.

However, a more general formulation might be to use displacements as the core spatial information in the CMP, such that a specific location (in body-centric coordinates or otherwise) is not the primary form of communication outside of an LM or sensor module.

Such an approach might align well with adding information about flow (see [Detect Local and Global Flow](../sensor-module-improvements/detect-local-and-global-flow.md)), modeling moving objects (see [Deal With Moving Objects](../learning-module-improvements/deal-with-moving-objects.md)), and supporting abstract movements like the transition from grandchild to grandparent. It would also result in a reformulation of "goal-states" to "goal-displacements".

Note that whatever approach is taken, we would still need to have some information about shared location representations at some level of the system in order to enable coordination and voting between LMs. This may relate to the division of "what" and "where" pathways in the brain, although this is not yet clear and requires further investigation.