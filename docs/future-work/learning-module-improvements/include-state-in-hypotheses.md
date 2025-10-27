---
title: Include State in Hypotheses
description: Infer state of object, in addition to its pose.
---

This item relates to the broader goal of [modeling object behaviors in Monty](../../theory/recent-progress/object-behaviors.md#implementation-in-monty).

Once we [integrate state into the models learned in LMs](include-state-in-models.md), we will also need to add state to our hypotheses to infer the state the object is in.

To do this, a hypothesis needs to include which of the states/intervals we are in. Hence, they are represented as (l, r, s), where l is the location on the object, r is the rotation of the object, and s is the state in the sequence. In Monty, we would add a `states` variable to the `Hypotheses` class (in addition to the existing `locations` and `poses` arrays).

Analogous to testing possible locations and possible poses, we would test possible states. Depending on sensory input, the evidence for different location-rotation-state combinations would be inceremented/decremented and hypotheses would be narrowed down.

We can use our existing nearest neighbor search algorithm to retrieve neighbors in 3D space to predict which feature should be sensed next in the current state, at the current location. 

Given the input interval stored during learning, the model can also predict with which global clock input the next input feature should coincide.
The timing within an interval/state is provided by the global clock and does not need to be inferred. It can however be used to correct the speed of the timer to recognize sequences at different speeds. For more details see the page on [speed detection to adjust timer](speed-detection-to-adjust-timer.md).

Note that adding an additional dimension to the hypothesis space will add a multiplicative factor to it's size and computational cost. Hence [using fewer points to represent models](../learning-module-improvements/use-models-with-fewer-points.md) will likely be important.