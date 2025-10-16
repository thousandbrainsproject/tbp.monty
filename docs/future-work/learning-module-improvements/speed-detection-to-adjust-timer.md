---
title: Speed Detection to Adjust Timer
description: Add ability to detect offsets in timer input and learned sequence model to speed up or slow down the global interval timer.
---

This item relates to the broader goal of [modeling object behaviors in Monty](../../theory/recent-progress/object-behaviors.md#implementation-in-monty). For more details on time representations and processing, see our future work page on the [interval timer](../cmp-hierarchy-improvements/global-interval-timer.md).

The LM has expectations of when it will sense the next feature at the current location. This is stored as the interval duration that came in from the global interval timer during learning at the same time as the sensory input.

If the next expected feature at the current location appears earlier than what is stored in the model (i.e. timer input < stored interval), the LM sends a signal to the global timer to speed up (by the magnitude of the difference).

If the next expected feature at the current location appears later than what is stored in the model (i.e. timer input > stored interval), the LM sends a signal to the global timer to slow down.

Note: This might be a noisy process and require voting to work well.
