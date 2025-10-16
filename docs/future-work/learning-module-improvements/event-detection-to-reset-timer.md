---
title: Event Detection to Reset Timer
description: Add ability to detect significant events which are used as signals to reset the global interval timer.
---

This item relates to the broader goal of [modeling object behaviors in Monty](../../theory/recent-progress/object-behaviors.md#implementation-in-monty). For a broader overview of the (semi)-global interval timer, see [this future work page](../cmp-hierarchy-improvements/global-interval-timer.md).

Any LM should have the ability to detect significant events (such as the attack of a new note, an object reaching a canonical state, or a significant change in input features). If one such event is detected, the LM (or LMs) send a signal to the interval timer (outside the LM) which is then reset. The interval timer provides input to a large group of LMs and hence a detected event in one LM can reset the input signal to many LMs.

We might also want SMs to have the ability to detect significant events but this is unclear yet.