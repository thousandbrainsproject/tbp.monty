---
title: Vote on State
description: Update the voting algorithm to take the state of an object into account.
---

This item relates to the broader goal of [modeling object behaviors in Monty](../../theory/recent-progress/object-behaviors.md#implementation-in-monty).

Since a state is kind of like a sub-ID of the object, we should probably treat it like that when voting. So if a hypothesis includes state A, it will only add evidence for state A of the object (using the existing mechanism to take into account location and orientation).