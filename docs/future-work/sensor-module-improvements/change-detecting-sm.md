---
title: Change Detecting SM
description: Add a new type of sensor module that detects local changes and output's those as CMP messages.
---

This item relates to the broader goal of [modeling object behaviors in Monty](../../theory/recent-progress/object-behaviors.md#implementation-in-monty). It also builds on [detecting local and global flow](detect-local-and-global-flow.md) in the SM.

As outlined in the [theory section on modeling object behaviors](../../theory/recent-progress/object-behaviors.md), the idea is that we can use the same LM for modeling object behaviors as we do for modeling static objects. The only difference is what kind of input the LM receives. An LM that receives input from the change detecting SM outlined here would be learning behavior models.

![The introduction of the change detecting LM means that the LM that receives input from this new SM will learn behavior models.](../../figures/theory/monty_implementation_change_SM.png)

The change detecting SM would detect changes such as a local moving feature (like a bar moving in a certain direction) or a local changing feature (like color or illumination changing). Whenever it detects such a change it output's this change as a CMP message. The output has the same structure as the output of other SMs. It contains the location in a common reference frame (e.g. rel the body) at which the change was detected, the orientation of the change (e.g. to direction in which the edge moved), and optional features describing the change (e.g. edge vs. curve moving).

It is important that the SM only outputs information about local changes. If there is global change detected (like all features across the sensor are moving), this usually indicates that the sensor itself is moving, not the objects in the world. For more details see [detecting local and global flow](detect-local-and-global-flow.md).

Potential details to keep in mind:
- May need to update static feature SM to make sure it doesn’t send output when local movement is detected
- May get noisy estimates of 3D movement from a 2D depth image
