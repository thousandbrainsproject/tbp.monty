---
title: Include State in Models
description: Add a 'state' dimension to the models learned in Monty that conditions which features to expect at what locations.
---

This item relates to the broader goal of [modeling object behaviors in Monty](../../theory/recent-progress/object-behaviors.md#implementation-in-monty).

Any model in an LM can be state conditioned. For instance a stapler might be open or closed, and an object behavior is represented as a sequence of states. Depending on the state of an object, the model expects to see different features at different locations.

![Both behavior (left) and morphology models can be state-conditioned. This means that depending on the state, different features/changes are expected at different locations. State might be traversed through passing time or by applying actions.](../../figures/theory/state_conditioning.png#width=600px)

In Monty, state can be interpreted as a sub-object ID. Depending on the state of an object Monty will look at the model stored for this state (also see [include state in hypotheses](include-state-in-hypotheses.md)). 

One alternative is to represent state as a 4th dimension in the models such that every location on the object is represented as (x, y, z, s). However, it seems like interpolation along the state/time dimension is likely limited. Furthermore, measuring distance along the state dimension is likely quite different from along the three spatial dimensions. Neither of these points would be reflected well if using a continuous 4D model. 

States are learned as an ordered sequence. The model of an object can include an ordered sequence of states as well as the temporal duration between states in the sequence. The interval length between two states is provided by the timer input to the LM and can be stores in the models. States might also be traversed by applying actions, however we don't have a concrete proposal for this yet.

Both behavior & morphology models can have different states and sequences and both can be driven by time or other factors. In other words, there is no difference between the LMs besides their input.