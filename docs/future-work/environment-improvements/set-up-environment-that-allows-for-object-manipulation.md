---
title: Set up Environment that Allows for Object Manipulation
description: Identify a simple setup for simulating and manipulating objects and turn it into an environment for Monty.
rfc: optional
estimated-scope: medium
improved-metric: dynamic, goal-policy
output-type: testbed
skills: python, simulator
contributor: 
status: open
---

See the page on [theory for decomposing goals into subgoals](../motor-system-improvements/theory-for-decomposing-goals-into-subgoals.md) for a discussion of the kind of tasks we are considering for early object-manipulation experiments. An even simpler task that we have recently considered is pressing a switch to turn a lamp on or off. We will provide further details on what these tasks might look like soon.

Beyond the specifics of any given task, an important part of this future-work component is to identify a good simulator for such settings. For example, we would like to have a setup where objects are subject to gravity, but are prevented from falling into a void by a table or floor. Other elements of physics such as friction should also be simulated, while it should be straightforward to reset an environment, and specify the arrangement of objects (for example using 3D modelling software).