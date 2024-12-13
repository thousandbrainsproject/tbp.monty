---
title: Add Infrastructure for Multiple Agents that Move Independently
---

Currently, Monty's infrastructure only supports a single agent that moves around the scene, where that agent can be associated with a plurality of sensors and LMs. We would like to add support for multiple agents that move independently.

For example, a hand-like surface-agent might explore the surface of an object, where each one of its "fingers" can move in a semi-independent manner. At the same time, a distant-agent might observe the object, saccading across its surface independent of the surface agent. At other times they might coordinate, such that they perceive the same location on an object at the same time, which would be useful while voting connections are still being learned (see [Generalize Voting to Associative Connections](../voting-improvements/generalize-voting-to-associative-connections.md)).

An example of a first task that could make use of this infrastructure is [Implement a Simple Cross-Modal Policy for Sensory Guidance](../motor-system-improvements/implement-a-simple-cross-modal-policy-for-sensory-guidance.md).

It's also worth noting that we would like to move towards the concept of "motor modules" in the code-base, i.e. a plurarity of motor modules that convert from CMP-compliant goal states to non-CMP actuator changes. This would be a shift from the singular "motor system" that we currently have.