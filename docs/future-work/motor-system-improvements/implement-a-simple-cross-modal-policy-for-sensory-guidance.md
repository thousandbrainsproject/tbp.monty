---
title: Implement a Simple Cross-Modal Policy for Sensory Guidance
---

Once we have infrastructure support for multiple agents that move independently (see [Add Infrastructure for Multiple Agents that Move Independently](../framework-improvements/add-infrastructure-for-multiple-agents-that-move-independently.md)), we would like to implement a simple cross-modal policy for sensory guidance.

In particular, we can imagine a distant-agent rapidly saccading across a scene, observing objects of interest (see also [Implement Efficient Saccades](implement-efficient-saccades-driven-by-model-free-and-model-based-signals.md)). When an object is observed, the LM associated with the distant-agent could send a goal-state (either directly or via an actuator-modeling LM) that results in the surface agent moving to that object and then beginning to explore it in detail.

Such a task would be relatively simple, while serving as a verification of a variety of components in the Cortical Messaging Protocol, such as:
- Recruiting agents that are not directly associated with the current LM, using goal-states (e.g. here we are recruiting the surface agent, rather than the distant agent).
- Coordination of multiple agents (the surface agent and distant agent might each inform areas of interest for the other to explore).
- Multi-modal voting (due to limited policies, voting has so far been limited to within-modality settings, although it supports cross-modal communication).