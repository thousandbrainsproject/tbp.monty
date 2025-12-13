---
title: Less Dependency on First Observation
description: Currently we only initialize hypotheses on the first step of an episode which make us dependent on the first observation having low noise .
rfc: 
estimated-scope: large
improved-metric: noise, multi-object
output-type: prototype, monty-feature, PR, analysis
skills: python, research, monty
contributor: ramyamounir
status: in-progress
---

In the current implementation, Monty initializes its entire hypothesis space during the first step of an episode.
This design makes the system highly dependent on the quality of that initial observation.
Any sensor noise, occlusion, or partial view influences all hypotheses generated for the remainder of the episode.
The model also becomes tightly anchored to the pose sensed from this single observation, which limits its ability to adapt when Monty moves onto a different objects, e.g., during compositional model inference, or when the pose of the object changes over time, e.g., during a behavior.

A more flexible approach is to remove the special sampling procedure of the first step entirely.
Hypotheses should be initialized, expanded, or replaced at any point during episode, based on evidence scores rather than episode boundaries.

> [!NOTE]
> See the [feat.dynamic_resizing](https://github.com/thousandbrainsproject/feat.dynamic_resizing) feature branch to see the current prototype. This prototype validated that removing the special sampling procedure at the beginning of the episode and allowing Monty to sample new hypotheses during the episode improves Monty's recognition accuracy and reduces the pose error. This feature is planned to be integrated into Monty.
