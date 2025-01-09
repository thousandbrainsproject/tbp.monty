---
title: Reinitialize Hypotheses When Starting to Recognize a new Object
---

We have recently implemented a method to detect when we have moved on to a new object based on significant changes in the accumulated evidence values for hypotheses. Integrating this method into the LMs is still in progress, but once complete, we would like to complement it with a process to reinitialize the evidence scores within the learning module. That way, when the LM detects it is on a new object, it can cleanly estimate what this new object might be.

Eventually this could be complemented with top-down feedback from a higher-level LM modeling a scene or compositional object. In this case, the high-level LM biases the evidence values initialized in the low-level LM, based on what object should be present there according to the higher-level LM's model. Improvements here could also interact with the tasks of [Re-Anchor Hypotheses](../learning-module-improvements/re-anchor-hypotheses.md), and [Use Better Priors for Hypothesis Initialization](../learning-module-improvements/use-better-priors-for-hypothesis-initialization.md).
