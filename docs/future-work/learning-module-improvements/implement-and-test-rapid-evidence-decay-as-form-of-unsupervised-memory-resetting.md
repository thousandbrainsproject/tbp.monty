---
title: Implement and Test Rapid Evidence Decay as Form of Unsupervised Memory Resetting
---

In a natural, unsupervised setting, the object that a Monty system is observing will change from time to time. Currently, Monty's internal representations of objects is only explicitly reset by the experimenter, for example at the start of an episode of training.

We would like to see if we can achieve reasonable performance when objects change (including during learning) by having a shorter memory horizon that rapidly decays. Assuming policies are sufficiently efficient in their exploration of objects, this should enable us to effectively determine whether we are still on the same object, on a different (but known) object, or on an entirely new object. This can subsequently inform changes such as switching to an exploration-focused policy (see [Implement Switching Between Learning and Inference-Focused Policies](../motor-system-improvements/implement-switching-between-learning-and-inference-focused-policies.md)).

Note that we already have the `past_weight` and `present_weight` parameters, which can be used for this approach. As such, the main task is to set up experiments where objects are switched out without resetting the LMs evidence values, and then evaluate the performance of the system.

If this fails to achieve the results we hope for, we might add a mechanism to explicitly reset evidence values when an LM believes it has moved on to a new object. In particular, we have implemented a method to detect when we have moved on to a new object based on significant changes in the accumulated evidence values for hypotheses. Integrating this method into the LMs is still in progress, but once complete, we would like to complement it with a process to reinitialize the evidence scores within the learning module. That way, when the LM detects it is on a new object, it can cleanly estimate what this new object might be.

Eventually this could be complemented with top-down feedback from a higher-level LM modeling a scene or compositional object. In this case, the high-level LM biases the evidence values initialized in the low-level LM, based on what object should be present there according to the higher-level LM's model. Improvements here could also interact with the tasks of [Re-Anchor Hypotheses](../learning-module-improvements/re-anchor-hypotheses.md), and [Use Better Priors for Hypothesis Initialization](../learning-module-improvements/use-better-priors-for-hypothesis-initialization.md).
