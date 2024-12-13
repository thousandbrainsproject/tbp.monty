---
title: Add Top-Down Connections
---

In Monty systems, low-level LMs project to high-level LMs, where this projection occurs if their sensory receptive fields are co-aligned. Hierarchical connections should be able to learn a mapping between objects represented at these low-level LMs, and objects represented in the high-level LMs that frequently co-occur.

For example, a high-level LM of a dinner-set might have learned that the fork is present at a particular location in its internal reference frame. When at that location, it would therefore predict that the low-level LM should be sensing a fork, enabling the perception of a fork in the low-level LM even when there is a degree of noise or other source of uncertainty in the low-level LM's representation.

In the brain, these top-down projections correspond to L6 to L1 connections, where the synapses at L1 would support predictions about object ID. However, these projections also form local synapses en-route through the L6 layer of the lower-level cortical column. In a Monty LM, this would correspond to the top-down connection predicting not just the object that the low-level LM should be sensing, but also the specific location that it should be sensing it at. This could be complemented with predicting a particular pose of the low-level object (see [Use Better Priors for Hypothesis Initialization](../learning-module-improvements/use-better-hypothesis-priors.md)).