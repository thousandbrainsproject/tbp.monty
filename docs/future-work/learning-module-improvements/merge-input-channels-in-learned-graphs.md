---
title: Merge Input Channels in Learned Graphs
description: Store a single graph per object in the learning module's memory, where each node can carry features from any input channel, instead of one graph per input channel.
rfc: required
estimated-scope: large
improved-metric: compositional
output-type: RFC, PR, monty-feature
skills: python, refactoring, monty-advanced
contributor: 
status: open
---

A learning module (LM) stores a separate graph per input channel for each object it learns.
An LM that receives input from multiple different channels, e.g., sensor module (SM) or a lower-level LM (LL-LM), learns multiple independent graphs of the same object, each storing the features received on that input channel.
We have already removed the per-channel split from the hypothesis space ([PR 851](https://github.com/thousandbrainsproject/tbp.monty/pull/851)), so that an LM maintains one hypothesis space per object and sums evidence across channels.
The learned graphs, however, are still split by channel.
Features observed through different channels at the same location are never stored together on one node.

The goal of this item is to store one graph per object, where each node can carry features from any input channel.
Since [PR 1051](https://github.com/thousandbrainsproject/tbp.monty/pull/1051), features received on a step from any input channel are stored at the single buffer location the SM sensed on that step, so the per-channel graphs already share their node locations.
Merging these graphs is a matter of attaching each channel's features to the node at that location.

A single graph also makes nearest neighbor search computationally cheaper.
Each per-channel graph currently builds its own KD-Tree over its node locations, and matching queries every tree separately.
With one graph there is one KD-Tree per object, and a single query returns the features from every channel at the nodes near the query location.
