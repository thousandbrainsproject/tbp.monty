---
title: Generalize Voting to Associative Connections
---
Currently, voting relies on all learning modules sharing the same object ID for any given object, as a form of supervised learning signal. Thanks to this, they can vote on this particular ID when communicating with one-another.

However, in the setting of unsupervised learning, the object ID that is associated with any given model is unique to the parent LM. As such, we need to organically learn the mapping between the object IDs that occur together across different LMs, such that voting can function without any supervised learning signal.

This challenge relates to [Use Pose for Voting](./use-pose-for-voting.md), where we would like to also vote on the poses of objects.