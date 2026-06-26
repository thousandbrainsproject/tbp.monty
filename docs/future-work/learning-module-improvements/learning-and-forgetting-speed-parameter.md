---
title: Learning and Forgetting Speed Parameter
description: Add a learning speed parameter to object models in LMs such that some LMs learn more temporary models and others more stable ones.
rfc: required
estimated-scope: unknown
improved-metric: learning, compositional
output-type: RFC, prototype, monty-feature, PR, publication
skills: python, research, monty
contributor: 
status: open
---

At the moment, all our learning modules learn at the same speed (instantaneously integrating information into their models). However, as Monty becomes more hierarchical, it will likely be necessary to have adjustable learning speeds where some LMs learn slower and others faster. For instance, we want to be able to quickly learn a representation of the current environment to be able to interact with it (and also quickly forget it again). This is similar to what happens in the hippocampus and is what Monty can currently do (without the forgetting part). However, we will also need more stable models that don't change quickly and are remembered for a long time. This is important since higher-level LMs rely on the output of lower-level LMs. If that output is changing faster than the higher level models can adapt, they will fail to learn useful compositional models.

The idea here is to leverage the existing [constrained object models](../../how-monty-works/learning-module/object-models.md#object-models) and add a learning speed parameter into it. The concrete way in which this parameter could work would need to be outlined in an [RFC](../../contributing/request-for-comments-rfc.md). Ideally, it should also incorporate forgetting.