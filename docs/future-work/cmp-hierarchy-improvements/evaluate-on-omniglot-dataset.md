---
title: Evaluate on Omniglot Dataset
description: Test the ability to model compositional objects on the Omniglot dataset.
rfc: optional
estimated-scope: medium
improved-metric: compositional, generalization
output-type: experiments, analysis, publication
skills: python, monty, research
contributor: vkakerbeck
status: paused
---

We have added the Omniglot dataset as an environment to test Monty on (to see [this tutorial](https://thousandbrainsproject.readme.io/docs/using-monty-in-a-custom-application#example-1-omniglot) to see how to run those experiments)

However, Monty's generalization ability on this dataset is not great, as the letters can vary widely in morphology if modeled as monolithic models. We expect this to be better when modeling the letters as compositional objects made up of individual strokes (modeled in the lower level) arranged relative to each other (modeled at higher level). See [this research meeting](https://youtu.be/-qPfBrTVoks?si=CEzb1uuZMXvgIJnJ&t=3782) for a discussion of the general issue and how hierarchy can solve it (starting at 1:03:00).

To test this, one needs to adapt the current Omniglot experiment setup such that during learning, Monty first sees the individual strokes in isolation, before it learns compositional objects of them. 

For some more discussion by our research team on the topic, see [this recent research meeting](https://youtu.be/ElO4SzHrg5g?si=3m6Q0ODndKHx-y_r&t=5465) (Omniglot discussion starting at 01:31:00)