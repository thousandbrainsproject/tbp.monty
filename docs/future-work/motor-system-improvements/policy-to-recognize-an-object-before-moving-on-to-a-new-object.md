---
title: Policy to Recognize an Object Before Moving onto a New Object
description: Use the SM and/or LM goal generator to remain on an object until it is recognized (e.g. using basic image segmentation)
rfc: required
estimated-scope: large
improved-metric: multi-object, compositional, goal-policy
output-type: prototype, monty-feature, PR, RFC, experiments, analysis
skills: research, python, monty
contributor: nielsleadholm
status: paused
---

When there are multiple objects in the world (including different parts of a compositional object), it is beneficial to recognize the object currently observed (i.e. converge to high confidence) before moving onto a new object.

Such a policy could have different approaches, such as moving back if an LM believes it has moved onto a new object (reactive), or using the model of the most-likely-hypothesis to try to stay on the object (pro-active), or a mixture of these. In either case, these would be instances of model-based policies.

Initial prototyping has tested using a sudden decay of evidence for the most likely hypothesis as an indicator to move back into the object. However, this work was put on hold a while ago and needs to be picked up again.

Recently we have also added the ability for sensor modules to output goals. We could therefor also use model-free signals, such as goals based on segmentation of the input image, as a way to stay in a region with similar depth or features. This would help especially early on when the LMs don't have any strong hypotheses yet.
