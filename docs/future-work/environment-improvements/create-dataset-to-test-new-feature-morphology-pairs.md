---
title: Create Dataset to Test new Feature-Morphology Pairs
description: Create a mesh dataset where we can test learning specific object-texture pairs and then infer arbitrary combinations.
rfc: optional
estimated-scope: small
improved-metric: generalization, transfer, features-and-morphology
output-type: experiments, mesh-models
skills: 3d-modeling
contributor: 
status: open
---

Since the addition of the [2D sensor module](../../how-monty-works/sensor-module/two-d-sensor-module.md), Monty has the ability to learn about object textures independently of the morphology they are on. We have evaluated this performance on a small, handcrafted dataset of logos on different shaped objects. However, it would be great to test this on a larger scale dataset.

Tackling this item would involve setting up Monty to run object recognition on a larger dataset with objects of different shapes and textures, where the same pattern can appear on many objects and vice versa. This could leverage an existing 3D mesh dataset or include creating a new, custom dataset.