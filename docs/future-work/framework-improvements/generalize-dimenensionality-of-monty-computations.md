---
title: Generalize Dimensionality of Monty's Computations
description: We are developing LMs that support 1D and 2D reference frames, but our computations all assume more expensive 3D representations.
rfc: optional
estimated-scope: unknown
improved-metric: speed
output-type: analysis, PR
skills: python
contributor: 
status: open
---

Recent work with [sensor modules has enabled reference frames that model 2D space](../sensor-module-improvements/surface-sm.md), and we hope to make [a similar advance with 1D space](../sensor-module-improvements/one-d-sensor-module.md). From a computational efficiency perspective however, the LMs that receive these inputs still assume that space is 3D. This restricts our ability to see greater computational benefits from using low-dimension representations. This Future Work item is therefore about refactoring Monty representations, both those sent between LMs, and the reference frames used within LMs, to be of lower dimensionality where this is consistent with the sensory input they receive.