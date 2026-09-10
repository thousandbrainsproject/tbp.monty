---
title: Explore Alternative Internal Representations for LMs 
description: Test different algorithms or representations inside of LMs, including alternative reference frame representations.
rfc: optional
estimated-scope: unknown
improved-metric: speed, pose, accuracy, noise
output-type: experiments, analysis, PR
skills: python, research, monty-advanced
status: evergreen
---

There is significant scope for custom learning modules in Monty. In particular, learning modules can take a variety of forms, so long as their input and output channels adhere to the Cortical Messaging Protocol, and that they model objects using reference frames. However, exactly how a "reference frame" is implemented is not specified.

Currently, our main approach is to use explicit graphs in Cartesian space, with evidence values accumulated, somewhat analogous to a particle filter. An example of an alternative approach would be using grid-cell modules to model reference frames, or to expand and modify our current use of voxels with [alternative tessellation methods](./evaluate-alternative-tessellation-methods-for-3d-space.md).

As LMs are the core component of Monty, any suggested changes should be clearly motivated by a particular problem or desired outcome. For example, anyone is welcome to create a fork of Monty that implements a more biologically plausible implementation of reference frames. However, such an implementation would never become a part of `tbp.monty` unless it demonstrates measurable advantages. This applies to all algorithmic changes to Monty, but we emphasize it here given the general interest we have received in making Monty's reference frames more biologically plausible.