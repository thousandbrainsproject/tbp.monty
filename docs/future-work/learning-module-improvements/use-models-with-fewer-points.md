---
title: Use Models with Fewer Points
description: Update learning algorithm, sensor module, and policy to learn sparser models.
rfc: 
estimated-scope: large
improved-metric: speed, generalization, numsteps
output-type: experiments, analysis, PR
skills: python, research, monty
contributor: ramyamounir, scottcanoe
status: in-progress
---

This task relates to the on-going implementation of hierarchically arranged LMs. As these become available, it should become possible to decompose objects into simpler sub-object components, which in turn will enable LMs to model objects with significantly fewer points than the ~2,000 per object currently used.