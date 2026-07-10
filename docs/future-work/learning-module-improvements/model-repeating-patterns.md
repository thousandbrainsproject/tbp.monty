---
title: Model Repeating Patterns
description: Come up with, implement, and test a way to model and recognize repeating patterns efficiently.
rfc: required
estimated-scope: unknown
improved-metric: learning, numsteps, features-and-morphology
output-type: RFC, experiments, analysis, PR, publication
skills: python, research, monty-beginner
contributor: 
status: open
---

Repeating patterns are somewhere between structured objects that require a reference frame (learned in LMs) and primitive features that can be extracted by sensor modules directly. Sensor modules can extract features that go beyond simple color values, such as textures (explored in the LBP work mentioned [here](../sensor-module-improvements/extract-better-features.md)). However, they don't learn structured models like LMs do and would therefore struggle with more structured patterns. 

Think of patterns on bed sheets or on a carpet, where you have an object or structured pattern that repeats many times. At the moment, we would need to learn or infer the pattern over and over again for each location where it applies. We could have a lower-level LM recognize the repeating aspect of the pattern and assign it to the different locations on the object where it repeats. However, the lower-level LM would have no concept of the pattern repeating and would have to infer it every time again (unless the higher LM helps with top-down connections). 

A better representation could be that the models RF wraps around itself. When incoming movement information would take the LM outside of the model's RF, it instead integrates the movement in such a way that it predicts coming out on the other side of the pattern again. This is similar to how in some video games you can move off on the left side of the screen and reappear on the right side again. This would implement a toroidal space where you never move out of the model's RF, related to how grid cells represent space.

This item is still a but unclear conceptually so it would be good to outline an [RFC](../../contributing/request-for-comments-rfc.md) first, outlining the approach and benefits as well as whether it could be solved using other existing mechanisms in Monty.
