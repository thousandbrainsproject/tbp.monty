---
title: Sample Offspring Hypotheses
description: Sample new hypotheses that are similar to the existing top hypotheses.
rfc: optional
estimated-scope: medium
improved-metric: speed, accuracy, pose
output-type: prototype, monty-feature, PR
skills: python, research
contributor: 
status: open
---

Monty currently samples new hypotheses using only the observed features from the sensor data and the points stored in the model.
This produces hypothesis spaces that do not benefit from Monty’s own evolving beliefs.
As a result, Monty cannot refine promising hypotheses through targeted exploration of nearby poses.
A useful extension is to introduce offspring hypotheses.
Offspring hypotheses are sampled around the current highest scoring hypotheses.
This approach resembles refinement strategies used in particle filters, and in SLAM systems that rely on particle filtering, where the most promising pose estimates generate additional candidate samples.
By sampling around these top contenders, Monty can progressively refine its estimate of the object’s pose and sensor position as more evidence accumulates.

This mechanism is particularly effective when the object is stable or stationary, because the refinement process can converge toward a true stationary pose with increasing precision.
One can imagine this sampling being controlled in a goal-directed manner by a learning module, such that offspring hypotheses are leveraged when it is particularly important to have a highly accurate hypothesis.
However, offspring sampling alone is not enough for general adaptability.
Monty must also be able to sample hypotheses informed directly by the latest observation, so that it can detect abrupt changes in object ID and avoid over committing to a single hypotheses cluster.
