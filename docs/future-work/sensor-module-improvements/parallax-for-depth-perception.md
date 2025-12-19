---
title: Parallax for Depth Perception
description: Implement a sensor module that can estimate depth based on binocular or motion parallax, reducing the need for a depth sensor.
rfc: optional
estimated-scope: medium
improved-metric: real-world
output-type: prototype, PR, monty-feature
skills: python, computer-vision
contributor: 
status: open
---

Monty relies a lot on estimated depth for figuring out how it's vision sensor patch has moved in 3D space. Currently we use a depth sensor in our simulator for this, but many real world applications would not have access to such a sensor. Having the option to estimate depth in other ways would improve Monty's applicability in places where only standard RGB cameras are available.

For this task, one could test incorporating existing computer vision techniques to estimate depth based on parallax into a sensor module. Extra points if the technique is motivated by how biological systems solve this!