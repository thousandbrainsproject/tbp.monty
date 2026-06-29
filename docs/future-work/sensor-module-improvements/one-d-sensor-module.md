---
title: One-D Sensor Module
description: Sensor module that converts 3D movement in space to 1D movement along a principal direction.
rfc: optional
estimated-scope: medium
improved-metric: features-and-morphology, transfer, deformations, generalization
output-type: prototype, monty-feature, PR, publication
skills: python, research, monty, computer-vision
contributor: 
status: open
---

As Monty's models are fixed in a metric space, it can only tolerate slight distortions of objects. However, some objects such as T-Shirts or cables can distort a lot and are still recognizable to us. It is unreasonable to assume that we have learned separate models for all the possible ways a T-shirt could be crumpled or a cable could be tangled.

One possible solution to this problem is to project the incoming movement to a learning module into a lower dimension that is unaffected by the distortion. This idea is realized in the [2D sensor module](../../how-monty-works/sensor-module/two-d-sensor-module.md), which projects movement in 3D space into 2D movement along a surface. This allows Monty to learn about an object, such as a logo, on one surface (e.g. on a mug) and recognize it projected onto another surface (e.g. on a baseball). This is possible because movement along a surface (up, down, left, right, no depth) is (mostly) invariant to curvature of a surface.

We want to extend this concept to even more extreme distortions by projecting movement into 1D space. One concrete example we imagine is recognizing cables or strings in various configurations. The movement in this case should be mapped onto one dimension: moving up or moving down the cable. It can be estimated using local features (curvature of the cable). In addition to transforming the incoming movement, Monty will also need to leverage a specific policy that lets its sensor follow this principal curvature direction. Just like with the 2D SM, we need to move smoothly along the cable instead of taking random jumps around it for this to work - note this is no different from a human, who would not be able to locate their position on a tangled cable without slowly moving a sensor (eye or finger) along it. 

For a more in-depth discussion of this idea, see this research meeting (starting at 45:30 minutes):
[New Insights Around Deformations](https://youtu.be/cJLH-C3EYxI?si=EJNUJzURbcIWJgk8&t=2728)