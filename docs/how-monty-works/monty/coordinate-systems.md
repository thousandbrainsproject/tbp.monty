---
title: Coordinate Systems and Conventions
---

# Coordinate Systems and Conventions

There are many ways to specify a coordinate system in three dimensions, and it's notoriously easy to make mental mistakes or coding errors when working in 3D. This is especially true when converting between coordinate systems, which we often need to do when working with cameras, actuators, object models, simulators, visualization libraries, etc.

This document describes Monty's adopted 3D conventions and provides a few common conversion formulas to map between them. While there is a bit of discussion and background, the primary aim is to give users and developers a quick reference and cut through the confusion.

### Axes Conventions

Monty adopts the _right-up-backward_ [axes convention](https://en.wikipedia.org/wiki/Axes_conventions), meaning that
 - The x-axis points to the right.
 - The y-axis points straight up.
 - The z-axis points backwards (i.e. directly behind you).

![](../../figures/how-monty-works/coordinate_conventions.png)

For angular coordinates, Monty uses the _forward-aligned zero-angle_ convention. More specifically,
  - Azimuth/Yaw (theta) is measured away from the forward axis.
  - Elevation/Pitch (phi) is measured upw from the horizontal plane.

Note positive yaw corresponds to turning leftward, and positive pitch corresponds to tilting up.

Converting between Cartesian and angular coordinates is common, especially when handling sensor data or actions. The specifics of the conversion formulae depend upon both the Cartesian and spherical axes conventions in use, and it's easy to make mistakes here. To avoid common pitfalls, we provide the following formulae.

Again, these formualae are specific to the right-up-backward _and_ forward-aligned zero-angle convention adopted by Monty.


### Example: 2-DOF Gimbal Distant Agent

### Rotations
