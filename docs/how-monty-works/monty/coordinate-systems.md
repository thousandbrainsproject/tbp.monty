---
title: Coordinate Systems
---
<!-- There are many ways to specify a coordinate system in three dimensions, and it's notoriously easy to make mental mistakes or coding errors when working in 3D. This is especially true when converting between coordinate systems, which we often need to do when working with cameras, actuators, object models, simulators, visualization libraries, etc.

This document describes Monty's adopted 3D conventions and provides a few common conversion formulas to map between them. While there is a bit of discussion and background, the primary aim is to give users and developers a quick reference and cut through the confusion. -->

### Conventions and Conversions

![](../../figures/how-monty-works/coordinate_conventions.png)

Monty uses the _right-up-backward_ [axes convention](https://en.wikipedia.org/wiki/Axes_conventions), meaning that
 - The x-axis points to the right.
 - The y-axis points up.
 - The z-axis points backwards.

Note that the forward direction corresponds to the _negative z-axis_. While users with a background in 3D graphics (esp. OpenGL) may be accustomed to this convention, but many find it unintuitive at first.

For spherical coordinates, we use the convention that places azimuth = 0 and elevation = 0 down the forward axis. More explicitly,
  - Azimuth (or Yaw) is measured away from the forward axis.
  - Elevation (or Pitch) is measured up from the horizontal plane.

While not uncommon in 3D graphics, this convention does differ from the [physics convention](https://en.wikipedia.org/wiki/Spherical_coordinate_system). As such, it's important to take care when interpreting angles or mapping between Cartesian and spherical coordinates. Should you need to handle conversions between Cartesian and spherical coordinates in your code, the following convention-compliant conversion formulae are provided as a reference:

<img src="../../figures/how-monty-works/cartesian_spherical_conversions.png" width="50%">
<!-- ![](../../figures/how-monty-works/cartesian_spherical_conversions.png#width=400px) -->


<!-- ### Example: Distant Agent -->

