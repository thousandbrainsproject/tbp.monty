---
title: Coordinate Systems and Conventions
---

# Coordinate Systems and Conventions

There are many ways to specify a coordinate system in three dimensions, and it's notoriously easy to make mental mistakes or coding errors when working in 3D. This is especially true when converting between coordinate systems, which we often need to do when working with cameras, actuators, object models, simulators, visualization libraries, etc.

This document describes Monty's adopted 3D conventions and provides a few common conversion formulas to map between them. While there is a bit of discussion and background, the primary aim is to give users and developers a quick reference and cut through the confusion.

### Cartesian Conventions

Let's say you are given the xyz coordinates (1, 0, 0), which was measured in meters relative to your body. Is the point 1 meter directly in front of you? Or is it above you, below you, or perhaps off to your right? Without additional metadata, we have no way to know.

More specifically, we need we need to know which _physical direction_ each axis points down, where directions are
 - Left or Right
 - Up or Down
 - Forward or Backward

An axis-direction mapping, such as (X = Right, Y = Up, Z = Backward), is known as an [axes convention](https://en.wikipedia.org/wiki/Axes_conventions).

**Monty uses the Right-Up-Backward (RUB) axes convention**. More specifially,
 - The x-axis points to the right.
 - The y-axis points straight up.
 - The z-axis points backwards (i.e. directly behind you).

Note that when we say that an axis "points" in some direction, the positively-oriented direction of the axis is implied. When we're being explicit about axis orientation, we write +X, -X, +Y, etc.

**Be advised**: Since +Z points backwards, -Z points straight ahead. Most people tend to associate "forward" with "positive", which is probably why flipping z-values is a common mistake.



### Spherical Coordinates

### Mapping Between Cartesian and Spherical Coordinates

### Example: 2-DOF Gimbal Distant Agent

### Rotations
