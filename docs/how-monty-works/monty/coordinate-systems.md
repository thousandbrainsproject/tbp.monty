---
title: Coordinate Systems and Conventions
---

# Coordinate Systems and Conventions

There are many ways to specify a coordinate system in three dimensions, and it's notoriously easy to make mental mistakes or coding errors when working in 3D. This is especially true when converting between coordinate systems, which we often need to do when working with cameras, actuators, object models, simulators, visualization libraries, etc.

This document describes Monty's adopted 3D conventions and provides a few common conversion formulas to map between them. While there is a bit of discussion and background, the primary aim is to give users and developers a quick reference and cut through the confusion.

### Cartesian Coordinates

Let's say you are given the point (1, 0, 0), which you are told are (x, y, z) coordinates relative to your location. Is the point (1, 0, 0) directly in front of you? Or is it above you, or maybe to the right? Without more context, there's no way to say one way or the other.

Clearly, the main piece of context that we need is the direction of the x-axis. There are six options:
 - Left or Right
 - Up or Down
 - Forward or Backward

Let's say the x-axis points to the right. That leaves four possibilities for the y-axis: up, down, forward, and backward. If, for example, we know that the y-axis points up, then the z-axis must point either forward or backward.

A set of axis-direction associations is known as an [axes convention](https://en.wikipedia.org/wiki/Axes_conventions).

In the same way that physical units, such as meters,

 of which there are 48 (6 * 4 * 2).


 
 and its especially important to remain aware of the conventions in place when


 and its critically important to be mindful of the convention in play. This is especially the


Note that when we say that an axis "points" in some direction, we're implicitly referring to its _positive_ direction. When we're being explicit about an axis' direction, we typically write +X or -X, using the x-axis as an example.

**Important**: The backwards-pointing z-axis often trips people up. Naturally, when +Z points backwards, the negative z-axis (-Z) points directly ahead. Anecdotally, most people seem to have a strong association between "forward" and "positive". Flipped z-coordinates are a common cause of error, so beware of this tendency!


The RIGHT-UP-BACKWARD (RUB) convention is also used by OpenGL and the Habitat simulator. It's important to

### Spherical Coordinates

### Mapping Between Cartesian and Spherical Coordinates

### Example: 2-DOF Gimbal Distant Agent

### Rotations
