---
title: Model-Free Repositioning for Natural Views
description: Let sensor modules propose the agent's pose so that an object is viewed from a cleaner or canonical view.
rfc: optional
estimated-scope: medium
improved-metric: accuracy, speed, learning, goal-policy
output-type: prototype, monty-feature, PR
skills: python, computer-vision
contributor: 
status: open
---

**Repositioning to obtain face-on views of planar surfaces.**
When viewing an object's surface from a steeply oblique angle, sensor modules are given data that is effectively compressed along an axis. This means that features, such as the color or principal curvatures, are necessarily computed over a larger physical area, and this skew is often along only one physical axis. In other words, features can be view-dependent, which introduces extra difficulty when comparing observed features to features stored in learned models. Furthermore, viewing a surface from an oblique angle makes it more difficult to evenly saccade across the surface, thereby reducing the effective coverage of an object's faces.

This can be addressed by allowing sensor modules to propose agent poses. In particular, an SM may detect that it is viewing a surface from an oblique angle based on its observed surface normals. The SM can then propose a goal that would place the agent and sensor(s) such that they view the surface "face-on". Here, "face-on" means viewing directly down the surface normal.

![Left: Monty sees the face of a cube from an oblique angle. Right: After repositioning according to the computed surface normal, Monty obtains a face-on view of the cube's face. Bottom: The view-finder image from each pose.](../../figures/future-work/model_free_face_on_view.png#width=600px)


**Rotating the sensor/agent around the forward axis for consistent views.**
If the model-free system can consistently view an object with the same apparent orientation (i.e., an orientation in retinotopic space), then in principle we should be able to seed fewer hypotheses during inference. In other words, we may be able to reduce the size of the hypothesis search space by leveraging a preference to view objects according to their physical characteristics (e.g., a focus of mass along a certain axis).

For intuition, consider how much more difficult it is to recognize a face when it isn't upright. Similarly, it's much more difficult to read a line of text when it is tilted at an angle. In both cases, we'll often subconsciously tilt our heads or rotate a page to get a more "natural" view.

Note that the above describes an entirely model-free system for obtaining a consistent view, and it may or may not match the object's typical appearance in the world (i.e., its canonical view).

![Top: The sensor is face-on to a cube with a logo but rolled by φ about its viewing axis (left); by taking the edges of the cube and/or the logo into account, we can roll our sensor/agent such that the object appears upright (right). Bottom: When the apparent orientation is unknown, each model location must be initialized with hypotheses covering many rotations (left); if the object is reliably seen in one or two orientations, we can bias our hypotheses towards those (right).](../../figures/future-work/model_free_forward_axis_roll.png#width=600px)

