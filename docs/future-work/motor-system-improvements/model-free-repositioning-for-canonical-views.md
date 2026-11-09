---
title: Model-Free Repositioning for Canonical Views
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
When viewing an object's surface from a steeply oblique angle, sensor modules are given data that is effectively compressed along an axis. This means that features, such as the color or principle curvatures, are necessarily computed over a larger physical area, and this skew is often along only one physical axis. In other words, features can be view-dependent, which introduces extra difficulty when comparing observed features to features stored in learned models. Furthermore, viewing a surface from an oblique angle makes it more difficult to evenly saccade across the surface, thereby reducing the effective coverage of an object's faces.

This can be addressed by allowing sensor modules to propose agent poses. In particular, an SM may detect that it is viewing a surface from an oblique angle based on its observed surface normals. The SM can then propose a goal that would place the agent and sensor(s) such that it will view the surface "face-on". Here, "face-on" means viewing directly down the surface normal.
![Left: Monty sees the face of a cube from an oblique angle. Right: After repositioning according to the computed surface normal, Monty obtains a face-on view of the cube's face. Bottom rows The bottom row shows the view finder image from each pose.](../../figures/future-work/model_free_face_on_view.png#width=600px)


**Rotating around the forward to place "oriented" objects (e.g., tall skinny items) in a "canonical view"**.
If the model-free system can consistently view an object with the same apparent orientation (i.e., an orientation in retinotopic space), then in principle we should be able to seed fewer hypothesis during inference.

![Top: the sensor is face-on to a cube but rolled by φ about its viewing axis (left); after rolling back to the vertical/canonical orientation, the object appears upright (right). Bottom: when the apparent orientation is unknown, each model location must be initialized with hypotheses covering many rotations (left); if the object is reliably seen upright, hypotheses can be biased towards vertical orientations (right).](../../figures/future-work/model_free_forward_axis_roll.png#width=600px)


**Possible Follow-Up: Replacing/Emulating Positioning Procedures.**
If the model-free repositioning behaviors are sufficiently powerful, then there may be an opportunity to use these features in place of positioning procedures used in our experiments. The goal isn't necessarily to eliminate positioning procedures -- they are still a valuable tool for initializing expriments. But there may be value in de-duplicating repositioning behaviors, especially if we expect them to be useful in platform code.
