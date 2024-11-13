---
title: Common Issues (And how to Fix Them)
---
# General Tips and Debugging

Below we highlight a few issues that often crop up and can present problems when they are not otherwise apparent:

## Quaternions

Be aware that in Numpy, and in the saved CSV result files, quaternions follow the wxyz format, where "w" is the real component. Thus the identity rotation would be [1, 0, 0, 0]. In contrast however, Scipy.Rotation expects them to be in xyzw format. When operating with quaternions, it is therefore important to be aware of what format you should be using for the particular setting.

## XYZ Conventions

Note that in Habitat (and therefore the Monty code-base), the "z" direction is positive in the direction coming out of the screen, while "y" is the "up" direction. "x" is positive pointing to the right, again if you are facing the screen.

## Environment vs. Monty's Internal Rotation

Note that the rotation that learning modules store in their Most-Likely Hypothesis (MLH) is the rotation required to transform a feature (such as a point-normal) to match the feature on the object in the environment. As such, it is the _inverse_ of the actual orientation of the object in the environment.

## Sensor Updates in Habitat

Note that sensor-based actions (such as set_sensor_pose), update \_all_ the sensors associated with that agent. For example, if the view-finder and main patch sensor associated with an agent were at different locations relative to one another, but set-sensor-pose sets a new absolute location (e.g. 0, 0, 0), both sensors will now have this location, and they will lose the relative offset they had before. While the effect will depend on whether the action is in relative or absolute coordinates, the modification of all sensors associated with an agent is an inherent property of Habitat.