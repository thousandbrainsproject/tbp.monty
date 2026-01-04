# 2D Sensor Module (2D SM) - Outline

## Purpose

The 2D sensor module (2D SM) is designed to extract information about the surface of an object. It serves two main purposes:

1. Extract the 2D pose of features on the surface of an object (not principal curvature), i.e. the orientation of edges. This is already implemented in `two_d_pose_sm.py` and `edge_detection_utils.py`.
2. Convert movement in 3D space to movement in 2D surface space.

As this SM returns both movement and poses in 2D space, the receiving learning module (LM) will learn a 2D model of the object surface. The LM however, should not require any customization to process the data.

## Implementation

### Extracting 2D Pose
(Skipped as this is already implemented)

### Extracting 2D Movement

The sensor might be moving in 3D space; however, we want to return how it is moving along the surface of an object. Imagine a finger tracing the outside of a mug. The finger is moving in 3D space; however, the output of this custom SM should just indicate how it moves along the surface (i.e., up, down, left, right...).

The 2D movement can be calculated as the movement in 3D space in the y-z plane defined by the surface normal. This means the 2D SM still needs to extract the 3D pose (or at least the surface normal). It then returns the part of the movement vector that is in the plane orthogonal to the surface normal.

The 2D SM would be learning something similar to that of a UV map from the field of texture mapping. 

When given a rolled cylinder object (think of a paper rolled into a cylinder), the 2D SM should learn a flat rectangular surface while the current `HabitatSM` class would learn the cylindrical object. 

When constructing the `State` class in the 2D SM module, we should probably pass 0 for z-coordinate so locations are still in 3D coordinates (so existing LM doesn't need to be modified). 

## Caveats and Questions

1. Path integration in the 2D space will only work locally. This might be okay but we should be aware of that:
We can only estimate the 2D movement accurately if the 3D movement is small (i.e. if we move from the cup to the handle, this won’t work well). We basically need to smoothly trace the surface with the sensor.
2. There is no representation of ending up back at the same place we started when the sensor does a full circle around the cup.
3. Let's say we have implemented the 2D movement and learned a 2D model, e.g. flat rectangular surface for a cylindrical object. Let's say this surface has some a word written and we have stored 2D pose (edge) information of these words. Will the learned model be different if we took different paths? This is important to think as during inference, we may not explore the entire object, and would take different paths from pretraining, but still want to detect the same edges for object recognition and object orientation. If the object during inference is identical and in the same pose, will we still detect edges in the same orientation? (otherwise it may miscalculate the object orientation)



