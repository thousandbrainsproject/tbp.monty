---
title: SM for RGBD data
---

# Sensor Modules for RGBD Data
Most of our experiments rely on RGBD cameras as sensors. `CameraSM`, `TwoDSensorModule`, and `Probe` expect RGBD images as input. We use the depth channel in combination with proprioceptive information (camera location in space) to determine the location of the patch in the world. `TwoDSensorModule` uses the same kind of input to learn a flattened 2D texture layout on a 3D surface; see [TwoDSensorModule](two-d-sensor-module.md) for details.

## Surface Normals and Principal Curvatures
Each Sensor Module needs to extract a pose from the sensory input it receives. This pose can be defined by the _surface normal_ and the two _principal curvature_ vectors. These three vectors are orthogonal to each other, where the surface normal is the vector perpendicular to the surface and pointing away from the object, and the two principal curvature vectors point in the directions of the greatest and least curvature of the surface.

We can use the surface normal (previously referred to as point-normal) and principal curvature to define the orientation of the sensor patch. The following video describes what those represent.
[Surface Normals and Principle Curvatures](https://res.cloudinary.com/dtnazefys/video/upload/v1731342526/point_normal.mp4)
