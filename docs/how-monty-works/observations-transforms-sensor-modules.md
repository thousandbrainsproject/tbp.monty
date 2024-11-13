---
title: Observations, Transforms & Sensor Modules
---
Before sending information to the sensor module which extracts features and poses we can apply transforms to the raw input. Possible transforms are listed in tables below.  **Transforms are applied to all sensors in an environment before sending observations to the SMs** and are specified in the data set arguments.

The transformed, **raw input is then sent to the sensor module and turned into the CMP-compliant format**. The universal format that all sensor modules output is **features at a pose** in 3D space. Each sensor connects to a sensor module which turns the raw sensory input into this format of features at locations. Each input therefore contains x, y, z coordinates of the feature location relative to the body and three orthonormal vectors indicating its rotation. In sensor modules these pose-defining vectors are defined by the point normal and principal curvature directions sensed at the center of the patch. In learning modules the pose vectors are defined by the detected object rotation. Additionally, the sensor module returns the sensed pose-independent features at this location (e.g. color, texture, curvature, ...). The sensed **features can be modality-specific** (e.g. color for vision or temperature for touch) while the **pose is modality agnostic**.

The Cortical Messaging Protocol is defined in the State class. The output of every SM and LM is an instance of the State class which makes sure it contains all required information. The required information stored in a State instance is:

- location (relative to the body)

- morphological features: pose_vectors (3x3 orthonormal), pose_fully_defined (bool), on_object (bool)

- non-morphological features: color, texture, curvature, ... (dict)

- confidence (in [0, 1])

- use_state (bool)

- sender_id (unique string identifying the sender)

- sender_type (string in ["SM", "LM"])

The State class is quite general and depending who outputs it, it can be interpreted in different ways. As output of the sensor module, it can be seen as the observed state. When output by the learning module it can be interpreted as the hypothesized or most likely state. When it is the motor output of the LM it can be seen as a goal state (for instance specifying the desired location and orientation of a sensor or object in the world). Lastly, when sent as lateral votes between LMs we send a list of state class instances which can be interpreted as all possible states (where states do not contain non-morphological, modality-specific features but only pose information associated with object IDs).

![Observation processing into Cortical Messaging Protocol. The sensor patch comprises a small area of the object (yellow square) and if the sensor is a camera it returns an RGBD image. We apply a transform to this image which calculates the x, y, z locations relative to the agent’s body for each pixel using the depth values and the sensor location. From these points in space, the sensor module then calculates the point normal and principal curvature directions at the center point of the patch (pose). Additionally, the sensor module can extract pose-independent features such as color and the magnitude of curvature. The pose (location + point normal and curvature direction) and features make up the observation at time step t and are the output of the sensor module.](../figures/how-monty-works/observations_w_labels.png)

## Point Normals and Principle Curvatures
[Point Normals and Principle Curvatures](https://res.cloudinary.com/dtnazefys/video/upload/v1731342526/point_normal.mp4)

In this implementation, some features are extracted using all of the information in the sensor patch (e.g. locations of all points in the patch for point normal and curvature calculation) but then refer to the center of the patch (e.g. only the curvature and point normal of the center are returned). At the moment all the feature extraction is predefined but in the future, one could also imagine some features being learned.

For an overview of **which type of data processing belongs where please refer to the following rules:**

- Any data transforms that apply to the entire dataset or to all SMs can be placed in the dataset.transform.

- Any data transforms that are specific to a sensory modality belong in SM. For example, an auditory SM could include a preprocessing step that computes a spectrogram, followed by a convolutional neural network that outputs feature maps.

- Transforming locations in space from coordinates relative to the sensor, to coordinates relative to the body, happens in the SM.

- The output of both SMs and LMs is a State class instance containing information about pose relative to the body and detected features at that pose. This is the input that any LM expects.

| List of all transform classes  | Description                                                                                                                                                                                       |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MissingToMaxDepth**          | Habitat depth sensors return 0 when no mesh is present at a location. Instead, return max_depth.                                                                                                  |
| **AddNoiseToRawDepthImage**    | Add gaussian noise to raw sensory input.                                                                                                                                                          |
| **DepthTo3DLocations**         | Transform semantic and depth observations from camera coordinate (2D) into agent (or world) coordinate (3D). Also estimates whether we are on the object or not if no semantic sensor is present. |

<br />

| List of all sensor module classes          | Description                                                                                                                                                                                                              |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **SensorModule**                           | Abstract sensor module class.                                                                                                                                                                                            |
| **SensorModuleBase**                       | Base class for all sensor modules. Just returns its unprocessed input.                                                                                                                                                   |
| **DetailedLoggingSM**                      | Extracts pose and features in CMP format from an RGBD patch. Also has the option to save the raw observations for logging.                                                                                               |
| **HabitatDistantPatchSM**                  | Custom version of previous SM for a habitat camera patch. Keeps track of agent and sensor states. Also checks if observation is on object and should be sent to LM.                                                      |
| **HabitatSurfacePatchSM**                  | Similar to previous but also sends off object observations to LM since this is needed for the compensation movements.                                                                                                    |
| **FeatureChangeSM**                        | Version of HabitatDistantPatchSM that only sends an observation to the LM if the sensed features changed significantly. How large this change should be is specified in the delta_thresholds parameter for each feature. |
| **NoiseMixin**                             | Option to add gaussian noise to processed sensor module output before sending it to the LM. Amount of noise can be specified in noise_params for features and locations individually.                                    |