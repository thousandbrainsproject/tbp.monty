---
title: Sensor Module
---

# Sensor Modules
The **raw output from sensors in the environment is sent to the sensor module and turned into the [CMP-compliant format](./monty/messages-in-monty.md)**. The universal format that all sensor modules output is **features at pose** in 3D space. Each sensor connects to a sensor module which turns the raw sensory input into this format of features at locations. Each input therefore contains x, y, z coordinates of the feature location relative to the body and three orthonormal vectors indicating its rotation. In sensor modules these pose-defining vectors can, for example, be defined by the surface normal and principal curvature directions sensed at the center of the patch. In learning modules the pose vectors are defined by the detected object rotation. Additionally, the sensor module returns the sensed pose-independent features at this location (e.g. color, texture, curvature, ...). The sensed **features can be modality-specific** (e.g. color for vision or temperature for touch) while the **pose is modality agnostic**.

![Observation processing into the Cortical Messaging Protocol on the example of an RGBD sensor. The sensor patch comprises a small area of the object (blue square) and if the sensor is a camera it returns an RGBD image. We apply a transform (see section below for details) to this image which calculates the x, y, z locations relative to the agent’s body for each pixel using the depth values and the sensor location. From these points in space, the sensor module then calculates the surface normal and principal curvature directions at the center point of the patch (pose). Additionally, the sensor module can extract pose-independent features such as color and the magnitude of curvature. The pose (location + surface normal and curvature direction) and features make up the observation at time step t and are the output of the sensor module.](../figures/how-monty-works/observations_w_labels.png)

| List of all sensor module classes | Description                                                                                                                                                                                                                                                                                                                    |
| --------------------------------- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **SensorModule**                  | Abstract sensor module class.                                                                                                                                                                                                                                                                                                  |
| **CameraSM**                     | Sensor module connected to an RGBD camera. Extracts pose and features in CMP format from an RGBD patch. Keeps track of agent and sensor states. Also checks if observation is on object and should be sent to LM. Can be configured to add feature noise.                                                                      |
| **TwoDSensorModule**             | Sensor module connected to an RGBD camera that converts observations into a 2D surface frame. It reports 2D movement through `location` and `displacement`, and represents local 2D morphological edge direction in `pose_vectors`.                                                                                           |
| **Probe**                   | A probe that can be inserted into Monty in place of a sensor module. It will track raw observations for logging, and can be used by experiments for positioning procedures, visualization, etc. What distinguishes a probe from a sensor module is that it does not process observations and does not emit a Cortical Message. |
| **SalienceSM** | A wide field-of-view sensor module that proposes locations to attend to. Implements inhibition-of-return, and can be configured with different strategies for ranking target locations. |

## Features Extracted by the Sensor Module
An SM can extract arbitrary features from its raw sensory input. In this implementation, some features are extracted using all of the information in the sensor patch (e.g. locations of all points in the patch for surface normal and curvature calculation) but then refer to the center of the patch (e.g. only the curvature and surface normal of the center are returned). An important point is that non-morphological features should be rotation invariant (i.e. remain the same, no matter from what angle or orientation they are viewed).

At the moment all the feature extraction is predefined but in the future, one could also imagine some features being learned.

## Noise

Each sensor module accepts `noise_params`, which configure the `DefaultMessageNoise` that adds feature and location noise to the created Cortical Message before sending. Features and location noise can be configured individually.

## Feature Change Filtering
Each sensor module accepts `delta_thresholds`, which configure a `FeatureChangeFilter` that may set the `use_state` message attribute to False if sensed features did not change significantly between subsequent observations. Significance is defined by the `delta_thresholds` parameter for each feature.

# Transforms
Before sending information to the sensor module which extracts features and poses, we can apply transforms to the raw input. Possible transforms are listed in tables below.  **Transforms are applied to all sensors in an environment before sending observations to the SMs** and are specified in the environment interface arguments.

> [!NOTE] 
> There is [ongoing work](https://github.com/thousandbrainsproject/tbp.monty/pull/911) to integrate transforms into the sensor module class.


| List of all transform classes  | Description                                                                                                                                                                                       |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MissingToMaxDepth**          | Habitat depth sensors return 0 when no mesh is present at a location. Instead, return max_depth.                                                                                                  |
| **AddNoiseToRawDepthImage**    | Add gaussian noise to raw sensory input.                                                                                                                                                          |
| **GaussianBlurRGB**            | Apply Gaussian blur to the RGB channels of selected sensors while preserving the alpha channel.                                                                                                  |
| **DepthTo3DLocations**         | Transform semantic and depth observations from camera coordinates (2D) into agent (or world) coordinates (3D). Also estimates whether we are on the object or not if no semantic sensor is present. |

<br />

## Transforms vs. Sensor Modules
For an overview of **which type of data processing belongs where please refer to the following rules:**

- Any data transforms that apply to the entire dataset or to all SMs can be placed in the dataset.transform.

- Any data transforms that are specific to a sensory modality belong in SM. For example, an auditory SM could include a preprocessing step that computes a spectrogram, followed by a convolutional neural network that outputs feature maps.

- Transforming locations in space from coordinates relative to the sensor, to coordinates relative to the body, happens in the SM.

- The output of both SMs and LMs is a Message class instance containing information about pose relative to the body and detected features at that pose. This is the input that any LM expects.
