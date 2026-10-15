---
title: SalienceSM
---

# Salience Sensor Module
Salience describes how strongly a location stands out as a candidate for attention or action. In biological vision, salience can be driven by bottom-up signals such as contrast, color, motion, orientation, or depth, as well as by task-dependent and top-down influences. The [superior colliculus](https://en.wikipedia.org/wiki/Superior_colliculus) is one important structure involved in generating and combining these signals into spatial maps, selecting targets, and issuing motor commands. In Monty, we use salience to guide exploration toward the most interesting and informative features in an environment. See [this video](https://www.youtube.com/watch?v=AE-ZZguDYY8) for more about how we use salience and the salience strategies we've implemented so far.

`SalienceSM` is a sensor module that operates on wide field-of-view imagery. Unlike `CameraSM`, it is not used to extract features and locations meant for a learning module. Rather, its purpose is to propose locations that the motor system should move to next. This enables a form of model-free movement guidance, in contrast to model-based policies located within Learning Modules.

`SalienceSM` relays its proposed locations and their degree of salience in the form of `Goal` objects. It is currently the only sensor module that produces goals which can be used by the motor system and its policies. Specifically, the `LookAtGoal` policy was designed to operate in conjunction with `SalienceSM`. It can also propose an `AttentionRegion`, a set of locations the [attention system](../attention-system.md) should favor when deciding which goals reach the motor system.

`SalienceSM` has three main components. First, it uses a `ReturnInhibitor` to implement [inhibition of return](https://en.wikipedia.org/wiki/Inhibition_of_return), a biologically-inspired mechanism that discourages returning to previously visited areas. Second, it uses a `SalienceStrategy`, which is used to rank candidate locations. Third, it can optionally use a `SegmentationStrategy` to propose an object-like region currently under fixation and propose it as an attention region.

## ReturnInhibitor

Inhibition of return is a mechanism observed in attention and eye-movement systems. After attention has been drawn to a location, the nervous system becomes less likely to immediately return to that same location. This encourages exploration of new parts of the visual field and supports efficient scanning.

In Monty, `ReturnInhibitor` implements this idea by keeping a decaying memory of recently visited locations. Each visited location is represented by a `DecayKernel`, whose influence is strongest at the visited point and decreases with both distance and time. The `DecayField` stores the active kernels, removes them once they have decayed far enough, and computes an inhibition weight for each candidate goal location. `SalienceSM` then uses those weights to reduce the salience of locations near recent fixations.

## SalienceStrategy
`SalienceSM` uses a `SalienceStrategy` to compute salience maps. The `SalienceStrategy` is a configurable component of `SalienceSM`. There are currently two `SalienceStrategy` implementations in Monty:
 - `UniformSalienceStrategy`: Assigns equal salience to every pixel before on-object filtering and inhibition of return are applied. This means the current behavior is mostly shaped by return inhibition rather than visual distinctiveness. This is the strategy used by many of our distant agent benchmark experiments.
 - `Vocus2`: A biologically inspired salience detection system introduced by [Frintrop et al.](https://ieeexplore.ieee.org/document/7298603) and based on their open-source [C++ implementation](https://github.com/GeeeG/VOCUS2). Briefly, this salience detection system works by computing conspicuity maps for luminance, red-green opponency, blue-yellow opponency, orientation content, and depth which are then combined to form a final salience map.

## SegmentationStrategy
When an object is in view, it is often useful to keep attending to that object rather than letting a salient feature elsewhere in the field of view pull the sensor away. `SalienceSM` supports this through an optional `SegmentationStrategy`, which takes the RGB image and returns a mask marking pixels that surround the central pixel (i.e., at the point of fixation), and which appear to form a coherent region. `SalienceSM` intersects this mask with the pixels that are on the object, looks up their 3D locations, and packages them as an `AttentionRegion` with every location at a weight of 1.0. The region is made available through `propose_region`, and the Monty class forwards it to the [attention system](../attention-system.md) on every step.

There is currently one `SegmentationStrategy` implementation in Monty:
 - `SlicMerge`: Segments the image into superpixels using [SLIC](https://scikit-image.org/docs/0.25.x/api/skimage.segmentation.html#skimage.segmentation.slic) (simple linear iterative clustering), computes the mean color of each superpixel in the [LAB](https://en.wikipedia.org/wiki/CIELAB_color_space) color space, and then grows a region outward from the superpixel under the fixation point. Starting from that superpixel, it repeatedly accepts neighboring superpixels whose mean color is within a configurable threshold of the superpixel they are reached from, and stops when no more neighbors qualify. The accepted superpixels form the mask. The number of superpixels, their compactness, and the color threshold for merging are all configurable.
