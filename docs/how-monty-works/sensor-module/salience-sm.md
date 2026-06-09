---
title: SalienceSM
---

# Salience Sensor Module
`SalienceSM` is a sensor module that operates on wide field-of-view imagery. Unlike `CameraSM`, it is not used to extract features and locations meant for a learning module. Rather, its purpose is to propose locations that the motor system should move to next. This enables a form of model-free movement guidance, in contrast to model-based policies located within Learning Modules.

`SalienceSM` relays its proposed locations and their degree of salience in the form of `Goal` objects. It is currently the only sensor module that produces goals which can be used by the motor system and its policies. Specifically, the `LookAtGoal` policy was designed to operate in conjunction with `SalienceSM`.

`SalienceSM` has two main components. First, it uses a `ReturnInhibitor` to implement [inhibition of return](https://en.wikipedia.org/wiki/Inhibition_of_return), a biologically-inspired mechanism that discourages returning to previously visited areas. Second, it uses a `SalienceStrategy`, which is used to rank candidate locations.

## ReturnInhibitor

Inhibition of return is a mechanism observed in attention and eye-movement systems. After attention has been drawn to a location, the nervous system becomes less likely to immediately return to that same location. This encourages exploration of new parts of the visual field and supports efficient scanning.

In Monty, `ReturnInhibitor` implements this idea by keeping a decaying memory of recently visited locations. Each visited location is represented by a `DecayKernel`, whose influence is strongest at the visited point and decreases with both distance and time. The `DecayField` stores the active kernels, removes them once they have decayed far enough, and computes an inhibition weight for each candidate goal location. `SalienceSM` then uses those weights to reduce the salience of locations near recent fixations.

## SalienceStrategy

Salience describes how strongly a location stands out as a candidate for attention or action. In biological vision, salience can be driven by bottom-up signals such as contrast, color, motion, orientation, or depth, as well as by task-dependent and top-down influences. The superior colliculus is one important structure involved in combining these signals into spatial maps that help guide orienting movements such as saccades.

In Monty, a `SalienceStrategy` computes a salience map from the current image observation. There are currently two strategies implemented.
 - `UniformSalienceStrategy`: Assigns equal salience to every pixel before on-object filtering and inhibition of return are applied. This means the current behavior is mostly shaped by return inhibition rather than visual distinctiveness. This is the strategy used by many of our distant agent benchmark experiments.
 - `Vocus2`: A biologically inspired salience detection system introduced by [Frintrop et al.](https://ieeexplore.ieee.org/document/7298603) and based on their open-source [C++ implementation](https://github.com/GeeeG/VOCUS2). Briefly, this salience detection system works by computing conspicuity maps for luminance, red-green opponency, blue-yellow opponency, orientation content, and depth which are then combined to form a final salience map.
