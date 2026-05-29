---
title: SalienceSM
---

# Salience Sensor Module
`SalienceSM` is a sensor module that operates on wide field-of-view imagery. Unlike `CameraSM`, it is not used to extract features and locations meant for a learning module. Rather, its purpose is to propose locations that the motor system should move to next. This is currently the only sensor module that produces `Goal` objects which can be used by the motor system and its policies. Specifically, the `LookAtGoal` policy was designed to operate in conjunction with `SalienceSM`.


`SalienceSM` has two main components. First, it uses a `ReturnInhibitor` to implement [inhibition of return](https://en.wikipedia.org/wiki/Inhibition_of_return), a biologically-inspired mechanism that discourages returning to previously visited areas. Second, it uses a `SalienceStrategy`, which is used to rank candidate locations.

## ReturnInhibitor

Inhibition of return is a mechanism observed in attention and eye-movement systems. After attention has been drawn to a location, the nervous system becomes less likely to immediately return to that same location. This encourages exploration of new parts of the visual field and supports efficient scanning, because recently inspected locations are temporarily suppressed in favor of novel ones.

In Monty, `ReturnInhibitor` implements this idea by keeping a decaying memory of recently visited locations. Each visited location is represented by a `DecayKernel`, whose influence is strongest at the visited point and decreases with both distance and time. The `DecayField` stores the active kernels, removes them once they have decayed far enough, and computes an inhibition weight for each candidate goal location. `SalienceSM` then uses those weights to reduce the salience of locations near recent fixations.

## SalienceStrategy

Salience describes how strongly a location stands out as a candidate for attention or action. In biological vision, salience can be driven by bottom-up signals such as contrast, color, motion, orientation, or depth, as well as by task-dependent and top-down influences. The superior colliculus is one important structure involved in combining these signals into spatial maps that help guide orienting movements such as saccades.

In Monty, a `SalienceStrategy` computes a salience map from the current image observation. The only strategy currently implemented is `UniformSalienceStrategy`, which assigns equal salience to every depth pixel before on-object filtering and inhibition of return are applied. This means the current behavior is mostly shaped by coverage and return inhibition rather than visual distinctiveness. We expect to add a `Vocus2` strategy soon, which will provide a more feature-driven salience map.
