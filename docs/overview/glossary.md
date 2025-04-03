---
title: Glossary
description: This section aims to provide concise definitions of terms commonly used at the Thousand Brains Project and in Monty.
---
[Axon]:
  https://en.wikipedia.org/wiki/Axon
[Bit array]:
  https://en.wikipedia.org/wiki/Bit_array
[Coordinate system]:
  https://en.wikipedia.org/wiki/Coordinate_system
[Cosine similarity]:
  https://en.wikipedia.org/wiki/Cosine_similarity
[Dendrite]:
  https://en.wikipedia.org/wiki/Dendrite
[Displacement (geometry)]:
  https://en.wikipedia.org/wiki/Displacement_(geometry)
[Efference copy]:
  https://en.wikipedia.org/wiki/Efference_copy
[Feature (machine learning)]:
  https://en.wikipedia.org/wiki/Feature_(machine_learning)
[Inductive bias]:
  https://en.wikipedia.org/wiki/Inductive_bias
[Neuron]:
  https://en.wikipedia.org/wiki/Neuron
[Path integration]:
  https://en.wikipedia.org/wiki/Path_integration
[Pattern recognition]:
  https://en.wikipedia.org/wiki/Pattern_recognition
[Frame of reference]:
  https://en.wikipedia.org/wiki/Frame_of_reference
[Synapse]:
  https://en.wikipedia.org/wiki/Synapse
[Wikipedia]:
  https://en.wikipedia.org

[SDR]:
  sparse-distributed-representation-sdr

**Usage Notes:**
Most of the "See Also" links go to [Wikipedia] entries. These may not be exact matches for TBP's usage, but they provide useful context.

# Dendrites

Dendrites implement pattern recognizers, identifying patterns such as a specific [SDR]. One neuron is typically associated with multiple dendrites such that it can identify multiple patterns. In biology, dendrites of a postsynaptic cell receive information from the axons of other presynaptic cells. The axons of these presynaptic cells connect to the dendrites of postsynaptic cells at a junction called a "synapse". An SDR can be thought of as a pattern which is represented by a set of synapses that are collocated on a single dendritic segment.

**See Also:** [Axon], [Dendrite], [Neuron], [Pattern recognition], [Synapse]

# Displacement

The displacement is defined as the spatial difference between two locations. In 3D space, this would be a 3D vector.

**See Also:** [Displacement (geometry)]

# Efference Copy

An efference copy duplicates a motor command that was output by the policy and sent to the actuators. This copy can be used by learning modules to update their states or make predictions.

**See Also:** [Efference copy]

# Environment

Monty's environment is defined as the set of entities with which it can interact (e.g., sense, manipulate). Depending on the environments' state and agents' actions and sensors, the environment returns an observation for each sensor.

# Features

In Monty, a [feature] is defined as a characteristic that can be sensed at a specific location. Features may vary depending on the sensory modality (for example, color in vision but not in touch).

**See Also:** [Feature (machine learning)]

# Graph

A set of nodes that are connected to each other with edges. Both nodes and edges can have features associated with them. For instance, all graphs used in the Monty project have a location associated with each node and a variable list of features. An edge can, for example, have a [displacement](#displacement) associated with it.

# Inductive Bias

Inductive bias is an assumption that is built into an algorithm/model. If the assumption holds, this can make the model a lot more efficient than without the inductive bias. However, it will cause problems when the assumption does not hold.

**See Also:** [Inductive bias]

# Learning Module

A computational unit that takes features at [poses](pose) as input and uses this information to learn models of the world. It is also able to recognize objects and their poses from the input if an object has been learned already.

# Model

In Monty, a model (sometimes referred to as [Object Model](../how-monty-works/how-learning-modules-work.md#object-models)), is a representation of an object stored entirely within the boundaries of a learning module. The notion of a model in Monty differs from the concept of a deep learning neural network model in several ways:

- A single learning module **stores multiple object models** in memory, simultaneously.
- The Monty system may have **multiple models of the same object** if there are multiple learning modules - this is a desired behavior.
- Learning modules **update models independently** of each other.
- Models are structured using [reference frames](#reference-frame), not just a bag of features.
- Models represent **complete objects**, not just parts of objects. These objects can still become subcomponents of compositional objects but are also objects themselves (like the light bulb in a lamp).

A useful analogy is to think of **Monty models** as **CAD representations** of objects that exist within the confines of a learning module.

**See Also:** [Do Cortical Columns in the Brain Really Model Whole Objects Like a Coffee Mug in V1?](../how-monty-works/faq-monty.md#do-cortical-columns-in-the-brain-really-model-whole-objects-like-a-coffee-mug-in-v1)

# Path Integration

Path integration is defined as updating an agent's location by using its own movement and [features](#feature) in the [environment](environment).

**See Also:** [Path integration]

# Policy

Defines the function used to select actions. Selected actions can be dependent on a model's internal state and on external inputs.

# Pose

An object's location and orientation (in a given [reference frame](reference_frame)). The location can for example be x, y, z coordinates and the orientation can be represented as a quaternion, Euler angles, or a rotation matrix.

# Reference Frame

A reference frame is a specific [coordinate system] within which locations and rotations can be represented. For instance, a location may be represented relative to the body (body/ego-centric reference frame) or relative to some point in the world (world/allo-centric reference frame) or relative to an object's center (object-centric reference frame).

**See Also:** [Frame of reference]

# Rigid Body Transformation

Applies a displacement/translation and a rotation to a set of points. Every point is transformed in the same way such that the overall shape stays the same (i.e., the relative distance between points is fixed).

# Sensor Module

A computational unit that turns raw sensory input into the cortical messaging protocol. The structure of the output of a sensor module is independent of the sensory modality and represents a list of features at a pose.

# Sensorimotor/Embodied

Learning or inference through interaction with an environment using a closed loop between action and perception. This means, observations depend on actions and in turn the choice of these actions depend on the observations.

# Sparse Distributed Representation (SDR)

In Monty, an SDR is a binary vector (i.e., [bit array]) with significantly more 0 bits than 1 bits. Significant overlap between the bit assignments in different SDRs captures [cosine similarity] in representational space (e.g., similar [features](#feature)).

# Transformation

Applies a displacement/translation and a rotation to a point.

# Voting

Multiple computational units share information about their current state with each other. This can for instance be their current estimate of an object's ID or [pose](#pose). This information is then used to update each unit's internal state until all units reach a consensus.
