---
title: Glossary
description: This section aims to provide concise definitions of terms commonly used at the Thousand Brains Project and in Monty.
---
# Dendrites

Implement pattern recognizers to identify patterns such as a specific SDR. One neuron is typically associated with multiple dendrites such that it can identify multiple patterns. In biology, dendrites of a postsynaptic cell receive information from the axons of other presynaptic cells. The axons of these presynaptic cells connect to the dendrites of postsynaptic cells at a junction called a "synapse". An SDR can be thought of as a pattern which is represented by a set of synapses that are collocated on a single dendritic segment.

# Efference Copy

A copy of the motor command that was output by the policy and sent to the actuators. This copy can be used by learning modules to update their states or make predictions.

# Environment

Depending on the environments' state and agents' actions and sensors, the environment returns an observation for each sensor.

# Features

Characteristics that can be sensed at a specific location. Features may vary depending on the sensory modality (for example, color in vision but not in touch).

# Graph

A set of nodes that are connected to each other with edges. Both nodes and edges can have features associated with them. For instance all graphs used in the Monty project have a location associated with each node and a variable list of features. An edge can, for example, have a displacement associated with it.

# Inductive bias

An assumption that is built into an algorithm/model. If the assumption holds, this can make the model a lot more efficient than without the inductive bias. However, it will cause problems when the assumption does not hold.

# Learning Module

A computational unit that takes features at poses as input and uses this information to learn models of the world. It is also able to recognize objects and their poses from the input if an object has been learned already.

# Path Integration

Updating an agent's location by using its own movement and features in the environment.

# Policy

Defines the function used to select actions. Selected actions can be dependent on a model's internal state and on external inputs.

# Pose

An object's location and orientation (in a given reference frame). The location can for example be x, y, z coordinates and the orientation can be represented as a quaternion, Euler angles or rotation matrix.

**displacement:** The spatial difference between two locations. In 3D space, this would be a 3D vector.

# Reference Frame

A specific coordinate system within which locations and rotations can be represented. For instance, a location may be represented relative to the body (body/ego-centric reference frame) or relative to some point in the world (world/allo-centric reference frame) or relative to an object's center (object-centric reference frame).

# Rigid Body Transformation

Applies a displacement/translation and a rotation to a set of points. Every point is transformed in the same way such that the overall shape stays the same (i.e. the relative distance between points is fixed).

# Sensor Module

A computational unit that turns raw sensory input into the cortical messaging protocol. The structure of the output of a sensor module is independent of the sensory modality and represents a list of features at a pose.

# Sensorimotor/Embodied

Learning or inference through interaction with an environment using a closed loop between action and perception. This means, observations depend on actions and in turn the choice of these actions depend on the observations.

# Sparse Distributed Representation (SDR)

A binary vector with significantly more 0 bits than 1 bits. Significant overlap between the bit assignments in different SDRs captures similarity in representational space (e.g. similar features).

# Transformation

Applies a displacement/translation and a rotation to a point.

# Voting

Multiple computational units share information about their current state with each other. This can for instance be their current estimate of an object's ID or pose. This information is then used to update each unit's internal state until all units reach a consensus.