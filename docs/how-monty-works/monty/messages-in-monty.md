---
title: Messages in Monty
---
## CMP and the Message Class
The [Cortical Messaging Protocol](../../overview/architecture-overview/cortical-messaging-protocol.md) is defined in the `Message` class. The output of every SM and LM is an instance of the `Message` class which makes sure it contains all required information. The required information stored in a `Message` instance is:

- location (relative to the body)

- morphological features: pose_vectors (3x3 orthonormal), pose_fully_defined (bool), on_object (bool)

- non-morphological features: color, texture, curvature, ... (dict)

- confidence (in [0, 1])

- use state (bool)

- sender id (unique string identifying the sender)

- sender type (string in ["SM", "LM"])

The `Message` class is quite general and depending who outputs it, it can be interpreted in different ways. As output of the sensor module, it can be seen as a percept. When output by the learning module it can be interpreted as the hypothesized or most likely percept (representing object ID and pose). When it is the motor output of the LM it can be seen as a goal (for instance specifying the desired location and orientation of a sensor or object in the world). Lastly, when sent as lateral votes between LMs we send a list of message class instances which are interpreted as votes (where votes do not contain non-morphological, modality-specific features but only pose information associated with object IDs). 

The figure below shows which messages in Monty are CMP compliant.

![Information communicated along solid lines follows the CMP (i.e., contains features and pose). Dashed lines are the interface of the system with the world and subcortical compute units and do not need to follow the CMP. Blue lines indicate the main flow of information up the hierarchy. Purple lines show top-down connections, biasing the lower-level learning modules. Green lines show lateral voting connections. Pink lines show the communication of goals which eventually translate into motor commands in the motor system. Discontinuations in the diagram are marked with dots on line-ends.](../../figures/overview/overview_diagram.png)
