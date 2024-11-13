---
title: Model
---
**The Monty class contains everything an agent needs to model and interact with the environment.** It contains (1) sensorimotor learning systems (also called [learning modules](how-learning-modules-work.md)), (2) communication between sensorimotor modules (see [here](../overview/architecture-overview/cortical-messaging-protocol.md)) , and (3) motor system for taking action (see [here](policy.md)).

A Monty instance can be arbitrarily customized as long as it implements a handful of types of abstract methods listed below.

It contains:

- **Step and Voting methods**, which define the modeling and action logic, and communications logic respectively

- Methods for **saving, loading, and logging**

- Methods called at **event markers** like pre-episode and post-episode

Below are the arguments associated with the Monty classes.

- a list of **SensorModule instances**, each of which is responsible for processing raw sensory input and transforming it into a canonical format that any LearningModule can operate on.

- a list of **LearningModule instances**, each of which is responsible for building models of objects given outputs from a sensor module

- a dictionary **mapping sensors to an agent** (`sm_to_agent_dict`)

- a matrix describing the **coupling from SensorModule to LearningModule** (`sm_to_lm_matrix`)

- a matrix describing the **coupling between LearningModules** used for voting (`lm_to_lm_vote_matrix`)

- a **motor system** responsible for moving the agent

- the **Monty class** used (`monty_class`) and its **arguments** (`monty_args`)

Using the above arguments we can specify the structure underlying our modeling system. For instance, if we have five sensors in the environment we would specify five sensor modules, each corresponding to one sensor (often we use an additional sensor module connected to a view finder sensor which does not connect to a learning module). Each sensor module would be connected to one learning module and the connection between the learning modules is specified in the lm_to_lm_matrix. The modeling in this Monty instance could then look as shown in the figure below.

![Example Monty class with five sensors, sensor modules, and learning modules as used in the FiveLMMontyConfig. Each sensor patch perceives a small part of the environment and sends it to its connected sensor module. The SM extracts features from the patch and a pose (location and rotation relative to the body). This is sent to the LM which models the input and outputs another feature (most likely object ID) and its pose (most likely rotation and location of the object). LMs have lateral connections between each other (dotted lines) to communicate possible poses and narrow down their hypotheses faster.](../figures/how-monty-works/five_lm_monty.png)


| List of all Monty classes         | Description                                                                                                                                                                       |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Monty**                         | Abstract Monty class from which all others inherit. Defines step method framework and other communication elements like voting.                                                   |
| **MontyBase**                     | Implements all the basic functionalities for stepping LMs and SMs and routing information between Monty components.                                                               |
| **MontyForGraphMatching**         | Implements custom step and vote functions adapted for graph-based learning modules. Also adds custom terminal conditions for object recognition and determining possible matches. |
| **MontyForEvidenceGraphMatching** | Customizes previous class with a voting function designed for the evidence-based LM. Also customizes motor suggestions to use evidence-based models.                              |