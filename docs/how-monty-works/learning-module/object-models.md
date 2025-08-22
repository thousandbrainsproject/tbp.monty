---
title: Object Models
---

# Learning Module Memory
First a few words on memory in learning modules in general. Each LM has two types of memory, a short-term memory ("buffer") which stores recent observations and a long-term memory which stored the structured object models. 

## The Buffer (STM)

Each learning module has a buffer class which could be compared to a short term memory (STM). The buffer only stores information from the current episode and is reset at the start of every new episode. Its content is used to update the graph memory at the end of an episode. (The current setup assumes that during an episode only one object is being explored, otherwise the buffer would have to be reset more often). The buffer is also used to retrieve the location observed at the previous step for calculating displacements. Additionally, the buffer stores some information for logging purposes.

## The Graph Memory (LTM)

Each learning module has one graph memory class which it uses as a long term memory (LTM) of previously acquired knowledge. In the graph learning modules, the memory stores explicit object models in the form of graphs (represented in the ObjectModel class). The graph memory is responsible for storing, updating, and retrieving models from memory.

![Graph memory classes and their relationships.](../../figures/how-monty-works/gm_classes.png)


# Object Models

Object models are stored in the graph memory and typically contain information about one object. The information they store is encoded in reference frames and contains poses relative to each other and features at those poses. There are currently two types of object models:

**Unconstrained graphs** are instances of the GraphObjectModel class. These encode an object as a graph with nodes and edges. Each node contains a pose and a list of features. Edges connect each node to its k nearest neighbors and contains a displacement and a list of features. More information on graphs will be provided in the next sections.

**Constrained graphs** are instances of the GridObjectModel class. These models are constrained by their size, resolution, and complexity. Unconstrained graphs can contain an unlimited amount of nodes and they can be arbitrarily close or far apart from each other. Constrained graphs make sure that the learned models have to be efficient by enforcing low resolution models for large objects such as a house and high resolution models for small objects such as a dice. This is more realistic and forces Monty to learn compositional objects which leads to more efficient representations of the environment, a higher representational capacity, faster learning, and better generalization to new objects composed of known parts. The three constraints on these models are applied to the raw observations from the buffer to generate a graph which can then be used for matching in the same way as the unconstrained graph. More information on constrained graphs will be provided in the following sections.

![Three constraints are enforced on constrained graph models. Max_size is defined in cm so if it is for example set to 10, it means the learned object models can cover an area of **10**cm<sup>3</sup> If max_voxels_per_dim would be set to 4 (as shown in this figure), then each voxel would be of size **2**cm<sup>3</sup> and any locations within that voxel cannot be distinguished. Max_nodes is the number of voxels that will be selected to become nodes in the graph that is eventually used for matching.](../../figures/how-monty-works/graph_constrains.png)


# Graph Building

A graph is constructed from a list of observations (poses, features). Each observation can become a node in the graph which in turn connects to its nearest neighbors in the graph (or by temporal sequence), indicated by the edges of the graph. **Each edge has a displacement associated with it** which is the action that is required to move from one node to the other. Edges can also have other information associated with them, for instance, rotation invariant point pair features (Drost et al., 2010). **Each node can have multiple features associated with it** or simply indicate that the object exists there (morphology). Each node must contain location and orientation information in a common reference frame (object centric with an arbitrary origin).

![A Graph of features (nodes), linked by displacements (edges). Each node represents a relative location and stores three pose vectors (for example the surface normal and the two principal curvature directions). Nodes can also have pose-independent features associated with them such as color and curvature. The graph stored in memory can then be used to recognize objects from actual feature-pose observations.](../../figures/how-monty-works/full_graph.png#width=300px)

## What are Surface Normals and Principal Curvatures?

See the video in this section for more details: [Surface Normals and Principal Curvatures](../observations-transforms-sensor-modules.md#surface-normals-and-principal-curvatures)


**Similar nodes in a graph** (no significant difference in pose or features to an existing node) **are removed** (see above figure) and nodes could have a variety of features attached to them. Removing similar points from a graph helps us to be more efficient when matching and avoids storing redundant information in memory. This way we store more points where features change quickly (like where the handle attaches to the mug) and fewer points where features are not changing as much (like on a flat surface).

# Building Constrained Graphs

When using constrained graphs, each learning module has three parameters that constrain the size (`max_size`), resolution (`num_voxels_per_dim`), and complexity (`max_nodes`) of models it can learn. These parameters do not influence what an LM sees, they only influence what it will store in memory. For example if an LM with a small maximum model size is moving over a large object, it will perceive every observation on the object and try to recognize it. However, it will not be able to store a complete model of the object. It might know about subcomponents of the object if it has seen them in isolation before and send those to other LMs that can model the entire large object (usually at a lower resolution). Once we let go of the assumption that each episode only contains one object, we also do not need to see the subcomponents in isolation anymore to learn them. We would simply need to recognize them as separate objects (for example because they move independently) and they would be learned as separate models.

![The LMs that get the same sensory input but have different model parameters. The first one can learn large, low resolution models and could learn a complete model of the cup. The second one can learn small, high resolution models and would not be able to learn a model of this cup. Both receive the same sensations (grey dots) as they get input from the same patch of the retina in this example.](../../figures/how-monty-works/grid_with_same_size.png)


To generate a constrained graph, the observations that should be added to memory are first sorted into a 3D grid. The first observed location will be put into the center voxel of the grid and all following locations will be sorted relative to this one. The size of each voxel is determined by the maximum size of the grid (in cm) and the number of voxels per dimension. If more than 10% of locations fall outside of the grid, the object can not be added to memory.

After all observations are assigned to a voxel in the grid, we retrieve three types of information for each voxel:

- The **number of observations** in the voxel.

- The **average location** of all observations in the voxel.

- The **average features** (including pose vectors) of all observations in the voxel.

Finally, we select the _k_ voxels with the highest observation count, where _k_ is the maximum number of nodes allowed in the graph. We then create a graph from these voxels by turning each of the _k_ voxels into a node in the graph and assigning the corresponding average location and features to it.

![Generating constrained graphs by first sorting observations into a grid, constrained by max_size and num_voxels_per_dim. Then observations in each voxel are summarized to get a count and average locations and features. The k (max_nodes) voxels with the highest count are used to build a graph.](../../figures/how-monty-works/grid_types_blue_mug.png)


When updating an existing constrained object model, the new observations are added to the existing summary statistics. Then the new k-winner voxels are picked to construct a new graph.

The three grids used to represent the summary statistics (middle in figure above) are represented as sparse matrices to limit their memory footprint.

# Using Graphs for Prediction and Querying Them

We can use graphs in memory to predict if there will be a feature sensed at the next location and what the next sensed feature will be, given an action/displacement (**forward model**). This prediction error can then be used for graph matching to update the possible matches and poses.

A graph can also be queried to provide an action that leads from the current feature to a desired feature (**inverse model**). This can be used for a goal-conditioned action policy and more directed exploration. To do this we need to have a hypothesis of the object pose.
