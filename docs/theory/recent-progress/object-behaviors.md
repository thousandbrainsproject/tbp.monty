---
title: Object Behaviors
description: Our theory on how object behaviors are represented, learned, inferred, and utilized.
---
# A Method for Learning the Changing Structure of the World (Object Behaviors)

# The Problem

The world is both static and dynamic. Some features of the world have a fixed arrangement relative to each other and don’t change over time. For example, the arrangement of edges and surfaces of a coffee cup or the arrangement of rooms in a house do not change from moment to moment. Sometimes, however, the arrangement of features in the world changes over time and in predictable ways. For example, a stapler does not have a fixed morphology; it can open and close and emit staples, a traffic light can change color, a human can move its limbs to run, and a lever may rotate, causing a door to open. We refer to these changes as *behaviors*. The problem we want to solve is **how an intelligent system can learn and represent the behaviors of objects**. A second and related problem is **how an intelligent system recognizes behaviors,** both on the object that they were learned on and on novel object-behavior combinations. These are essential requirements for intelligent systems interacting with the world and solving problems. 

# The Solution

The Thousand Brains Theory (TBT) currently explains how an intelligent system learns the static structure of the world. In this document we describe how to extend the TBT to include behaviors. In the TBT, static objects are learned as a set of features at poses (location and orientation) in a reference frame. There is an assumption that the features are not changing or moving, therefore, the existing theory and implementation works well for representing static structure.

The invention summarized here is that, to represent behaviors, we add a second reference frame. This second reference frame is used to represent behaviors. Static features are stored at locations in the first reference frame, and changing features are stored at locations in the second reference frame. Static structures are inferred in the first reference frame, and dynamic behaviors are inferred in the second reference frame. Because the two modeling systems are collocated (the same movement information is used to move through the two reference frames) but independent, behaviors learned by observing one object can be recognized on new objects. This method is very general and applies to every type of behavior we observe in the world.  

![Behavior models (purple) use an analogous mechanism to morphology models (green). The main difference is that they store changes instead of static features and have a temporal dimension.](../../figures/theory/behavior_models.png#width=600px)

## Implementation \- Illustrated on the Example of the Neocortex

There are multiple ways this solution could be implemented. This figure shows how it might be implemented n the brain. We believe that each neocortical column contains the two types of reference frames, one for static objects and one for behaviors. In the brain, they might be located in L6a and L5b, respectively. Both reference frames are updated together using the same movement input, transformed into the object’s and behavior’s reference frame separately.  
Unlike models of static structure, models of dynamic structure consist of **sequences** of behavioral elements over time, and the timing between elements is often important. In cortical columns, we propose that cells in L4 represent features on static objects and cells in L3 represent changes in dynamic behaviors. Unlike L4 cells, L3 cells have apical dendrites that extend to L1. We propose that the apical dendrites in L1 learn the timing of events via a projection from matrix cells in the thalamus. The brain represents one implementation of this idea, but other implementations could be used (see [Implementation \- Illustrated on the Example of Monty](#implementation---illustrated-on-the-example-of-monty) for another example below).  
   
As we have previously demonstrated [2, 7, 10], cortical columns can vote to quickly reach a consensus of the static object being observed. The same mechanism applies to behaviors. Multiple columns can vote to quickly infer behaviors.  
 

### Details

The figure shows how we believe this system is implemented in neocortical columns. Here we describe the theory in neuroscience terms, but it can be implemented in other ways. For example, some of the details shown in the figure are implemented differently in learning modules in Monty (see [Implementation \- Illustrated on the Example of Monty](#implementation---illustrated-on-the-example-of-monty)). Although the theory maps well onto the anatomy and physiology of cortical layers as shown in the figure, the solution presented here does not depend on this mapping being correct.  
   
There are two modeling systems within each column. One comprises a reference frame in L6 associatively connected with static features in L4. This is the system we have previously described [2-8] and implemented [9]. We refer to this as a morphology or object model (green arrows). The second modeling system comprises another reference frame, possibly in L5b, associatively connected to changing features in L3. We refer to this as a behavior model (purple arrows). At any point in time, the location in the two reference frames is updated using the same physical movement. The two reference frames anchor independently. This is essential in that it allows the system to learn a behavior while observing one object, but then infer or apply the behavior to a different object.  
   
Analogous to object models, behavior models can be recognized in any location, orientation, and scale by transforming the physical movement vector into the behavior's reference frame. This allows for recognizing a behavior at different locations on an object in varying orientations and scales, and therefore represents a flexible way to apply and recognize behaviors in novel situations. Notably, the behavior can be recognized independently of the object on which it was learned.   
   
In many ways, the two modeling systems are similar. The biggest difference is that one stores features@locations and the other stores changes@locations. Another difference is that the behavioral model has a temporal dimension. Behaviors are high-order sequences, and the time between elements of the sequence is often important. Previous work at Numenta showed how any layer of neurons can learn high-order sequences [1]. We assume this is occurring in L3. We also proposed a mechanism for learning the time between sequence elements. Matrix cells in the thalamus represent a kind of countdown clock. Matrix cells project to L1 where they form synapses on the apical dendrites of L3 cells, allowing the L3 behavioral model to encode the timing of behavioral elements. The static model in L4 has no need for time, thus most cells in L4 lack apical dendrites.  
 

### Further Details

Our current best guess is that minicolumns in L4 represent the orientation of a feature relative to the current object. In a primary sensory region, the feature may be as simple as an edge and the minicolumns represent edges at different orientations. The set of active cells in the set of L4 minicolumns is unique to a particular location on a particular object. (This has been observed, for example, see border ownership cells.)  
   
Our current best guess is that L3 minicolumns represent the same orientations as L4, but the cells in L3 only become active when the edge is moving, which is the dominant response property of L3 cells. We would also expect L3 cells to become active when an edge appears or disappears, not just moving. The sharing of minicolumn attributes across L4 and L3 is not essential, but it appears to be the case and simplifies learning.

 Any particular object may exhibit multiple independent behaviors. For example, the top of a stapler can be raised or lowered and, independently, the stapler deflection plate can be rotated. A coffee maker may have a power switch, a lid to add water, and a grinds basket that swings out. Each of these parts exhibit their own behaviors.

Similar to compositional object models, where an object recognized in a lower region can become a feature on an object in a higher region, behaviors can become features on higher-level objects. This way, a behavior can be associated with different locations on an object. This association also encodes the behavior's orientation and scale relative to the object.  
   
The above examples illustrate that the two modeling systems, morphology/static and behavior/dynamic are similar and share many mechanisms and attributes. This commonality makes it easier to understand and implement the two systems. 

## Implementation \- Illustrated on the Example of Monty

We have implemented a system for learning and recognizing static object models in an open-source project called Monty [9]. The static object models are represented as features (vectors) at locations and orientations in Euclidean space. We then use sensed features and movements (displacements) to recognize an object in any location and orientation in the world. This is done by iteratively updating hypotheses about sensed objects and their poses. The pose hypothesis is used to rotate the sensed displacement into the model’s reference frame. The features stored in the model at the location in the object’s reference frame are then compared to the sensed features. For a detailed explanation of the algorithm, see our publications [7, 8] and documentation [10].  
For behavior models, we propose using the same mechanism. The main difference is that the behavior model only stores points in the behavior’s reference frame if a change has been detected. This could be a local movement (e.g. a moving edge) or a feature change (e.g. color changing). It therefore encodes changes at locations. We use the same mechanism of iteratively testing hypotheses when inferring a behavior. Analogously to the object recognition mechanism, we apply a pose hypothesis to the sensed sensor displacement to transform it into the behavior model’s reference frame. We can then compare the sensed change to the stored change at that location.  
For a concrete implementation in tbp.monty, one would need to add a capability to sensor modules to detect changes. Those could, for example, be local optic flow (indicating movement of the object) or features appearing or disappearing. These changes would be communicated to learning modules as part of the CMP messages (instances of the `State` class), separated from static features. The learning module would then look at the incoming CMP message and, based on whether a static feature or a changing feature was detected, look at the object model or the behavior model, respectively. The static mechanisms are what we have implemented to date. For the behavior model, we would use the same mechanisms for learning and inference, only that they are applied to the sensed changing features.  
Additionally, behavior models have a temporal dimension. This could be implemented as multiple graphs or sets of points in the same reference frame that are traversed as time passes. In code, there are many possible ways to achieve this. The important thing is that the timing in the temporal sequence can condition the changes to expect at a location.  
Recognizing a behavior model in practice will likely depend more strongly on voting between multiple learning modules (at least for more complex behaviors where one sensor patch may not be able to observe a lot of the behavior fast enough on its own). The voting process for behavior models would work analogously to the voting process already implemented for object models. It would additionally require a shared temporal signal (the same as used internally in an LM) to keep the learning modules in sync.

# Videos of Meetings Discussing the Invention

For in-depth descriptions of the invention presented here, see the series of meeting recordings in which we conceived of the idea, formalized the general mechanism, and discussed its implementation in the brain and in Monty.  
You can find the whole Playlist here:    
[https://www.youtube.com/playlist?list=PLXpTU6oIscrn\_v8pVxwJKnfKPpKSMEUvU](https://www.youtube.com/playlist?list=PLXpTU6oIscrn_v8pVxwJKnfKPpKSMEUvU)   
Over the next weeks, we will add more videos to this playlist as we continue to explore the remaining open questions. For now, you can find the following videos:

* **Brainstorming on Modeling Object Behaviors and Representations in Minicolumns** [https://youtu.be/TzP53N2LsRs](https://youtu.be/TzP53N2LsRs) \- The first meeting after we had our breakthrough idea outline in this document (using the same mechanism for behavior models as for object models but storing changes instead of features).  
* **Review of the Cortical Circuit, Object Behavior** [https://youtu.be/Dt4hT4FxQls](https://youtu.be/Dt4hT4FxQls) \- A long follow-on meeting the next day where keep brainstorming about remaining open issues  
* **Behavior Models Review / Open Questions Around Behavior**   
  [https://youtu.be/LZmEgcTsgUU](https://youtu.be/LZmEgcTsgUU) \- We review the mechanism we propose for modeling object behaviors and how it could map onto cortical anatomy. We then go through our list of open questions and discuss some further ideas around them.  
* **A Solution for Behavior in the Cortex** [https://youtu.be/BCXL2Ir\_qh4](https://youtu.be/BCXL2Ir_qh4) \- Viviane presents some new diagrams illustrating our theory and implementation so far, how the new ideas would extend them to model object behaviors, and how remaining questions could be solved. She starts out with an overview of the problem space and then presents solutions to each of the constraints we formulated. This is a good summary video to start with.  
* **Behavior Models in Monty & Potential Solutions to Open Questions**   
  [https://youtu.be/LocV1X0WH2E](https://youtu.be/LocV1X0WH2E) \- Viviane presents a more conceptual view of our proposed solution and how it would map to our implementation in Monty. She then suggests a potential solution to a big open question that remained at the end of the previous meeting (communicating location changes to make correct predictions about object morphology).


Start with the last two videos to get a big picture overview of where we are today. If you would like to follow along our journey and be a fly on the wall of how we got to this point, you can start at the beginning of the playlist, which is sorted chronologically.

# References

[1] Hawkins, J., & Ahmad, S. (2016). **Why neurons have thousands of synapses: A theory of sequence memory in neocortex.** Frontiers in Neural Circuits, 10, Article 23\. [https://doi.org/10.3389/fncir.2016.00023](https://doi.org/10.3389/fncir.2016.00023)  
[2] Hawkins, J., Ahmad, S., & Cui, Y. (2017). **A theory of how columns in the neocortex enable learning the structure of the world.** Frontiers in Neural Circuits, 11, Article 81\. [https://doi.org/10.3389/fncir.2017.00081](https://doi.org/10.3389/fncir.2017.00081)  
[3] Hawkins, J., Lewis, M., Klukas, M., Purdy, S., & Ahmad, S. (2019). **A framework for intelligence and cortical function based on grid cells in the neocortex.** Frontiers in Neural Circuits, 12, Article 121\. [https://doi.org/10.3389/fncir.2018.00121](https://doi.org/10.3389/fncir.2018.00121)  
[4] Hawkins, J., and Dawkins, R. (2021). **A Thousand Brains: A New Theory of Intelligence.** Basic Books. ISBN 9781541675810\. URL: [https://books.google.de/books?id=FQ-pzQEACAAJ](https://books.google.de/books?id=FQ-pzQEACAAJ).  
[5] Hawkins, J. C., Ahmad, S., Cui, Y., & Lewis, M. A. (2021). *U.S. Patent No. 10,977,566*. **Inferencing and learning based on sensorimotor input data.** Washington, DC: U.S. Patent and Trademark Office.  
[6] Hawkins, J. C., & Ahmad, S. (2024). *U.S. Patent No. 12,094,192*. **Inferencing and learning based on sensorimotor input data.** Washington, DC: U.S. Patent and Trademark Office.  
[7] Clay, V., Leadholm, N., and Hawkins, J. (2024). **The thousand brains project: A new paradigm for sensorimotor intelligence.** URL: [https://arxiv.org/abs/2412.18354](https://arxiv.org/abs/2412.18354).  
[8] Hawkins, J. C., Ahmad, S., Clay, V., & Leadholm, N. (2025). **Architecture and Operation of Intelligent System.** *U.S. Patent Application No. 18/751,199*.  
[9] **Monty code:** [https://github.com/thousandbrainsproject/tbp.monty](https://github.com/thousandbrainsproject/tbp.monty)  
[10] **TBP documentation**: [https://thousandbrainsproject.readme.io/docs/welcome-to-the-thousand-brains-project-documentation](https://thousandbrainsproject.readme.io/docs/welcome-to-the-thousand-brains-project-documentation) 
